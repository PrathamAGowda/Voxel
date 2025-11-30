"""Real-time voice assistant with wake word detection, STT, and command execution."""

import os
import sys
import time
import difflib
from threading import Lock

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import ctranslate2 as ct2

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

import config as cfg
from ai_module.parser import parse_command

# ===== RUNTIME STATE =====
is_processing = False
process_lock = Lock()

# Wake buffer and state
WAKE_BUFFER_SAMPLES = int(cfg.STT_SAMPLE_RATE * cfg.WAKE_BUFFER_WINDOW_SEC)
wake_ring = np.zeros(WAKE_BUFFER_SAMPLES, dtype=np.float32)
wake_write_pos = 0
wake_confirm_hits = 0
_last_wake_decode_time = 0.0

# Vocabulary learning state
_vocab_hints = set()
_vocab_usage_count = {}

# ===== MODEL LOADING =====
def _load_whisper_model():
    """Load Whisper with GPU, fallback to lower precision if needed."""
    try:
        m = WhisperModel(
            cfg.STT_MODEL_SIZE,
            device=cfg.STT_DEVICE,
            compute_type=cfg.STT_COMPUTE_TYPE_PRIMARY,
            num_workers=cfg.STT_NUM_WORKERS,
            device_index=cfg.STT_DEVICE_INDEX,
            cpu_threads=cfg.STT_CPU_THREADS,
        )
        if cfg.LOG_GPU_STATUS:
            print(f"[STT] Loaded {cfg.STT_MODEL_SIZE} with {cfg.STT_COMPUTE_TYPE_PRIMARY} on {cfg.STT_DEVICE}")
        return m
    except Exception as e:
        if cfg.LOG_GPU_STATUS:
            print(f"[STT] {cfg.STT_COMPUTE_TYPE_PRIMARY} failed ({e.__class__.__name__}), trying {cfg.STT_COMPUTE_TYPE_FALLBACK}")
        m = WhisperModel(
            cfg.STT_MODEL_SIZE,
            device=cfg.STT_DEVICE,
            compute_type=cfg.STT_COMPUTE_TYPE_FALLBACK,
            num_workers=cfg.STT_NUM_WORKERS,
            device_index=cfg.STT_DEVICE_INDEX,
            cpu_threads=cfg.STT_CPU_THREADS,
        )
        if cfg.LOG_GPU_STATUS:
            print(f"[STT] Loaded {cfg.STT_MODEL_SIZE} with {cfg.STT_COMPUTE_TYPE_FALLBACK} on {cfg.STT_DEVICE}")
        return m

def _report_gpu_status():
    """Report GPU usage for Whisper."""
    if not cfg.LOG_GPU_STATUS:
        return
    try:
        cuda_count = ct2.get_cuda_device_count()
    except Exception:
        cuda_count = 0
    print(f"[GPU] Whisper CUDA devices: {cuda_count}; GPU: {'ON' if cuda_count > 0 else 'OFF'}")
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                name_str = name.decode() if isinstance(name, bytes) else str(name)
                print(f"[GPU] GPU{i}: {name_str} | VRAM: {mem.used/1e9:.2f}GB used")
        except Exception as e:
            print(f"[GPU] NVML unavailable: {e}")

print("[STT] Loading Whisper model...")
model = _load_whisper_model()
_report_gpu_status()

# Warm up
print("[STT] Warming up Whisper...")
dummy_audio = np.random.randn(cfg.STT_SAMPLE_RATE).astype(np.float32)
_ = list(model.transcribe(dummy_audio, beam_size=1, language="en", task="transcribe", vad_filter=False))
print("[STT] Whisper ready")

print("[STT] Warming up LLM parser...")
_ = parse_command("open chrome")
print(f"✓ All models ready!\nListening for wake word: '{cfg.WAKE_WORD}'...")

# ===== AUDIO PROCESSING =====
def _pre_emphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to boost high frequencies."""
    if signal.size <= 1:
        return signal
    out = np.empty_like(signal)
    out[0] = signal[0]
    out[1:] = signal[1:] - coef * signal[:-1]
    return out

def _stft(signal: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Short-time Fourier transform."""
    w = np.hanning(frame_len)
    windows = []
    for start in range(0, len(signal) - frame_len + 1, hop):
        frame = signal[start:start+frame_len] * w
        windows.append(np.fft.rfft(frame))
    return np.array(windows)

def _istft(spec: np.ndarray, frame_len: int, hop: int, total_len: int) -> np.ndarray:
    """Inverse short-time Fourier transform."""
    w = np.hanning(frame_len)
    out = np.zeros(total_len, dtype=np.float32)
    win_norm = np.zeros(total_len, dtype=np.float32)
    for i, frame_spec in enumerate(spec):
        start = i * hop
        frame = np.fft.irfft(frame_spec)
        out[start:start+frame_len] += frame * w
        win_norm[start:start+frame_len] += w**2
    win_norm[win_norm < 1e-8] = 1.0
    return out / win_norm

def _spectral_denoise(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Spectral gating noise suppression."""
    if not cfg.NOISE_SUPPRESSION_ENABLED or signal.size < sample_rate * 0.05:
        return signal
    
    frame_len = int(sample_rate * cfg.NS_FRAME_MS / 1000)
    frame_len = max(256, min(2048, frame_len))
    hop = int(frame_len * (1 - cfg.NS_OVERLAP))
    if hop <= 0:
        hop = frame_len // 2
    
    # Build noise profile from first portion
    init_len = int(sample_rate * cfg.NS_INIT_SECONDS)
    noise_region = signal[:init_len] if init_len < signal.size else signal
    spec_noise = _stft(noise_region, frame_len, hop)
    noise_mag = np.mean(np.abs(spec_noise), axis=0)
    
    # STFT full signal
    spec_full = _stft(signal, frame_len, hop)
    cleaned = []
    floor = cfg.NS_FLOOR
    atten = 10 ** (-cfg.NS_ATTEN_DB / 20.0)
    
    for mag_complex in spec_full:
        mag = np.abs(mag_complex)
        phase = np.angle(mag_complex)
        thresh = noise_mag * 1.25
        mask = mag < np.maximum(thresh, cfg.NS_MIN_ENERGY)
        new_mag = np.where(mask, np.maximum(mag * atten, noise_mag * floor), mag)
        cleaned.append(new_mag * np.exp(1j * phase))
    
    cleaned = np.array(cleaned)
    recon = _istft(cleaned, frame_len, hop, signal.size)
    
    # Safety normalize
    mx = np.max(np.abs(recon))
    if mx > 0:
        recon = recon / mx * np.max(np.abs(signal))
    return recon.astype(np.float32)

# ===== TRANSCRIPTION QUALITY =====
def _quality_is_poor(segments, info) -> bool:
    """Heuristic to detect low-quality transcriptions."""
    try:
        comp = getattr(info, "compression_ratio", 0.0) or 0.0
    except Exception:
        comp = 0.0
    min_lp = 0.0
    try:
        lps = [getattr(s, "avg_logprob", 0.0) for s in segments]
        if lps:
            min_lp = min(lps)
    except Exception:
        pass
    return (comp and comp > cfg.CMD_FALLBACK_COMPRESSION_THRESHOLD) or (min_lp < cfg.CMD_FALLBACK_LOGPROB_THRESHOLD)

# ===== VOCABULARY LEARNING =====
def _correct_with_vocab(text: str) -> str:
    """Post-correct transcription using learned vocabulary."""
    if not text or not cfg.VOCAB_POST_CORRECT or not _vocab_hints:
        return text
    words = text.split()
    corrected = []
    for w in words:
        if len(w) <= 2 or any(ch.isdigit() for ch in w):
            corrected.append(w)
            continue
        matches = difflib.get_close_matches(w.lower(), _vocab_hints, n=1, cutoff=cfg.VOCAB_MATCH_CUTOFF)
        if matches:
            m = matches[0]
            corrected.append(m if w.islower() else m.title())
        else:
            corrected.append(w)
    return " ".join(corrected)

def _learn_from_parse(result: dict):
    """Extract targets from parsed JSON and add to vocabulary."""
    global _vocab_hints, _vocab_usage_count
    if not isinstance(result, dict):
        return
    actions = result.get("actions", [])
    if not isinstance(actions, list):
        return
    for action in actions:
        if not isinstance(action, dict):
            continue
        target = action.get("target", "").strip()
        if target and len(target) > 2:
            target_lower = target.lower()
            _vocab_hints.add(target_lower)
            _vocab_usage_count[target_lower] = _vocab_usage_count.get(target_lower, 0) + 1
            # Prune if vocabulary grows too large
            if len(_vocab_hints) > cfg.VOCAB_MAX_SIZE:
                sorted_items = sorted(_vocab_usage_count.items(), key=lambda x: x[1], reverse=True)
                keep = set(item[0] for item in sorted_items[:cfg.VOCAB_PRUNE_TO])
                _vocab_hints = keep
                _vocab_usage_count = {k: v for k, v in _vocab_usage_count.items() if k in keep}

# ===== TRANSCRIPTION =====
def transcribe_audio(audio: np.ndarray, is_wake_word: bool = False) -> str:
    """Transcribe audio with noise suppression, pre-emphasis, and quality-based fallback."""
    # Noise suppression then pre-emphasis
    audio = _spectral_denoise(audio, cfg.STT_SAMPLE_RATE) if cfg.NOISE_SUPPRESSION_ENABLED else audio
    audio = _pre_emphasis(audio)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    if is_wake_word:
        segments, _ = model.transcribe(
            audio,
            beam_size=cfg.WAKE_BEAM_SIZE,
            best_of=cfg.WAKE_BEST_OF,
            language="en",
            task="transcribe",
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
            temperature=0.0,
            patience=cfg.WAKE_PATIENCE,
            no_speech_threshold=cfg.WAKE_NO_SPEECH_THRESHOLD,
        )
    else:
        vad_params = {
            "min_silence_duration_ms": cfg.CMD_VAD_MIN_SILENCE_MS,
            "speech_pad_ms": cfg.CMD_VAD_SPEECH_PAD_MS
        } if cfg.CMD_USE_VAD else None
        
        segments, info = model.transcribe(
            audio,
            beam_size=cfg.CMD_BEAM_SIZE,
            best_of=cfg.CMD_BEST_OF,
            language="en",
            task="transcribe",
            condition_on_previous_text=False,
            initial_prompt=cfg.CMD_INITIAL_PROMPT,
            vad_filter=cfg.CMD_USE_VAD,
            vad_parameters=vad_params,
            word_timestamps=False,
            temperature=0.0,
            compression_ratio_threshold=cfg.CMD_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=cfg.CMD_LOG_PROB_THRESHOLD,
            no_speech_threshold=cfg.CMD_NO_SPEECH_THRESHOLD,
            length_penalty=cfg.CMD_LENGTH_PENALTY,
            patience=cfg.CMD_PATIENCE,
        )
    
    text = " ".join(seg.text for seg in segments).strip()
    
    # Escalation fallback for command if quality is poor
    if not is_wake_word and cfg.CMD_FALLBACK_ENABLED and (not text or _quality_is_poor(segments, info)):
        try:
            # Second pass: deterministic escalation
            segments2, info2 = model.transcribe(
                audio,
                beam_size=cfg.CMD_FALLBACK2_BEAM,
                best_of=cfg.CMD_FALLBACK2_BEST_OF,
                language="en",
                task="transcribe",
                condition_on_previous_text=False,
                initial_prompt=cfg.CMD_INITIAL_PROMPT,
                vad_filter=cfg.CMD_USE_VAD,
                vad_parameters=vad_params,
                word_timestamps=False,
                temperature=0.0,
                length_penalty=cfg.CMD_LENGTH_PENALTY,
                patience=cfg.CMD_FALLBACK2_PATIENCE,
                no_speech_threshold=cfg.CMD_NO_SPEECH_THRESHOLD,
            )
            text2 = " ".join(seg.text for seg in segments2).strip()
            
            # Third pass: mild temperature if still poor
            if (not text2 or _quality_is_poor(segments2, info2)):
                segments3, info3 = model.transcribe(
                    audio,
                    beam_size=cfg.CMD_FALLBACK3_BEAM,
                    best_of=cfg.CMD_FALLBACK3_BEST_OF,
                    language="en",
                    task="transcribe",
                    condition_on_previous_text=False,
                    initial_prompt=cfg.CMD_INITIAL_PROMPT,
                    vad_filter=cfg.CMD_USE_VAD,
                    vad_parameters=vad_params,
                    word_timestamps=False,
                    temperature=cfg.CMD_FALLBACK3_TEMPERATURE,
                    length_penalty=cfg.CMD_LENGTH_PENALTY,
                    patience=cfg.CMD_FALLBACK3_PATIENCE,
                    no_speech_threshold=cfg.CMD_NO_SPEECH_THRESHOLD,
                )
                text3 = " ".join(seg.text for seg in segments3).strip()
                if text3 and (not text2 or _quality_is_poor(segments2, info2)):
                    text2 = text3
            
            if text2 and (_quality_is_poor(segments, info) and not _quality_is_poor(segments2, info2)):
                text = text2
        except Exception:
            pass
    
    # Vocabulary-based correction
    if not is_wake_word and cfg.VOCAB_POST_CORRECT:
        text = _correct_with_vocab(text)
    
    return text

# ===== WAKE WORD DETECTION =====
def _wake_buffer_add(chunk: np.ndarray):
    """Add audio chunk to ring buffer."""
    global wake_write_pos
    n = len(chunk)
    if n >= WAKE_BUFFER_SAMPLES:
        wake_ring[:] = chunk[-WAKE_BUFFER_SAMPLES:]
        wake_write_pos = 0
        return
    end = wake_write_pos + n
    if end <= WAKE_BUFFER_SAMPLES:
        wake_ring[wake_write_pos:end] = chunk
    else:
        first = WAKE_BUFFER_SAMPLES - wake_write_pos
        wake_ring[wake_write_pos:] = chunk[:first]
        wake_ring[: n - first] = chunk[first:]
    wake_write_pos = (wake_write_pos + n) % WAKE_BUFFER_SAMPLES

def _attempt_wake_decode(now_ts: float) -> str | None:
    """Attempt wake word decode from buffered audio."""
    global _last_wake_decode_time
    if now_ts - _last_wake_decode_time < 0.2:  # rate-limit to ~5Hz
        return None
    _last_wake_decode_time = now_ts
    
    # Energy gate
    if float(np.max(np.abs(wake_ring))) < cfg.WAKE_SPEECH_ENERGY_THRESHOLD:
        return None
    
    audio = wake_ring.copy()
    mx = float(np.max(np.abs(audio)))
    if mx > 0:
        audio /= mx
    
    segments, _ = model.transcribe(
        audio,
        beam_size=4,
        best_of=4,
        language="en",
        task="transcribe",
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
        temperature=0.0,
        no_speech_threshold=0.6,
    )
    
    text = " ".join(s.text for s in segments).strip().lower()
    if cfg.WAKE_VERBOSE_MONITOR and text:
        print(f"[WAKE_MONITOR] Buffered: '{text}'")
    
    if not text:
        return None
    
    t = text.replace(".", " ").replace(",", " ")
    t = " ".join(t.split())
    
    # Reject phrases
    for phrase in cfg.WAKE_REJECT_PHRASES:
        if phrase in t:
            return None
    
    # Match wake word or aliases
    toks = set(t.split())
    if cfg.WAKE_ALIASES & toks:
        return t
    return None

# ===== COMMAND RECORDING =====
def record_command():
    """Record audio until silence detected."""
    print("🎤 Listening for command...")
    mini_chunk_samples = int(cfg.STT_SAMPLE_RATE * cfg.CMD_MINI_CHUNK_MS / 1000)
    max_chunks = int(cfg.CMD_MAX_DURATION_SEC / (cfg.CMD_MINI_CHUNK_MS / 1000))
    
    recorded = []
    silence_accum = 0.0
    
    for _ in range(max_chunks):
        chunk = sd.rec(
            mini_chunk_samples,
            samplerate=cfg.STT_SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocking=True
        )
        audio_chunk = chunk[:, 0]
        recorded.append(audio_chunk)
        
        max_amp = float(np.max(np.abs(audio_chunk)))
        if max_amp < cfg.CMD_SILENCE_THRESHOLD:
            silence_accum += cfg.CMD_MINI_CHUNK_MS / 1000
            if silence_accum >= cfg.CMD_MIN_SILENCE_SEC:
                break
        else:
            silence_accum = 0.0
    
    audio = np.concatenate(recorded) if recorded else np.zeros(1, dtype=np.float32)
    return transcribe_audio(audio, is_wake_word=False)

# ===== MAIN CALLBACK =====
def callback(indata, frames, time_info, status):
    """Audio stream callback for wake word detection."""
    global is_processing, wake_confirm_hits
    
    if status:
        print(f"⚠️  Status: {status}")
    
    if is_processing:
        return
    
    audio = indata[:, 0].astype(np.float32)
    _wake_buffer_add(audio)
    
    # Skip decode if chunk is silence
    if float(np.max(np.abs(audio))) < cfg.CMD_SILENCE_THRESHOLD:
        return
    
    detected = _attempt_wake_decode(time.time())
    if detected:
        wake_confirm_hits += 1
        if cfg.WAKE_VERBOSE_MONITOR:
            print(f"[WAKE_MONITOR] Possible wake hit ({wake_confirm_hits}/{cfg.WAKE_CONFIRM_REQUIRED}): '{detected}'")
    else:
        if wake_confirm_hits > 0:
            wake_confirm_hits -= 1
    
    if wake_confirm_hits >= cfg.WAKE_CONFIRM_REQUIRED:
        wake_confirm_hits = 0
        print("✓ Wake word confirmed!")
        
        with process_lock:
            if is_processing:
                return
            is_processing = True
        
        start_time = time.time()
        try:
            command_text = record_command()
            stt_time = time.time() - start_time
            print(f"💬 Command: '{command_text}' ({stt_time:.2f}s)")
            
            if not command_text or len(command_text) < 2:
                print("⚠️  No command detected")
            else:
                parse_start = time.time()
                result = parse_command(command_text)
                parse_time = time.time() - parse_start
                
                if cfg.LOG_TIMING:
                    print(f"🤖 Parsed: {result}")
                    print(f"⏱️  Timing: STT={stt_time:.2f}s, Parse={parse_time:.2f}s, Total={time.time()-start_time:.2f}s")
                
                # Learn from successful parse
                if cfg.VOCAB_LEARNING_ENABLED and result and isinstance(result, dict):
                    _learn_from_parse(result)
                
                # Execute parsed actions
                if isinstance(result, dict) and "actions" in result:
                    exec_start = time.time()
                    exec_out = execute_actions(result["actions"])
                    exec_time = time.time() - exec_start
                    
                    if cfg.LOG_EXECUTION_RESULTS:
                        print(f"⚙️  Execution: {exec_out}")
                    if cfg.LOG_TIMING:
                        print(f"⏱️  Exec: {exec_time:.2f}s (Total: {time.time()-start_time:.2f}s)")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            is_processing = False
            print(f"👂 Listening for '{cfg.WAKE_WORD}'...\n")

# ===== MAIN ENTRY =====
if __name__ == "__main__":
    print("=" * 50)
    print("🎙️  VOXEL VOICE ASSISTANT (OPTIMIZED)")
    print("=" * 50)
    print(f"Wake word: '{cfg.WAKE_WORD}'")
    print(f"Models: Whisper({cfg.STT_MODEL_SIZE}) + Llama({cfg.LLM_MODEL_NAME})")
    print(f"Features: Noise suppression, VAD, vocabulary learning, fallback decoding")
    print("=" * 50)
    
    try:
        with sd.InputStream(
            channels=1,
            samplerate=cfg.STT_SAMPLE_RATE,
            callback=callback,
            blocksize=int(cfg.STT_SAMPLE_RATE * cfg.CMD_CHUNK_DURATION_SEC),
            latency='low'
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Voxel stopped.")
