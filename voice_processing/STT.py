import os
import sys
import time
from threading import Lock
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# Import parser AFTER to ensure clean initialization
from ai_module.parser import parse_command

# -------------------------------
# Configuration
# -------------------------------
is_processing = False
process_lock = Lock()
WAKE_WORD = "z"  # More distinctive than "z"
MODEL_SIZE = "medium"
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # Reduced for faster wake word detection
MAX_COMMAND_DURATION = 6.0
GRACE_PERIOD = 0.6
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 0.35

# Pre-allocate audio buffer to avoid repeated allocations
BUFFER_SIZE = int(SAMPLE_RATE * MAX_COMMAND_DURATION)
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# Load Whisper with optimized settings
print("Loading Whisper model...")
model = WhisperModel(
    MODEL_SIZE, 
    device="cuda",
    compute_type="int8_float16",  # Hybrid for better speed/accuracy balance
    num_workers=1,
    device_index=0,
    cpu_threads=2
)

# Warm up the model with dummy data to avoid first-run overhead
print("Warming up Whisper model...")
dummy_audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
_ = list(model.transcribe(dummy_audio, beam_size=1, language="en"))
print("Whisper loaded and warmed up on GPU")

# Warm up Llama parser (loads it into memory)
print("Warming up Llama parser...")
_ = parse_command("open chrome")
print(f"✓ All models ready!\nListening for wake word: '{WAKE_WORD}'...")

# -------------------------------
# Optimized transcription
# -------------------------------
def transcribe_audio(audio: np.ndarray, is_wake_word: bool = False) -> str:
    """Transcribe with different settings for wake-word vs command"""
    # Normalize audio to prevent volume issues
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    if is_wake_word:
        # Ultra-fast for wake word - minimal beam search
        segments, _ = model.transcribe(
            audio,
            beam_size=1,
            best_of=1,
            language="en",
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
            temperature=0.0
        )
    else:
        # Balanced settings for commands
        segments, _ = model.transcribe(
            audio,
            beam_size=3,
            best_of=2,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"threshold": 0.4, "min_silence_duration_ms": 300},
            word_timestamps=False,
            temperature=0.0
        )
    
    return " ".join(seg.text for seg in segments).strip()

# -------------------------------
# Optimized command recording with pre-allocated buffer
# -------------------------------
def record_command():
    print("🎤 Listening for command...")
    chunk_idx = 0
    silent_chunks = 0
    mini_chunk_duration = 0.12
    mini_chunk_samples = int(SAMPLE_RATE * mini_chunk_duration)
    max_chunks = int(MAX_COMMAND_DURATION / mini_chunk_duration)
    grace_chunks = int(GRACE_PERIOD / mini_chunk_duration)

    for i in range(max_chunks):
        chunk = sd.rec(
            mini_chunk_samples, 
            samplerate=SAMPLE_RATE,
            channels=1, 
            dtype='float32',
            blocking=True
        )
        
        audio_chunk = chunk[:, 0]
        
        # Copy into pre-allocated buffer
        start_idx = chunk_idx * mini_chunk_samples
        end_idx = start_idx + mini_chunk_samples
        audio_buffer[start_idx:end_idx] = audio_chunk
        chunk_idx += 1

        if i < grace_chunks:
            continue

        # Fast silence detection
        if np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks * mini_chunk_duration >= SILENCE_DURATION:
            break

    # Use only the recorded portion of the buffer
    audio = audio_buffer[:chunk_idx * mini_chunk_samples].copy()
    text = transcribe_audio(audio, is_wake_word=False)
    return text

# -------------------------------
# Wake-word detection callback
# -------------------------------
def callback(indata, frames, time_info, status):
    global is_processing
    
    if status:
        print(f"⚠️  Status: {status}")

    if is_processing:
        return

    audio = indata[:, 0].astype(np.float32)

    # Fast vectorized silence check
    if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
        return

    with process_lock:
        if is_processing:
            return
            
        # Fast wake-word detection
        spoken = transcribe_audio(audio, is_wake_word=True).lower()

        if WAKE_WORD in spoken:
            print(f"✓ Wake word detected!")
            is_processing = True
            
            start_time = time.time()
            
            try:
                # 1. Record command
                command_text = record_command()
                stt_time = time.time() - start_time
                print(f"💬 Command: '{command_text}' ({stt_time:.2f}s)")
                
                if not command_text or len(command_text) < 2:
                    print("⚠️  No command detected")
                    return
                
                # 2. Parse with Llama (sequential - important!)
                parse_start = time.time()
                result = parse_command(command_text)
                parse_time = time.time() - parse_start
                
                total_time = time.time() - start_time
                
                print(f"🤖 Parsed: {result}")
                print(f"⏱️  Timing: STT={stt_time:.2f}s, Parse={parse_time:.2f}s, Total={total_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                is_processing = False
                print(f"👂 Listening for '{WAKE_WORD}'...\n")

# -------------------------------
# Start listening
# -------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("🎙️  VOXEL VOICE ASSISTANT")
    print("=" * 50)
    print(f"Wake word: '{WAKE_WORD}'")
    print(f"GPU: RTX 4070 | Models: Whisper({MODEL_SIZE}) + Llama(7B)")
    print(f"Optimizations: Pre-allocated buffers, model warmup, batch processing")
    print("=" * 50)
    
    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=callback,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            latency='low'
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n👋 Voxel stopped.")