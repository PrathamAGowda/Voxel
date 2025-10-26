import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# -------------------------------
# Configuration
# -------------------------------
WAKE_WORD = "z"            # Wake word
MODEL_SIZE = "large-v2"          # medium for higher accuracy
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5           # seconds per wake-word chunk
MAX_COMMAND_DURATION = 8.0     # max seconds to listen after wake word
GRACE_PERIOD = 1.0             # seconds before checking silence
SILENCE_THRESHOLD = 0.01       # minimum amplitude to consider as speech
SILENCE_DURATION = 0.5         # stop after 0.5s of silence

# Load Faster-Whisper model once
model = WhisperModel(MODEL_SIZE, device="cuda")
print("Voxel is listening for wake word...")

# -------------------------------
# Function to record command after wake word
# -------------------------------
def record_command():
    print("Listening for command...")
    recorded_chunks = []
    silent_chunks = 0
    mini_chunk_duration = 0.2
    max_chunks = int(MAX_COMMAND_DURATION / mini_chunk_duration)
    grace_chunks = int(GRACE_PERIOD / mini_chunk_duration)

    for i in range(max_chunks):
        chunk = sd.rec(int(SAMPLE_RATE * mini_chunk_duration), samplerate=SAMPLE_RATE,
                       channels=1, dtype='float32')
        sd.wait()
        audio_chunk = chunk[:, 0]
        recorded_chunks.append(audio_chunk)

        # Skip silence detection for initial grace period
        if i < grace_chunks:
            continue

        # Check for silence after grace period
        if np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks * mini_chunk_duration >= SILENCE_DURATION:
            break

    audio = np.concatenate(recorded_chunks)
    # Transcribe with higher beam size, disable VAD to avoid trimming start/end
    segments, _ = model.transcribe(audio, beam_size=15, vad_filter=False, language="en")
    text = " ".join(segment.text for segment in segments)
    return text

# -------------------------------
# Wake-word detection callback
# -------------------------------
def callback(indata, frames, time, status):
    if status:
        print("Status:", status)

    audio = indata[:, 0].astype(np.float32)

    # Skip silent or very low-volume chunks
    if np.max(np.abs(audio)) < SILENCE_THRESHOLD:
        return

    # Disable VAD for wake-word detection to avoid cutting off
    segments, _ = model.transcribe(audio, beam_size=15, vad_filter=False, language="en")

    for segment in segments:
        spoken = segment.text.lower()
        print(f"[DEBUG] Recognized: '{spoken}'")  # Shows everything detected
        if WAKE_WORD in spoken:
            print(f"Wake word detected: '{WAKE_WORD}'")
            command_text = record_command()
            print("Command recognized:", command_text)

# -------------------------------
# Start microphone stream
# -------------------------------
with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback,
                    blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nVoxel stopped.")
