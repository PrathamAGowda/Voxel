"""
Central configuration for Voxel Voice Assistant.
All tunable parameters for STT, LLM, and execution are defined here.
Override via environment variables or by editing this file directly.
"""

import os
from pathlib import Path

# ===== PROJECT PATHS =====
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

# ===== LLM PARSER CONFIG =====
LLM_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
LLM_MODEL_PATH = MODELS_DIR / LLM_MODEL_NAME
LLM_N_CTX = 512
LLM_N_THREADS = 4
LLM_N_BATCH = 512
LLM_MAX_TOKENS = 120
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 0.9
LLM_TOP_K = 20
LLM_REPEAT_PENALTY = 1.1
LLM_STOP_TOKENS = ["\n\n", "Command:", "Example:"]

# GPU layers: override via LLAMA_N_GPU_LAYERS env. -1 = auto offload.
LLM_GPU_LAYERS_DEFAULT = -1
LLM_GPU_LAYERS_FALLBACK = [48, 40, 32, 24, 16, 8, 0]

# ===== STT (WHISPER) CONFIG =====
STT_MODEL_SIZE = "large-v3"
STT_DEVICE = "cuda"
STT_COMPUTE_TYPE_PRIMARY = "float16"
STT_COMPUTE_TYPE_FALLBACK = "int8_float16"
STT_NUM_WORKERS = 2
STT_DEVICE_INDEX = 0
STT_CPU_THREADS = 2
STT_SAMPLE_RATE = 16000

# Wake word detection
WAKE_WORD = "spot"
WAKE_ALIASES = {"spot"}  # normalized; add phonetic variants if needed
WAKE_REJECT_PHRASES = {"rejected", "thank you"}
WAKE_CONFIRM_REQUIRED = 1
WAKE_BUFFER_WINDOW_SEC = 0.8
WAKE_SPEECH_ENERGY_THRESHOLD = 0.02
WAKE_BEAM_SIZE = 16
WAKE_BEST_OF = 4
WAKE_PATIENCE = 1.0
WAKE_NO_SPEECH_THRESHOLD = 0.55
WAKE_VERBOSE_MONITOR = True

# Command recording
CMD_MAX_DURATION_SEC = 8.0
CMD_MIN_SILENCE_SEC = 0.6
CMD_SILENCE_THRESHOLD = 0.015
CMD_CHUNK_DURATION_SEC = 0.2
CMD_MINI_CHUNK_MS = 40

# Command transcription (primary pass)
CMD_BEAM_SIZE = 16
CMD_BEST_OF = 6
CMD_PATIENCE = 1.3
CMD_LENGTH_PENALTY = 0.1
CMD_USE_VAD = True
CMD_VAD_MIN_SILENCE_MS = 400
CMD_VAD_SPEECH_PAD_MS = 120
CMD_NO_SPEECH_THRESHOLD = 0.5
CMD_COMPRESSION_RATIO_THRESHOLD = 2.4
CMD_LOG_PROB_THRESHOLD = -0.2

# Command transcription quality-based fallback
CMD_FALLBACK_ENABLED = True
CMD_FALLBACK_COMPRESSION_THRESHOLD = 2.1
CMD_FALLBACK_LOGPROB_THRESHOLD = -0.9
# Second pass (deterministic escalation)
CMD_FALLBACK2_BEAM = 18
CMD_FALLBACK2_BEST_OF = 6
CMD_FALLBACK2_PATIENCE = 1.3
# Third pass (mild temperature)
CMD_FALLBACK3_BEAM = 20
CMD_FALLBACK3_BEST_OF = 7
CMD_FALLBACK3_PATIENCE = 1.35
CMD_FALLBACK3_TEMPERATURE = 0.2

# Initial prompt bias
CMD_INITIAL_PROMPT = (
    "Transcribe concise English voice commands for a desktop assistant. "
    "Prefer imperative phrasing (open chrome, close window, play music, search web, "
    "scroll down, volume up, brightness 50 percent, take screenshot). "
    "Do not add filler words. Use standard words, not phonetic spellings."
)

# ===== NOISE SUPPRESSION =====
NOISE_SUPPRESSION_ENABLED = True
NS_FRAME_MS = 20
NS_OVERLAP = 0.5
NS_INIT_SECONDS = 0.4
NS_FLOOR = 0.12
NS_ATTEN_DB = 18
NS_MIN_ENERGY = 1e-8

# ===== VOCABULARY LEARNING =====
VOCAB_LEARNING_ENABLED = True
VOCAB_POST_CORRECT = True
VOCAB_MAX_SIZE = 150
VOCAB_PRUNE_TO = 100
VOCAB_MATCH_CUTOFF = 0.82

# ===== EXECUTION CONFIG =====
# Extend app registry in system_control/command_executor.py as needed.

# ===== LOGGING / DEBUG =====
LOG_GPU_STATUS = True
LOG_TIMING = True
LOG_EXECUTION_RESULTS = True
