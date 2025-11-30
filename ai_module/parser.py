"""Parser module for converting natural language commands to structured JSON.
Uses Mistral 7B Instruct GGUF with lazy initialization and adaptive GPU offload.
"""

from llama_cpp import Llama, llama_print_system_info
import json
import re
import os
import time
import sys

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
try:
    import pynvml
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# Validate model exists
if not cfg.LLM_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Required model not found: {cfg.LLM_MODEL_PATH}\n"
        f"Please place '{cfg.LLM_MODEL_NAME}' in {cfg.MODELS_DIR}"
    )

_llm = None
_requested_gpu_layers = None
_used_gpu_layers = 0
_llm_gpu_on = False
_build_gpu_capable = False
_system_flags = ""

def _examine_build_flags():
    global _build_gpu_capable, _system_flags
    try:
        _system_flags = llama_print_system_info().decode(errors="ignore")
        if ("CUBLAS" in _system_flags or "CUDA" in _system_flags):
            _build_gpu_capable = True
    except Exception:
        _system_flags = "<unavailable>"

def _nvml_snapshot():
    if not _HAS_NVML:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mi = pynvml.nvmlDeviceGetMemoryInfo(h)
        return mi
    except Exception:
        return None

def _try_init_llm(n_gpu_layers: int):
    """Attempt to initialize Llama with given GPU layers."""
    try:
        return Llama(
            model_path=str(cfg.LLM_MODEL_PATH),
            n_ctx=cfg.LLM_N_CTX,
            n_threads=cfg.LLM_N_THREADS,
            n_gpu_layers=n_gpu_layers,
            n_batch=cfg.LLM_N_BATCH,
            verbose=False,
            use_mmap=True,
            use_mlock=False,
            mul_mat_q=True,
            logits_all=False,
        ), n_gpu_layers
    except Exception:
        return None, 0

def _init_llm_if_needed():
    """Lazy LLM initialization with adaptive GPU layer selection."""
    global _llm, _used_gpu_layers, _requested_gpu_layers, _llm_gpu_on
    if _llm is not None:
        return

    start = time.time()
    _examine_build_flags()

    # Priority: env override, then default, then fallback ladder
    env_layers = os.getenv("LLAMA_N_GPU_LAYERS")
    if env_layers is not None:
        try:
            candidates = [int(env_layers)]
        except ValueError:
            candidates = [cfg.LLM_GPU_LAYERS_DEFAULT] + cfg.LLM_GPU_LAYERS_FALLBACK
    else:
        candidates = [cfg.LLM_GPU_LAYERS_DEFAULT] + cfg.LLM_GPU_LAYERS_FALLBACK

    for c in candidates:
        llm_try, used = _try_init_llm(c)
        if llm_try is not None:
            _llm = llm_try
            _used_gpu_layers = used
            _requested_gpu_layers = c
            _llm_gpu_on = (used != 0)
            break

    if _llm is None:
        raise RuntimeError("Failed to initialize LLM with any GPU layer configuration.")

    init_ms = (time.time() - start) * 1000

    if cfg.LOG_GPU_STATUS:
        torch_hint = None
        if _HAS_TORCH:
            try:
                torch_hint = torch.cuda.is_available()
            except Exception:
                pass
        mi = _nvml_snapshot()
        vram = f"{mi.used/1e9:.2f}GB" if mi else "N/A"
        print(
            f"[LLM] Loaded {cfg.LLM_MODEL_PATH.name}\n"
            f"  Build: {_system_flags.strip().splitlines()[0] if _system_flags else 'UNKNOWN'}\n"
            f"  GPU: {'ON' if _llm_gpu_on else 'CPU ONLY'} | Requested: {_requested_gpu_layers} | Active: {_used_gpu_layers}\n"
            f"  Torch CUDA: {torch_hint} | VRAM: {vram} | Init: {init_ms:.0f}ms"
        )

# ULTRA-COMPACT prompt (fewer tokens = faster)
SYSTEM_PROMPT = """Convert commands to JSON: {"actions": [{"action": "ACTION", "target": "TARGET", "parameters": {}}]}

Actions: open_app, close_app, search, play_music
Multi-commands: separate by "and"/"then"/","

Examples:
"open chrome" -> {"actions": [{"action": "open_app", "target": "Chrome", "parameters": {}}]}
"open spotify and chrome" -> {"actions": [{"action": "open_app", "target": "Spotify", "parameters": {}}, {"action": "open_app", "target": "Chrome", "parameters": {}}]}"""

def parse_command(command_text: str) -> dict:
    """Parse natural language command into structured JSON actions."""
    _init_llm_if_needed()
    
    phrases = split_command(command_text)
    
    if len(phrases) > 1:
        prompt = f"""{SYSTEM_PROMPT}

Command: "{command_text}" ({len(phrases)} actions)
JSON:"""
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nCommand: {command_text}\nJSON:"
    
    response = _llm(
        prompt, 
        max_tokens=cfg.LLM_MAX_TOKENS,
        temperature=cfg.LLM_TEMPERATURE,
        top_p=cfg.LLM_TOP_P,
        top_k=cfg.LLM_TOP_K,
        repeat_penalty=cfg.LLM_REPEAT_PENALTY,
        stop=cfg.LLM_STOP_TOKENS,
        echo=False
    )
    
    text = response["choices"][0]["text"].strip()
    
    try:
        parsed = extract_json(text)
        # Validate multi-command completeness
        if len(phrases) > 1 and len(parsed.get("actions", [])) < len(phrases):
            return construct_from_phrases(phrases)
        return parsed
    except Exception:
        return construct_from_phrases(phrases)

def split_command(text: str) -> list:
    """Fast command splitting"""
    text = text.strip().lower()
    
    # Quick check for separators
    for sep in [' and ', ' then ', ', and ', ', then ', ', ']:
        if sep in text:
            parts = text.split(sep)
            return [p.strip() for p in parts if p.strip()]
    
    return [text]

def construct_from_phrases(phrases: list) -> dict:
    """Lightning-fast fallback parser"""
    actions = []
    
    for phrase in phrases:
        p = phrase.lower().strip()
        
        # Pattern matching with early returns
        if "open" in p:
            target = p.replace("open", "").strip().title()
            actions.append({"action": "open_app", "target": target, "parameters": {}})
        elif "close" in p:
            target = p.replace("close", "").strip().title()
            actions.append({"action": "close_app", "target": target, "parameters": {}})
        elif "play" in p:
            actions.append({"action": "play_music", "target": "", "parameters": {}})
        elif "search" in p:
            query = p.replace("search", "").replace("for", "").strip()
            actions.append({"action": "search", "target": "Google", "parameters": {"query": query}})
        else:
            actions.append({"action": "unknown", "target": phrase, "parameters": {}})
    
    return {"actions": actions}

def extract_json(text: str) -> dict:
    """Fast JSON extraction"""
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start == -1 or end <= start:
        raise ValueError("No JSON found")
    
    json_str = text[start:end]
    
    # Quick cleanup
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    parsed = json.loads(json_str)
    
    # Fast structure validation
    if "actions" not in parsed:
        parsed = {"actions": [parsed] if "action" in parsed else []}
    
    if not isinstance(parsed["actions"], list):
        parsed["actions"] = [parsed["actions"]]
    
    # Ensure fields exist
    for action in parsed["actions"]:
        action.setdefault("action", "unknown")
        action.setdefault("target", "")
        action.setdefault("parameters", {})
    
    return parsed

# Lazy init mode: model loads on first parse() to avoid VRAM stacking with Whisper.
if cfg.LOG_GPU_STATUS:
    print("[LLM] Parser ready (lazy init). Model will load on first parse().")