# ai_module/llama_engine/parser.py
from llama_cpp import Llama
import json
import re

# Optimized Llama initialization
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=1024,  # Increased for longer multi-command inputs
    n_threads=2,
    n_gpu_layers=-1,
    n_batch=512,
    verbose=False,
    use_mmap=True,
    use_mlock=False,
    mul_mat_q=True,
    logits_all=False,
)

# Improved system prompt with better examples for multiple commands
SYSTEM_PROMPT = """You are a command parser. Convert user commands to JSON format.

Format: {"actions": [{"action": "ACTION", "target": "TARGET", "parameters": {}}]}

Common actions: open_app, close_app, search, play_music, minimize, maximize

Rules:
1. If command has "and", "then", or commas, create MULTIPLE action objects
2. Each action gets its own object in the actions array
3. Return ONLY valid JSON, no explanations

Examples:
"open chrome" -> {"actions": [{"action": "open_app", "target": "Chrome", "parameters": {}}]}

"open spotify and chrome" -> {"actions": [{"action": "open_app", "target": "Spotify", "parameters": {}}, {"action": "open_app", "target": "Chrome", "parameters": {}}]}

"open firefox then play music" -> {"actions": [{"action": "open_app", "target": "Firefox", "parameters": {}}, {"action": "play_music", "target": "", "parameters": {}}]}"""

def preprocess_command(text: str) -> list:
    """Split command by separators"""
    text = text.strip()
    
    # Split on common separators
    separators = [' and ', ', and ', ' then ', ', then ', ', ']
    
    for sep in separators:
        if sep in text.lower():
            # Case-insensitive split but preserve original case
            parts = re.split(re.escape(sep), text, flags=re.IGNORECASE)
            return [p.strip() for p in parts if p.strip()]
    
    return [text]

def parse_command(command_text: str) -> dict:
    """Main parsing function"""
    
    # Check if it's a multi-command first
    phrases = preprocess_command(command_text)
    
    if len(phrases) > 1:
        # Explicitly tell the model there are multiple commands
        prompt = f"""{SYSTEM_PROMPT}

Command: "{command_text}"

This command contains {len(phrases)} separate actions: {', '.join([f'"{p}"' for p in phrases])}

JSON:"""
    else:
        prompt = f"""{SYSTEM_PROMPT}

Command: {command_text}
JSON:"""
    
    # Generate with settings optimized for accuracy on multi-commands
    response = llm(
        prompt, 
        max_tokens=300,  # Increased for multiple actions
        temperature=0.0,  # Deterministic
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.15,  # Prevent repeating same action
        stop=["\n\n", "Command:", "User:", "Example:", "---"],
        echo=False
    )
    
    text = response["choices"][0]["text"].strip()
    
    # Try to extract and validate JSON
    try:
        parsed = extract_json(text)
        
        # Validate we got the right number of actions for multi-commands
        if len(phrases) > 1 and len(parsed.get("actions", [])) < len(phrases):
            print(f"⚠️  Expected {len(phrases)} actions but got {len(parsed['actions'])}. Retrying with explicit format...")
            # Fallback: construct manually from phrases
            return construct_from_phrases(phrases)
        
        return parsed
        
    except Exception as e:
        print(f"Parse error: {e}, attempting fallback...")
        return construct_from_phrases(phrases)

def construct_from_phrases(phrases: list) -> dict:
    """Fallback: construct actions from split phrases using simple rules"""
    actions = []
    
    for phrase in phrases:
        phrase_lower = phrase.lower().strip()
        
        # Simple pattern matching
        if "open" in phrase_lower:
            target = phrase_lower.replace("open", "").strip()
            actions.append({
                "action": "open_app",
                "target": target.title(),
                "parameters": {}
            })
        elif "close" in phrase_lower:
            target = phrase_lower.replace("close", "").strip()
            actions.append({
                "action": "close_app",
                "target": target.title(),
                "parameters": {}
            })
        elif "play" in phrase_lower and "music" in phrase_lower:
            actions.append({
                "action": "play_music",
                "target": "",
                "parameters": {}
            })
        elif "search" in phrase_lower:
            query = phrase_lower.replace("search", "").replace("for", "").strip()
            actions.append({
                "action": "search",
                "target": "Google",
                "parameters": {"query": query}
            })
        else:
            # Unknown command
            actions.append({
                "action": "unknown",
                "target": phrase,
                "parameters": {}
            })
    
    return {"actions": actions}

def extract_json(text: str) -> dict:
    """Robust JSON extraction"""
    # Find JSON object
    start = text.find('{')
    end = text.rfind('}') + 1
    
    if start != -1 and end > start:
        json_str = text[start:end]
        
        # Clean common issues
        json_str = re.sub(r',\s*}', '}', json_str)  # Trailing commas in objects
        json_str = re.sub(r',\s*]', ']', json_str)  # Trailing commas in arrays
        
        try:
            parsed = json.loads(json_str)
            
            # Ensure proper structure
            if "actions" not in parsed:
                if "action" in parsed:
                    parsed = {"actions": [parsed]}
                else:
                    raise ValueError("No actions found")
            
            # Validate actions is a list
            if not isinstance(parsed["actions"], list):
                parsed["actions"] = [parsed["actions"]]
            
            # Ensure each action has required fields
            for action in parsed["actions"]:
                if "action" not in action:
                    action["action"] = "unknown"
                if "target" not in action:
                    action["target"] = ""
                if "parameters" not in action:
                    action["parameters"] = {}
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parse failed: {e}")
            raise
    
    raise ValueError("No valid JSON found")

# Warm up the model
print("Warming up Llama parser...")
_ = parse_command("open chrome")
_ = parse_command("open chrome and spotify")  # Test multi-command warmup
print("✓ Llama parser ready")