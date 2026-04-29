import json
import urllib.request
from anthropic import Anthropic
from .db import get_existing_decisions_for_files

def _call_openai_compatible(url, key, model, prompt):
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    })
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        return res['choices'][0]['message']['content']

def extract_decisions(diff, commit_message, existing_decisions, config_data, model='claude-sonnet-4-6'):
    files = []
    for line in diff.split('\n'):
        if line.startswith('diff --git a/'):
            parts = line.split(' b/')
            if len(parts) == 2:
                files.append(parts[1].strip())
    
    invariants = get_existing_decisions_for_files(list(set(files)))
    invariants_text = "\n".join([f"- [{inv['id']}] {inv['decision']}" for inv in invariants]) if invariants else "None"

    prompt = f"""You are an architectural decision extractor for a software project.

You will receive a git diff, a commit message, and a list of previously recorded architectural decisions for the files touched in this diff.

Your job: extract the literal text of the architectural decisions that this commit clearly enacts.

Architectural decisions are: data store choice, auth strategy, API boundary contracts, module responsibility assignments, dependency/library choices, error handling patterns.

Rules:
- Only record if the diff explicitly introduces, changes, or contradicts a decision
- Do not guess. Do not infer. Only record what is clearly shown in the diff.
- If the commit is a style fix, test update, or minor refactor with no architectural significance: return []
- For each decision, classify as NEW (not seen before), REINFORCE (confirms an existing decision), or CONTRADICT (conflicts with an existing decision)
- Assign confidence: HIGH (unmistakable from diff), MEDIUM (clear but could have context), LOW (possible but uncertain)
- If this commit contradicts any of the listed invariants, set event_type to 'CONTRADICT' and add a note in the 'decision' field describing the violation. Otherwise, proceed with NEW or REINFORCE as before.

Return ONLY valid JSON. No prose. No markdown. No explanation.
- The 'decision' field must contain the exact architectural rule or invariant written in the diff, quoted verbatim if possible. Do not summarise module responsibilities. For example, prefer 'Authentication uses JWT access tokens exclusively; no server-side sessions' over 'Authentication is handled by auth.py'.
Schema: [{{"decision": str, "module": str, "file_patterns": str, "confidence": "LOW"|"MEDIUM"|"HIGH", "event": "NEW"|"REINFORCE"|"CONTRADICT"}}]
If no decisions: return []

EXISTING DECISIONS FOR TOUCHED FILES:
{existing_decisions}

COMMIT MESSAGE:
{commit_message}

DIFF:
{diff}

Existing architectural invariants (do not violate unless explicitly instructed):
{invariants_text}
"""
    try:
        # Determine provider
        if model.startswith('claude-') or 'api_key' in config_data and not any(k in config_data for k in ['openai_key', 'openrouter_key', 'gemini_key', 'groq_key']):
            if 'api_key' not in config_data:
                raise ValueError("Anthropic API key is required.")
            client = Anthropic(api_key=config_data['api_key'])
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
        else:
            # OpenAI compatible providers
            if model.startswith('gpt-') or model.startswith('o1') or model.startswith('o3'):
                if 'openai_key' not in config_data: raise ValueError("OpenAI API key missing.")
                url, key = "https://api.openai.com/v1/chat/completions", config_data['openai_key']
            elif model.startswith('gemini-') or model.startswith('gemma-'):
                if 'gemini_key' not in config_data: raise ValueError("Gemini API key missing.")
                url, key = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", config_data['gemini_key']
            elif '/' in model:
                if 'openrouter_key' not in config_data: raise ValueError("OpenRouter API key missing.")
                url, key = "https://openrouter.ai/api/v1/chat/completions", config_data['openrouter_key']
            elif config_data.get('groq_key'):
                url, key = "https://api.groq.com/openai/v1/chat/completions", config_data['groq_key']
            elif config_data.get('openai_key'):
                url, key = "https://api.openai.com/v1/chat/completions", config_data['openai_key']
            elif config_data.get('openrouter_key'):
                url, key = "https://openrouter.ai/api/v1/chat/completions", config_data['openrouter_key']
            else:
                raise ValueError(f"No valid API key configured for model: {model}")
                
            content = _call_openai_compatible(url, key, model, prompt)
            
        # Parse output as JSON
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
            
        content = content.strip()
        
        result = json.loads(content)
        if not isinstance(result, list):
            return []
        
        normalized = []
        for r in result:
            event_val = r.get('event') or r.get('event_type') or 'NEW'
            normalized.append({
                'decision': r.get('decision', 'Unknown'),
                'module': r.get('module', '/'),
                'file_patterns': r.get('file_patterns', '*'),
                'confidence': r.get('confidence', 'LOW'),
                'event_type': event_val
            })
            
        return normalized

    except Exception as e:
        raise
