import json
from pathlib import Path
from .logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_INJECTION_TEMPLATE = """[klyd] Architectural decisions governing files in this session:

{decisions}

Do not contradict these decisions unless the user explicitly instructs you to change them.
"""

def get_config_path():
    return Path('.klyd') / 'config.json'

def init_config():
    klyd_dir = Path('.klyd')
    klyd_dir.mkdir(exist_ok=True)
    
    config_path = get_config_path()
    if not config_path.exists():
        default = {
            "injection_template": DEFAULT_INJECTION_TEMPLATE,
            "strict_mode": False,
            "pinned_decision_ids": [],
            "max_decisions_inject": 10,
            "min_confidence": "LOW",
            "module_filter": []
        }
        with open(config_path, 'w') as f:
            json.dump(default, f, indent=2)
        logger.info("Initialized config file", extra={'path': str(config_path)})

def set_config(key, value):
    init_config()
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        data = json.load(f)
    data[key] = value
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.debug("Set config key", extra={'key': key})

def get_config(key, default=None):
    config_path = get_config_path()
    if not config_path.exists():
        return default
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        return data.get(key, default)
    except:
        return default

def get_all_config():
    config_path = get_config_path()
    if not config_path.exists():
        return {}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def get_injection_template() -> str:
    return get_config('injection_template', DEFAULT_INJECTION_TEMPLATE)

def set_injection_template(template: str):
    set_config('injection_template', template)

def get_strict_mode() -> bool:
    return get_config('strict_mode', False)

def set_strict_mode(mode: bool):
    set_config('strict_mode', mode)

def get_pinned_decision_ids() -> list[int]:
    return get_config('pinned_decision_ids', [])

def set_pinned_decision_ids(ids: list[int]):
    set_config('pinned_decision_ids', ids)

def add_pinned_decision_id(decision_id: int):
    ids = get_pinned_decision_ids()
    if decision_id not in ids:
        ids.append(decision_id)
        set_pinned_decision_ids(ids)

def remove_pinned_decision_id(decision_id: int):
    ids = get_pinned_decision_ids()
    if decision_id in ids:
        ids.remove(decision_id)
        set_pinned_decision_ids(ids)

def clear_pinned_decision_ids():
    set_pinned_decision_ids([])

def get_max_decisions_inject() -> int:
    return get_config('max_decisions_inject', 10)

def set_max_decisions_inject(n: int):
    set_config('max_decisions_inject', n)

def get_min_confidence() -> str:
    return get_config('min_confidence', 'LOW')

def set_min_confidence(level: str):
    set_config('min_confidence', level.upper())

def get_module_filter() -> list[str]:
    return get_config('module_filter', [])

def set_module_filter(modules: list[str]):
    set_config('module_filter', modules)
