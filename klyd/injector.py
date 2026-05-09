def format_injection(decisions):
    if not decisions:
        return ""

    lines = [
        "[klyd] Architectural decisions governing files in this session:\n"
    ]
    
    for i, d in enumerate(decisions, 1):
        mod = f"[{d['module']}]"
        conf = f"{d['confidence']} confidence"
        count = f"confirmed {d['reinforcement_count']} times"
        dec = d['decision'].rstrip('.')  # remove trailing period if present
        lines.append(f"{i}. {mod} {dec}. ({conf}, {count})")
        
    lines.append("\nDo not contradict these decisions unless the user explicitly instructs you to change them.")
    
    return "\n".join(lines)
