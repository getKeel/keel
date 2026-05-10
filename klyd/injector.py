import fnmatch
from .db import (
    compute_embedding,
    get_embedding_for_decision,
    get_reinforcement_recency_score,
    _cosine_similarity
)

def _file_pattern_match(file_patterns: str, file_list: list[str]) -> float:
    """Return 1.0 if any file matches any pattern, else 0.0."""
    patterns = [p.strip() for p in file_patterns.split(',')]
    for f in file_list:
        for p in patterns:
            if fnmatch.fnmatch(f, p):
                return 1.0
    return 0.0

def _compute_semantic_similarity(db_path: str, decision_id: int, task_embedding: bytes) -> float:
    """Compute cosine similarity between task embedding and decision embedding."""
    decision_emb = get_embedding_for_decision(db_path, decision_id)
    if not decision_emb or not task_embedding:
        return 0.0
    return _cosine_similarity(task_embedding, decision_emb)

def _compute_recency_score(db_path: str, decision_id: int) -> float:
    """Return recency score between 0 and 1."""
    return get_reinforcement_recency_score(db_path, decision_id)

def format_injection(decisions, db_path=None, task_description=None, relevance_mode='balanced', top_k=10):
    """
    Format decisions for injection, optionally scoring by relevance.
    
    Parameters:
        decisions: list of decision dicts (as returned by get_decisions_for_files)
        db_path: path to the memory.db (needed for embedding retrieval)
        task_description: optional string describing the current task
        relevance_mode: 'balanced' (default) or 'strict'
        top_k: maximum number of decisions to include
    Returns:
        formatted string
    """
    if not decisions:
        return ""

    # If task_description is provided, compute its embedding once
    task_embedding = None
    if task_description and db_path:
        task_embedding = compute_embedding(task_description)

    # Compute scores for each decision
    scored = []
    for d in decisions:
        file_match = _file_pattern_match(d['file_patterns'], [])  # file_list not available here; will be passed separately
        # We'll get file_list from the caller later; for now we need to pass it.
        # Actually, the decisions already come from get_decisions_for_files which already filtered by file patterns.
        # So file_match is 1 for all decisions in this list.
        # But we still compute it for completeness.
        # We'll set file_match to 1 because the decisions are already filtered.
        file_match = 1.0  # because they were selected by file pattern matching

        semantic_sim = 0.0
        if task_embedding and db_path:
            semantic_sim = _compute_semantic_similarity(db_path, d['id'], task_embedding)

        recency = 0.0
        if db_path:
            recency = _compute_recency_score(db_path, d['id'])

        # Weighted score
        score = file_match * 0.4 + semantic_sim * 0.4 + recency * 0.2

        # In strict mode, only include decisions with file_match=1 (already true)
        if relevance_mode == 'strict' and file_match < 1.0:
            continue

        scored.append((score, d))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top_k
    top_decisions = [d for _, d in scored[:top_k]]

    if not top_decisions:
        return ""

    lines = [
        "[klyd] Architectural decisions governing files in this session:\n"
    ]
    
    for i, d in enumerate(top_decisions, 1):
        mod = f"[{d['module']}]"
        conf = f"{d['confidence']} confidence"
        count = f"confirmed {d['reinforcement_count']} times"
        dec = d['decision'].rstrip('.')  # remove trailing period if present
        lines.append(f"{i}. {mod} {dec}. ({conf}, {count})")
        
    lines.append("\nDo not contradict these decisions unless the user explicitly instructs you to change them.")
    
    return "\n".join(lines)
