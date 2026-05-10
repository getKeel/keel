import sqlite3
import fnmatch
import json
import time
import hashlib
from pathlib import Path

def get_schema_path():
    return Path(__file__).resolve().parent.parent / 'schema' / 'v1.sql'

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    schema = get_schema_path().read_text()
    conn.executescript(schema)
    conn.commit()
    conn.close()
    migrate_db(db_path)  # ensure new columns exist (safe for existing DBs)

def migrate_db(db_path):
    """Add new columns if they don't exist (safe migration)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Check existing columns
    cur.execute("PRAGMA table_info(decisions)")
    existing = {row[1] for row in cur.fetchall()}
    additions = []
    if 'embedding' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN embedding BLOB DEFAULT NULL")
    if 'last_reinforced_at' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN last_reinforced_at TEXT DEFAULT NULL")
    if 'relevance_score' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN relevance_score REAL DEFAULT 0.0")
    if 'version_id' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN version_id INTEGER DEFAULT 1")
    if 'parent_decision_id' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN parent_decision_id INTEGER DEFAULT NULL")
    if 'merged_into_id' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN merged_into_id INTEGER DEFAULT NULL")
    if 'auto_archive_after' not in existing:
        additions.append("ALTER TABLE decisions ADD COLUMN auto_archive_after INTEGER DEFAULT NULL")
    for stmt in additions:
        cur.execute(stmt)
    conn.commit()
    conn.close()

def _compute_dummy_embedding(text: str) -> bytes:
    """Placeholder: returns a fixed-size byte array (128 zeros) for now.
    Replace with real embedding computation later."""
    return b'\x00' * 128

def compute_embedding(text: str) -> bytes:
    """Compute embedding for a decision text.
    Currently returns a dummy embedding; override with real model."""
    return _compute_dummy_embedding(text)

def store_decision(db_path, decision_dict):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute('''
            INSERT INTO decisions (decision, module, file_patterns, confidence, event_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            decision_dict['decision'],
            decision_dict['module'],
            decision_dict['file_patterns'],
            decision_dict['confidence'],
            decision_dict.get('event_type', 'NEW')
        ))
        conn.commit()
        decision_id = cur.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate decision+module – update existing row
        cur.execute('''
            UPDATE decisions
            SET file_patterns = ?,
                confidence = ?,
                event_type = ?,
                reinforcement_count = reinforcement_count + 1,
                last_seen_commit = ?,
                last_reinforced_at = datetime('now')
            WHERE decision = ? AND module = ?
        ''', (
            decision_dict['file_patterns'],
            decision_dict['confidence'],
            decision_dict.get('event_type', 'NEW'),
            decision_dict.get('last_seen_commit'),
            decision_dict['decision'],
            decision_dict['module']
        ))
        conn.commit()
        cur.execute('SELECT id FROM decisions WHERE decision = ? AND module = ?',
                    (decision_dict['decision'], decision_dict['module']))
        decision_id = cur.fetchone()[0]
    conn.close()
    return decision_id

def store_decision_with_embedding(db_path, decision_dict, embedding_bytes=None):
    """Store a decision with an optional embedding vector.
    If embedding_bytes is None, compute it from decision text."""
    if embedding_bytes is None:
        embedding_bytes = compute_embedding(decision_dict['decision'])
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute('''
            INSERT INTO decisions (decision, module, file_patterns, confidence, event_type, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            decision_dict['decision'],
            decision_dict['module'],
            decision_dict['file_patterns'],
            decision_dict['confidence'],
            decision_dict.get('event_type', 'NEW'),
            embedding_bytes
        ))
        conn.commit()
        decision_id = cur.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate – update embedding and other fields
        cur.execute('''
            UPDATE decisions
            SET file_patterns = ?,
                confidence = ?,
                event_type = ?,
                reinforcement_count = reinforcement_count + 1,
                last_seen_commit = ?,
                last_reinforced_at = datetime('now'),
                embedding = ?
            WHERE decision = ? AND module = ?
        ''', (
            decision_dict['file_patterns'],
            decision_dict['confidence'],
            decision_dict.get('event_type', 'NEW'),
            decision_dict.get('last_seen_commit'),
            embedding_bytes,
            decision_dict['decision'],
            decision_dict['module']
        ))
        conn.commit()
        cur.execute('SELECT id FROM decisions WHERE decision = ? AND module = ?',
                    (decision_dict['decision'], decision_dict['module']))
        decision_id = cur.fetchone()[0]
    conn.close()
    return decision_id

def _match_any_file(file_patterns, file_list_str):
    patterns = [p.strip() for p in file_patterns.split(',')]
    files = file_list_str.split('|')
    for f in files:
        for p in patterns:
            if fnmatch.fnmatch(f, p):
                return 1
    return 0

def get_decisions_for_files(db_path, file_list, top_k=5):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.create_function("match_files", 2, _match_any_file)
    
    file_list_str = '|'.join(file_list)
    
    cur = conn.cursor()
    cur.execute('''
        SELECT * FROM decisions 
        WHERE flagged = 0 AND archived = 0 
        AND match_files(file_patterns, ?) = 1
        ORDER BY (reinforcement_count * CASE confidence WHEN 'HIGH' THEN 3 WHEN 'MEDIUM' THEN 2 ELSE 1 END) DESC, last_seen_commit DESC
        LIMIT ?
    ''', (file_list_str, top_k))
    
    results = [dict(row) for row in cur.fetchall()]
    conn.close()
    return results

def get_relevant_decisions(db_path, file_list, task_description=None, top_k=10):
    """Return decisions relevant to the given files and optional task description.
    Uses file pattern matching plus a placeholder for semantic similarity.
    Currently returns same as get_decisions_for_files but with top_k default 10."""
    # For now, use the existing file-pattern matching.
    # In future, incorporate embedding similarity with task_description.
    return get_decisions_for_files(db_path, file_list, top_k=top_k)

def reinforce_decision(db_path, decision_id, commit_hash):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        UPDATE decisions 
        SET reinforcement_count = reinforcement_count + 1, 
            last_seen_commit = ?,
            last_reinforced_at = datetime('now')
        WHERE id = ?
    ''', (commit_hash, decision_id))
    conn.commit()
    conn.close()

def flag_decision(db_path, decision_id):
    conn = sqlite3.connect(db_path)
    conn.execute('UPDATE decisions SET flagged = 1 WHERE id = ?', (decision_id,))
    conn.commit()
    conn.close()

def archive_decision(db_path, decision_id):
    conn = sqlite3.connect(db_path)
    conn.execute('UPDATE decisions SET archived = 1 WHERE id = ?', (decision_id,))
    conn.commit()
    conn.close()

def get_flagged_decisions(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM decisions WHERE flagged = 1 AND archived = 0")
    res = [dict(r) for r in cur.fetchall()]
    conn.close()
    return res

def get_active_decisions_by_module(db_path, module):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM decisions WHERE module = ? AND flagged = 0 AND archived = 0 ORDER BY reinforcement_count DESC", (module,))
    res = [dict(r) for r in cur.fetchall()]
    conn.close()
    return res

def resolve_decision(db_path, decision_id, action, old_id=None, new_text=None):
    conn = sqlite3.connect(db_path)
    if action == 'accept':
        if old_id:
            conn.execute('UPDATE decisions SET archived = 1 WHERE id = ?', (old_id,))
        conn.execute("UPDATE decisions SET flagged = 0, confidence = 'MEDIUM' WHERE id = ?", (decision_id,))
    elif action == 'reject':
        conn.execute('UPDATE decisions SET archived = 1 WHERE id = ?', (decision_id,))
    elif action == 'edit':
        if old_id:
            conn.execute('UPDATE decisions SET archived = 1 WHERE id = ?', (old_id,))
        conn.execute("UPDATE decisions SET flagged = 0, confidence = 'MEDIUM', decision = ? WHERE id = ?", (new_text, decision_id))
    conn.commit()
    conn.close()

def get_existing_decisions_for_files(file_paths: list[str]) -> list[dict]:
    db_path = str(Path('.klyd/memory.db').resolve())
    if not Path(db_path).exists():
        return []
    decisions = get_decisions_for_files(db_path, file_paths, top_k=50)
    high_conf = [d for d in decisions if d['confidence'] == 'HIGH']
    return [{'id': d['id'], 'decision': d['decision']} for d in high_conf[:5]]

# ----------------------------------------------------------------------
# Conflict resolution helpers
# ----------------------------------------------------------------------

def get_decision_by_id(db_path: str, decision_id: int) -> dict | None:
    """Retrieve a single decision by its id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def create_decision_version(db_path: str, original_id: int, new_decision_dict: dict) -> int:
    """
    Create a new version of a decision, linking it to the original via parent_decision_id.
    The new decision gets version_id = original.version_id + 1.
    Returns the id of the new decision.
    """
    original = get_decision_by_id(db_path, original_id)
    if not original:
        raise ValueError(f"Decision with id {original_id} not found")
    
    new_version = original['version_id'] + 1
    # Build the new dict, overriding fields from new_decision_dict
    new_dict = {
        'decision': new_decision_dict.get('decision', original['decision']),
        'module': new_decision_dict.get('module', original['module']),
        'file_patterns': new_decision_dict.get('file_patterns', original['file_patterns']),
        'confidence': new_decision_dict.get('confidence', original['confidence']),
        'event_type': new_decision_dict.get('event_type', 'NEW'),
        'last_seen_commit': new_decision_dict.get('last_seen_commit', original.get('last_seen_commit')),
        'embedding': new_decision_dict.get('embedding', original.get('embedding')),
    }
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO decisions
            (decision, module, file_patterns, confidence, event_type,
             last_seen_commit, embedding, version_id, parent_decision_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        new_dict['decision'],
        new_dict['module'],
        new_dict['file_patterns'],
        new_dict['confidence'],
        new_dict['event_type'],
        new_dict.get('last_seen_commit'),
        new_dict.get('embedding'),
        new_version,
        original_id
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id

def merge_decisions(db_path: str, keep_id: int, archive_id: int):
    """
    Merge two decisions: archive the one with archive_id and set its merged_into_id to keep_id.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        UPDATE decisions
        SET archived = 1, merged_into_id = ?
        WHERE id = ?
    ''', (keep_id, archive_id))
    conn.commit()
    conn.close()

def auto_archive_old_decisions(db_path: str):
    """
    Archive decisions that have auto_archive_after set and are older than that many days.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Archive decisions where auto_archive_after is not null and created_at is older than now - auto_archive_after days
    cur.execute('''
        UPDATE decisions
        SET archived = 1
        WHERE archived = 0
          AND auto_archive_after IS NOT NULL
          AND datetime(created_at, '+' || auto_archive_after || ' days') < datetime('now')
    ''')
    conn.commit()
    conn.close()

def get_decision_versions(db_path: str, decision_id: int) -> list[dict]:
    """
    Return all versions of a decision, following the parent chain.
    Returns a list ordered by version_id ascending.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    # First get the root (the one with no parent)
    cur.execute('''
        WITH RECURSIVE ancestors(id, parent_decision_id, version_id, decision, module, file_patterns, confidence, event_type, reinforcement_count, last_seen_commit, created_at, flagged, archived, embedding, last_reinforced_at, relevance_score, merged_into_id, auto_archive_after) AS (
            SELECT id, parent_decision_id, version_id, decision, module, file_patterns, confidence, event_type, reinforcement_count, last_seen_commit, created_at, flagged, archived, embedding, last_reinforced_at, relevance_score, merged_into_id, auto_archive_after
            FROM decisions
            WHERE id = ?
            UNION ALL
            SELECT d.id, d.parent_decision_id, d.version_id, d.decision, d.module, d.file_patterns, d.confidence, d.event_type, d.reinforcement_count, d.last_seen_commit, d.created_at, d.flagged, d.archived, d.embedding, d.last_reinforced_at, d.relevance_score, d.merged_into_id, d.auto_archive_after
            FROM decisions d
            INNER JOIN ancestors a ON d.id = a.parent_decision_id
        )
        SELECT * FROM ancestors ORDER BY version_id ASC
    ''', (decision_id,))
    results = [dict(row) for row in cur.fetchall()]
    conn.close()
    return results

def get_decision_ancestry(db_path: str, decision_id: int) -> list[dict]:
    """
    Return the chain of ancestors (parents) for a given decision, starting from the root.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('''
        WITH RECURSIVE ancestors(id, parent_decision_id, version_id, decision, module, file_patterns, confidence, event_type, reinforcement_count, last_seen_commit, created_at, flagged, archived, embedding, last_reinforced_at, relevance_score, merged_into_id, auto_archive_after) AS (
            SELECT id, parent_decision_id, version_id, decision, module, file_patterns, confidence, event_type, reinforcement_count, last_seen_commit, created_at, flagged, archived, embedding, last_reinforced_at, relevance_score, merged_into_id, auto_archive_after
            FROM decisions
            WHERE id = ?
            UNION ALL
            SELECT d.id, d.parent_decision_id, d.version_id, d.decision, d.module, d.file_patterns, d.confidence, d.event_type, d.reinforcement_count, d.last_seen_commit, d.created_at, d.flagged, d.archived, d.embedding, d.last_reinforced_at, d.relevance_score, d.merged_into_id, d.auto_archive_after
            FROM decisions d
            INNER JOIN ancestors a ON d.id = a.parent_decision_id
        )
        SELECT * FROM ancestors ORDER BY version_id ASC
    ''', (decision_id,))
    results = [dict(row) for row in cur.fetchall()]
    conn.close()
    return results
