import sqlite3
import fnmatch
from pathlib import Path

def get_schema_path():
    return Path(__file__).resolve().parent.parent / 'schema' / 'v1.sql'

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    schema = get_schema_path().read_text()
    conn.executescript(schema)
    conn.commit()
    conn.close()

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
                last_seen_commit = ?
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

def reinforce_decision(db_path, decision_id, commit_hash):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        UPDATE decisions 
        SET reinforcement_count = reinforcement_count + 1, last_seen_commit = ?
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
