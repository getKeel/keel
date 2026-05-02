import os
from klyd.db import init_db, store_decision, get_decisions_for_files, flag_decision

def test_db_flow():
    db_path = 'test.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    init_db(db_path)

    # 1. Insert 5 decisions
    d1 = store_decision(db_path, {
        'decision': 'JWT-only, no sessions', 'module': 'auth', 
        'file_patterns': 'auth/*', 'confidence': 'HIGH', 'event_type': 'NEW'
    })
    d2 = store_decision(db_path, {
        'decision': 'No GraphQL', 'module': 'auth', 
        'file_patterns': 'auth/*', 'confidence': 'MEDIUM', 'event_type': 'NEW'
    })
    d3 = store_decision(db_path, {
        'decision': 'Prisma ORM', 'module': 'db', 
        'file_patterns': 'db/*', 'confidence': 'HIGH', 'event_type': 'NEW'
    })
    d4 = store_decision(db_path, {
        'decision': 'REST only', 'module': 'api', 
        'file_patterns': 'api/*', 'confidence': 'MEDIUM', 'event_type': 'NEW'
    })
    d5 = store_decision(db_path, {
        'decision': 'Drizzle ORM', 'module': 'db', 
        'file_patterns': 'db/*', 'confidence': 'LOW', 'event_type': 'CONTRADICT'
    })

    # Flag the CONTRADICT one
    flag_decision(db_path, d5)

    # 2. Query for module A
    res_a = get_decisions_for_files(db_path, ['auth/login.py'])
    assert 'JWT-only, no sessions' in [r['decision'] for r in res_a]
    assert 'No GraphQL' in [r['decision'] for r in res_a]

    # 3. Query for module B
    res_b = get_decisions_for_files(db_path, ['db/schema.py'])
    assert 'Prisma ORM' in [r['decision'] for r in res_b]

    # 4. CONTRADICT does not appear
    assert not any(r['decision'] == 'Drizzle ORM' for r in res_b)

    if os.path.exists(db_path):
        os.remove(db_path)
