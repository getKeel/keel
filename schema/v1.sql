CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision TEXT NOT NULL,
    module TEXT NOT NULL,
    file_patterns TEXT NOT NULL,      -- comma-separated glob patterns
    confidence TEXT NOT NULL,         -- LOW | MEDIUM | HIGH
    event_type TEXT NOT NULL,         -- NEW | REINFORCE | CONTRADICT
    reinforcement_count INTEGER DEFAULT 1,
    last_seen_commit TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    flagged INTEGER DEFAULT 0,        -- 1 = needs human review
    archived INTEGER DEFAULT 0,        -- 1 = soft-deleted
    UNIQUE(decision, module)
);
