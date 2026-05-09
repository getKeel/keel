# klyd

<p align="center">

```
888  /         888           Y88b    /       888~-_   
888 /          888            Y88b  /        888   \  
888/\          888             Y88b/         888    | 
888  \         888              Y8Y          888    | 
888   \        888               Y           888   /  
888    \       888____          /            888_-~   
```

#### *An open-source project, not affiliated with the Klyd SaaS*

</p>

---

## What is klyd?

klyd is a **decision memory harness** for terminal coding agents (aider, opencode, claude code). It wraps your agent workflow to inject persistent architectural memory into every session.

### The Problem

Agentic coding tools fail in a specific, reproducible way:

1. Agent writes a file
2. Something breaks (test, CI)
3. Agent patches the nearest thing to make the signal go green
4. It doesn't fix the root — it satisfies the metric
5. Next session: agent reads the patched code as ground truth, builds on top of it
6. After ~20 turns: **slop fortress** — a codebase that compiles but nobody understands

This is not a model quality problem. It's a **memory and context architecture problem**.

### The Solution

klyd extracts architectural decisions from every commit and injects them into your agent's context at the exact moment of file write — when architectural damage happens.

- **Two LLM calls per commit cycle maximum**
- **All state lives in `.klyd/`** (local to your repo)
- **BYOK** — bring your own API key
- **Zero infrastructure** — no cloud, no servers

---

## Installation

```bash
# From PyPI (recommended)
pip install klyd

# Or from source
pip install .
```

After installation, verify:
```bash
kl --help
```

---

## Quickstart

```bash
# 1. Initialize in your project repository
cd your-project
kl init

# 2. Configure your API key
kl config --api-key sk-ant-...

# 3. Make commits with architectural decisions
# (e.g., adding auth, changing DB, adopting a new library)

# 4. Check extracted decisions
kl status

# 5. Run your agent with injected memory
kl run aider
```

---

## Configuration

klyd supports multiple LLM providers via BYOK (Bring Your Own Key).

### Anthropic (Default)

```bash
kl config --api-key sk-ant-... --model claude-sonnet-4-6
```

### OpenAI

```bash
kl config --openai-key sk-proj-... --model gpt-4o
```

### OpenRouter

```bash
kl config --openrouter-key sk-or-... --model openrouter/free
```

### Google Gemini

```bash
kl config --gemini-key AIza... --model gemini-1.5-pro
```

### Groq

```bash
kl config --groq-key gsk_... --model llama3-8b-8192
```

### View Current Configuration

```bash
kl config --show
```

---

## Commands

### `kl init`

Initialize klyd in the current git repository.

```bash
kl init
```

This:
- Creates `.klyd/` directory
- Initializes SQLite database at `.klyd/memory.db`
- Installs git hooks (post-commit, pre-commit)
- Creates default config at `.klyd/config.json`

### `kl config`

Configure klyd settings.

```bash
# Set API key
kl config --api-key sk-ant-...

# Set model
kl config --model claude-sonnet-4-6

# Show current config
kl config --show
```

### `kl status`

View the current decision memory store.

```bash
kl status
```

Shows:
- Active decisions (sorted by reinforcement count)
- Confidence levels (HIGH/MEDIUM/LOW)
- Event types (NEW/REINFORCE/CONTRADICT)
- Flagged conflicts requiring review

### `kl run <agent>`

Run a coding agent with injected architectural memory.

```bash
kl run aider --message "Add user authentication"
kl run opencode
```

Options:
- `--no-inject` - Skip generating injection file

The agent receives your stored architectural decisions as context, preventing contradiction of established patterns.

### `kl review`

Review flagged conflicting decisions.

```bash
kl review
```

When klyd detects a CONTRADICT (a new decision that conflicts with existing memory), it flags it for human review. This interactive command lets you:

- **Accept** new decision (archive old)
- **Reject** new decision (keep old)
- **Edit** decision manually
- **Skip** for now

### `kl extract-commit`

Manually trigger decision extraction for the last commit.

```bash
kl extract-commit
```

Usually called automatically by the post-commit hook.

### `kl prepare-injection`

Generate the injection file for agent sessions.

```bash
kl prepare-injection
```

Usually called automatically by the pre-commit hook. Outputs to `.klyd/injection.txt`.

---

## How It Works

### Git Hooks

klyd installs two git hooks:

**post-commit** (`hooks/post-commit.sh`):
```bash
#!/bin/sh
kl extract-commit >> .klyd/errors.log 2>&1 || true
```

After every commit, extracts architectural decisions via LLM and stores them in SQLite.

**pre-commit** (`hooks/pre-commit.sh`):
```bash
#!/bin/sh
kl prepare-injection >> .klyd/errors.log 2>&1 || true
```

Before files are committed, queries relevant decisions and writes injection file.

### Extraction Flow

1. Get diff: `git diff HEAD~1 HEAD`
2. Get commit message: `git log -1 --format=%B`
3. Get touched files
4. Query existing decisions for those files
5. Call LLM with prompt
6. Store results:
   - **NEW** → `store_decision()`
   - **REINFORCE** → `reinforce_decision()` on matching existing
   - **CONTRADICT** → `store_decision()` with `flagged=True`

### Injection Flow

1. Get staged files: `git diff --cached --name-only`
2. Query top-k decisions for those files (excludes flagged, archived)
3. Format as injection message
4. Write to `.klyd/injection.txt`

### Decision Schema

```sql
CREATE TABLE decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision TEXT NOT NULL,
    module TEXT NOT NULL,
    file_patterns TEXT NOT NULL,
    confidence TEXT NOT NULL,       -- LOW | MEDIUM | HIGH
    event_type TEXT NOT NULL,        -- NEW | REINFORCE | CONTRADICT
    reinforcement_count INTEGER DEFAULT 1,
    last_seen_commit TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    flagged INTEGER DEFAULT 0,      -- 1 = needs human review
    archived INTEGER DEFAULT 0      -- 1 = soft-deleted
);
```

---

## Examples

### Example 1: First-Time Setup

```bash
$ cd my-project
$ kl init

# Configure with your API key
$ kl config --api-key sk-ant-...

# Make your first architectural commit
$ echo "import sqlite3" > db.py
$ git add db.py && git commit -m "feat: add sqlite for data storage"

# Check extracted decisions
$ kl status
```

### Example 2: Resolving Conflicts

```bash
# Make a commit that contradicts existing decision
$ echo "import mongoengine" > nosql.py
$ git add nosql.py && git commit -m "feat: switch to mongodb"

# LLM detects contradiction, flags for review
$ kl status
# Shows flagged decision in NEEDS REVIEW section

$ kl review
# Interactive resolution
```

### Example 3: Running Agent with Memory

```bash
# Inject memory and run aider
$ kl run aider --message "Add user login to the app"

# Agent receives your architectural decisions as context
# Won't contradict established patterns (e.g., "use SQLite, not sessions")
```

---

## File Structure

```
your-project/
├── .klyd/                    # klyd's working directory
│   ├── memory.db            # SQLite decision store
│   ├── config.json          # Your API keys & settings
│   ├── injection.txt        # Generated for agent sessions
│   └── errors.log           # Error logs
├── .git/hooks/
│   ├── post-commit          # Installed by kl init
│   └── pre-commit           # Installed by kl init
└── [your project files]
```

---

## Troubleshooting

### "klyd is not initialized. Run `kl init`."

```bash
kl init
```

### API Key Errors

Check your config:
```bash
kl config --show
```

Ensure the key is correct. Errors are logged to `.klyd/errors.log`.

### No Decisions Extracted

- Make commits with clear architectural changes (not just style fixes)
- Ensure API key has credits/quota
- Check `.klyd/errors.log` for details

### Hook Not Firing

Verify hooks are installed:
```bash
ls -la .git/hooks/post-commit
ls -la .git/hooks/pre-commit
```

If missing, re-run:
```bash
kl init
```

---

## Requirements

- Python 3.11+
- click
- anthropic
- Git repository

## License

MIT License - see LICENSE file.

---

## Links

- PyPI: https://pypi.org/project/klyd/
- GitHub: https://github.com/klyd-studio/klyd-harness
