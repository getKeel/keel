import click
import subprocess
import json
import traceback
from .hooks import install_hooks
from .config import init_config, set_config, get_config, get_all_config
from .db import get_schema_path
import sqlite3
from pathlib import Path

@click.group()
def cli():
    """keel: a CLI tool that wraps coding agents via git hooks to inject architectural memory."""
    pass

def echo_brand(msg, bold=False):
    click.secho("keel", fg="cyan", bold=True, nl=False)
    click.echo(f" | {msg}")

@cli.command()
def init():
    """Initialize keel in the current git repository."""
    try:
        install_hooks()
        init_config()
        # Init DB as well
        keel_dir = Path('.keel')
        db_path = keel_dir / 'memory.db'
        from .db import init_db
        init_db(str(db_path))
        echo_brand("Initialized successfully. Hooks installed.")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        raise click.Abort()

@cli.command()
@click.option('--api-key', help='Anthropic API key')
@click.option('--openai-key', help='OpenAI API key')
@click.option('--openrouter-key', help='OpenRouter API key')
@click.option('--gemini-key', help='Gemini API key')
@click.option('--groq-key', help='Groq API key')
@click.option('--model', help='Model to use (default: claude-sonnet-4-6)')
def config(api_key, openai_key, openrouter_key, gemini_key, groq_key, model):
    """Set keel configuration."""
    if api_key:
        set_config('api_key', api_key)
        echo_brand("Anthropic key saved.")
    if openai_key:
        set_config('openai_key', openai_key)
        echo_brand("OpenAI key saved.")
    if openrouter_key:
        set_config('openrouter_key', openrouter_key)
        echo_brand("OpenRouter key saved.")
    if gemini_key:
        set_config('gemini_key', gemini_key)
        echo_brand("Gemini key saved.")
    if groq_key:
        set_config('groq_key', groq_key)
        echo_brand("Groq key saved.")
    if model:
        set_config('model', model)
        echo_brand(f"Model set to {click.style(model, fg='blue', bold=True)}.")
    if not any([api_key, openai_key, openrouter_key, gemini_key, groq_key, model]):
        click.secho("Usage: ", nl=False, fg="yellow")
        click.echo("keel config --api-key ... --openai-key ... --model ...")

@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument('cmd', nargs=-1, type=click.UNPROCESSED)
def run(cmd):
    """Run an agent with injected architectural memory."""
    if not cmd:
        click.echo("Usage: keel run <agent> [args...]")
        return
        
    keel_dir = Path('.keel')
    inj_path = keel_dir / 'injection.txt'
    
    run_cmd = list(cmd)
    
    if inj_path.exists() and inj_path.stat().st_size > 0:
        agent_name = run_cmd[0].lower()
        if agent_name == 'aider':
            run_cmd.extend(['--message-file', str(inj_path)])
        elif agent_name == 'opencode':
            run_cmd.extend(['-m', inj_path.read_text()])
        # For others, we might need specific flags, but we default to just passing it or letting the user know
    
    try:
        subprocess.run(run_cmd)
    except FileNotFoundError:
        click.echo(f"Command not found: {cmd[0]}", err=True)

@cli.command()
def extract_commit():
    """Extract decisions from the last commit."""
    from .extractor import extract_decisions
    from .db import get_decisions_for_files, store_decision, reinforce_decision, flag_decision
    
    keel_dir = Path('.keel')
    if not (keel_dir / 'memory.db').exists():
        return
        
    try:
        # Get git info
        try:
            diff = subprocess.check_output(['git', 'diff', 'HEAD~1', 'HEAD'], text=True)
            msg = subprocess.check_output(['git', 'log', '-1', '--format=%B'], text=True)
            files_out = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'], text=True)
            files = [f for f in files_out.strip().split('\n') if f]
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        except subprocess.CalledProcessError:
            # Initial commit fallback
            diff = subprocess.check_output(['git', 'show', 'HEAD'], text=True)
            msg = subprocess.check_output(['git', 'log', '-1', '--format=%B'], text=True)
            files_out = subprocess.check_output(['git', 'show', '--name-only', '--format=', 'HEAD'], text=True)
            files = [f for f in files_out.strip().split('\n') if f]
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()

        if not files:
            return

        db_path = str(keel_dir / 'memory.db')
        existing = get_decisions_for_files(db_path, files, top_k=20)
        existing_json = json.dumps(existing, indent=2)

        config_data = get_all_config()
        model = config_data.get('model', 'claude-sonnet-4-6')
        
        decisions = extract_decisions(diff, msg, existing_json, config_data, model)

        for d in decisions:
            event = d.get('event_type')
            
            if event == 'REINFORCE':
                # Find matching decision to reinforce
                match = next((e for e in existing if e['module'] == d['module'] and e['decision'] == d['decision']), None)
                if match:
                    reinforce_decision(db_path, match['id'], commit_hash)
                else:
                    d['event_type'] = 'NEW'
                    store_decision(db_path, d)
            elif event == 'CONTRADICT':
                did = store_decision(db_path, d)
                flag_decision(db_path, did)
            else:
                store_decision(db_path, d)

    except Exception as e:
        err_log = keel_dir / 'errors.log'
        with open(err_log, 'a') as f:
            f.write(f"Error extracting commit:\n{traceback.format_exc()}\n")
        return

@cli.command()
def prepare_injection():
    """Prepare injection file for agent sessions."""
    from .db import get_decisions_for_files
    from .injector import format_injection
    
    keel_dir = Path('.keel')
    if not (keel_dir / 'memory.db').exists():
        return
        
    try:
        files_out = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True)
        files = [f for f in files_out.strip().split('\n') if f]
        
        if not files:
            with open(keel_dir / 'injection.txt', 'w') as f:
                f.write('')
            return
            
        db_path = str(keel_dir / 'memory.db')
        decisions = get_decisions_for_files(db_path, files, top_k=20)
        
        injection = format_injection(decisions)
        with open(keel_dir / 'injection.txt', 'w') as f:
            f.write(injection)
            
    except Exception as e:
        with open(keel_dir / 'errors.log', 'a') as f:
            f.write(f"Error preparing injection:\n{traceback.format_exc()}\n")
        return

@cli.command()
def review():
    """Review flagged conflicting decisions."""
    from .db import get_flagged_decisions, get_active_decisions_by_module, resolve_decision
    
    keel_dir = Path('.keel')
    db_path = keel_dir / 'memory.db'
    if not db_path.exists():
        click.secho("keel is not initialized. Run `keel init`.", fg="red")
        return

    db_str = str(db_path)
    flagged = get_flagged_decisions(db_str)
    
    if not flagged:
        echo_brand("No conflicts to review.", bold=True)
        return

    for d in flagged:
        click.echo("")
        click.secho("! CONFLICT DETECTED", fg="red", bold=True, bg="black")
        click.secho(f"Module: ", fg="cyan", nl=False)
        click.secho(d['module'], fg="blue", bold=True)
        
        commit_ref = d.get('last_seen_commit') or "unknown commit"
        click.secho(f"New:      ", fg="cyan", nl=False)
        click.secho(f"\"{d['decision']}\"", fg="white", bold=True, nl=False)
        click.secho(f" (from commit {commit_ref[:7]})", fg="cyan", dim=True)
        
        active = get_active_decisions_by_module(db_str, d['module'])
        old_id = None
        click.secho(f"Existing: ", fg="cyan", nl=False)
        if active:
            old = active[0]
            old_id = old['id']
            click.secho(f"\"{old['decision']}\"", fg="white", nl=False)
            click.secho(f" ({old['confidence']} confidence, x{old['reinforcement_count']})", fg="cyan", dim=True)
        else:
            click.secho("(none)", fg="white")

        click.echo("")
        click.secho("[a] ", fg="green", bold=True, nl=False); click.echo("Accept new decision (archive old)")
        click.secho("[r] ", fg="red", bold=True, nl=False); click.echo("Reject new decision (keep old, discard this finding)")
        click.secho("[e] ", fg="yellow", bold=True, nl=False); click.echo("Edit decision manually")
        click.secho("[s] ", fg="blue", bold=True, nl=False); click.echo("Skip for now")
        
        while True:
            choice = click.prompt("Choice", type=click.Choice(['a', 'r', 'e', 's']), show_choices=False).lower()
            if choice == 's':
                click.secho("Skipped.", fg="cyan", dim=True)
                break
            elif choice == 'a':
                resolve_decision(db_str, d['id'], 'accept', old_id=old_id)
                click.secho("Accepted new decision.", fg="green", bold=True)
                break
            elif choice == 'r':
                resolve_decision(db_str, d['id'], 'reject')
                click.secho("Rejected new decision.", fg="red", bold=True)
                break
            elif choice == 'e':
                new_text = click.edit(d['decision'])
                if new_text is not None:
                    new_text = new_text.strip()
                    resolve_decision(db_str, d['id'], 'edit', old_id=old_id, new_text=new_text)
                    click.secho("Saved edited decision.", fg="yellow", bold=True)
                    break
                else:
                    click.secho("Edit cancelled. Please choose an option.", fg="red")

@cli.command()
def status():
    """Show the current memory store status."""
    keel_dir = Path('.keel')
    db_path = keel_dir / 'memory.db'
    if not db_path.exists():
        click.secho("keel is not initialized. Run `keel init`.", fg="red")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM decisions WHERE archived = 0")
    all_decisions = [dict(r) for r in cur.fetchall()]
    conn.close()

    active = [d for d in all_decisions if d['flagged'] == 0]
    flagged = [d for d in all_decisions if d['flagged'] == 1]

    click.secho("-" * 45, fg="cyan", dim=True)
    click.secho(f"DECISIONS ", fg="cyan", bold=True, nl=False)
    click.echo(f"({len(active)} total, {len(flagged)} flagged)\n")

    # Sort active by count desc
    active.sort(key=lambda x: x['reinforcement_count'], reverse=True)
    
    for d in active:
        mod = click.style(d['module'].ljust(14), fg="blue")
        dec = click.style(d['decision'].ljust(30), fg="white")
        
        conf_color = "green" if d['confidence'] == 'HIGH' else ("yellow" if d['confidence'] == 'MEDIUM' else "red")
        conf = click.style(d['confidence'].ljust(7), fg=conf_color)
        
        count = click.style(f"x{d['reinforcement_count']}", fg="cyan", dim=True)
        click.echo(f"  {mod} {dec} {conf} {count}")

    if flagged:
        click.echo("")
        click.secho(f"! NEEDS REVIEW ({len(flagged)})", fg="yellow", bold=True)
        for d in flagged:
            mod = click.style(d['module'].ljust(14), fg="blue")
            conflicts = [a for a in active if a['module'] == d['module']]
            
            conflict_text = ""
            if conflicts:
                conflict_text = click.style(f" [CONTRADICTS: {conflicts[0]['decision']}]", fg="red", dim=True)
            
            dec = f"{d['decision']}{conflict_text}"
            
            # Formatting padding accounting for color codes is annoying, so we pad manually
            # or just let it flow since it's an error state
            conf = click.style(d['confidence'], fg="red")
            click.echo(f"  {mod} {dec}  {conf}")
        
        click.secho("  -> run `keel review` to resolve", fg="yellow", dim=True)
    
    click.secho("-" * 45, fg="cyan", dim=True)

