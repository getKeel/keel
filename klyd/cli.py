import click
import subprocess
import json
import traceback
import urllib.request
from .hooks import install_hooks
from .config import init_config, set_config, get_config, get_all_config, get_injection_template, set_injection_template, get_strict_mode, set_strict_mode, get_pinned_decision_ids, add_pinned_decision_id, remove_pinned_decision_id, clear_pinned_decision_ids, get_max_decisions_inject, set_max_decisions_inject
from .db import get_schema_path, migrate_db, get_decision_by_id, create_decision_version, merge_decisions, auto_archive_old_decisions
import sqlite3
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.columns import Columns
from rich.prompt import Prompt
from rich.text import Text
import rich.box

console = Console()

class KlydGroup(click.Group):
    def format_help(self, ctx, formatter):
        console.print()
        super().format_help(ctx, formatter)

@click.group(cls=KlydGroup)
def cli():
    """klyd: a CLI tool that wraps coding agents via git hooks to inject architectural memory."""
    pass

def echo_brand(msg, bold=False):
    console.print(f"[cyan bold]klyd[/cyan bold] | {msg}")

# ----------------------------------------------------------------------
# LLM helper for merge suggestions
# ----------------------------------------------------------------------

def _call_llm_for_merge(old_decision: str, new_decision: str, config_data: dict) -> str:
    """Call the configured LLM to propose a unified decision."""
    prompt = f"""You are an architectural decision merger. You have two conflicting decisions about the same module.

Old decision:
{old_decision}

New decision:
{new_decision}

Propose a single unified decision that combines the best of both, resolving any contradictions.
Return ONLY the unified decision text, no explanation, no markdown, no JSON.
"""
    model = config_data.get('model', 'claude-sonnet-4-6')
    is_anthropic_model = model.startswith('claude-') or model.startswith('anthropic/')
    
    try:
        if is_anthropic_model and 'api_key' in config_data and not any(k in config_data for k in ['openai_key', 'openrouter_key', 'gemini_key', 'groq_key']):
            from anthropic import Anthropic
            client = Anthropic(api_key=config_data['api_key'])
            actual_model = model.replace('anthropic/', '')
            response = client.messages.create(
                model=actual_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        elif is_anthropic_model and 'openrouter_key' in config_data:
            url = "https://openrouter.ai/api/v1/chat/completions"
            key = config_data['openrouter_key']
        else:
            if model.startswith('gpt-') or model.startswith('o1') or model.startswith('o3'):
                if 'openai_key' not in config_data:
                    raise ValueError("OpenAI API key missing.")
                url = "https://api.openai.com/v1/chat/completions"
                key = config_data['openai_key']
            elif model.startswith('gemini-') or model.startswith('gemma-'):
                if 'gemini_key' not in config_data:
                    raise ValueError("Gemini API key missing.")
                url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
                key = config_data['gemini_key']
            elif '/' in model:
                if 'openrouter_key' not in config_data:
                    raise ValueError("OpenRouter API key missing.")
                url = "https://openrouter.ai/api/v1/chat/completions"
                key = config_data['openrouter_key']
            elif config_data.get('groq_key'):
                url = "https://api.groq.com/openai/v1/chat/completions"
                key = config_data['groq_key']
            elif config_data.get('openai_key'):
                url = "https://api.openai.com/v1/chat/completions"
                key = config_data['openai_key']
            elif config_data.get('openrouter_key'):
                url = "https://openrouter.ai/api/v1/chat/completions"
                key = config_data['openrouter_key']
            else:
                raise ValueError(f"No valid API key configured for model: {model}")
            
            data = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }).encode('utf-8')
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            }
            if "openrouter.ai" in url:
                headers["HTTP-Referer"] = "https://github.com/getKlyd/klyd"
                headers["X-Title"] = "klyd"
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req) as response:
                res = json.loads(response.read().decode('utf-8'))
                return res['choices'][0]['message']['content'].strip()
    except Exception as e:
        console.print(f"[red]LLM merge suggestion failed: {e}[/red]")
        return None

# ----------------------------------------------------------------------
# Diff display helper
# ----------------------------------------------------------------------

def _format_diff(old_text: str, new_text: str) -> str:
    """Create a simple side-by-side diff representation."""
    old_lines = old_text.split('\n')
    new_lines = new_text.split('\n')
    max_len = max(len(old_lines), len(new_lines))
    result_lines = []
    for i in range(max_len):
        old_line = old_lines[i] if i < len(old_lines) else ''
        new_line = new_lines[i] if i < len(new_lines) else ''
        # Truncate to 40 chars for side-by-side
        old_trunc = old_line[:40] if len(old_line) > 40 else old_line.ljust(40)
        new_trunc = new_line[:40] if len(new_line) > 40 else new_line.ljust(40)
        if old_line != new_line:
            result_lines.append(f"[red]{old_trunc}[/red] | [green]{new_trunc}[/green]")
        else:
            result_lines.append(f"[dim]{old_trunc}[/dim] | [dim]{new_trunc}[/dim]")
    return '\n'.join(result_lines)

# ----------------------------------------------------------------------
# CLI commands
# ----------------------------------------------------------------------

@cli.command()
def init():
    """Initialize klyd in the current git repository."""
    console.print()
    console.print(r"[green bold]888  /         888           Y88b    /       888~-_   [/green bold]")
    console.print(r"[green bold]888 /          888            Y88b  /        888   \  [/green bold]")
    console.print(r"[green bold]888/\          888             Y88b/         888    | [/green bold]")
    console.print(r"[green bold]888  \         888              Y8Y          888    | [/green bold]")
    console.print(r"[green bold]888   \        888               Y           888   /  [/green bold]")
    console.print(r"[green bold]888    \       888____          /            888_-~   [/green bold]")
    console.print()
    console.print(r"                   [dim](An open-source project, not affiliated with the Klyd SaaS)[/dim]")
    console.print()
    console.print()
    try:
        with console.status("[bold cyan]Creating .klyd directory, database, and hooks...[/bold cyan]", spinner="dots12"):
            install_hooks()
            init_config()
            klyd_dir = Path('.klyd')
            db_path = klyd_dir / 'memory.db'
            from .db import init_db
            init_db(str(db_path))
            # Migration is called inside init_db, but call again for safety
            migrate_db(str(db_path))
            
        console.print(Panel(
            "Klyd harness initialised in [cyan].klyd[/cyan]\n\n[dim]Installed git hooks for automatic extraction.[/dim]\n[dim]Errors are logged to .klyd/errors.log[/dim]",
            title="[bold green]SUCCESS[/bold green]", border_style="green", padding=(1, 2)
        ))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()

@cli.command()
@click.option('--api-key', help='Anthropic API key')
@click.option('--openai-key', help='OpenAI API key')
@click.option('--openrouter-key', help='OpenRouter API key')
@click.option('--gemini-key', help='Gemini API key')
@click.option('--groq-key', help='Groq API key')
@click.option('--model', help='Model to use (default: claude-sonnet-4-6)')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--injection-template', help='Custom injection template with {decisions} placeholder')
@click.option('--strict-mode', type=bool, help='Enable strict mode (only file-matched decisions)')
@click.option('--max-decisions', type=int, help='Maximum number of decisions to inject')
def config(api_key, openai_key, openrouter_key, gemini_key, groq_key, model, show, injection_template, strict_mode, max_decisions):
    """Set klyd configuration."""
    if show:
        cfg = get_all_config()
        table = Table(title="[bold cyan]klyd Configuration[/bold cyan]", box=rich.box.SIMPLE, header_style="bold cyan")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        if not cfg:
            table.add_row("[dim]No configuration set[/dim]", "[dim]Run: kl config --api-key ...[/dim]")
        else:
            for k, v in cfg.items():
                if 'key' in k and v:
                    v = v[:4] + '*' * (len(v) - 8) + v[-4:] if len(v) > 8 else '*' * len(v)
                table.add_row(k, v)
        console.print(table)
        return

    changes = False
    with console.status("[bold cyan]Saving configuration...[/bold cyan]", spinner="dots12"):
        if api_key:
            set_config('api_key', api_key)
            changes = True
        if openai_key:
            set_config('openai_key', openai_key)
            changes = True
        if openrouter_key:
            set_config('openrouter_key', openrouter_key)
            changes = True
        if gemini_key:
            set_config('gemini_key', gemini_key)
            changes = True
        if groq_key:
            set_config('groq_key', groq_key)
            changes = True
        if model:
            set_config('model', model)
            changes = True
        if injection_template is not None:
            set_injection_template(injection_template)
            changes = True
        if strict_mode is not None:
            set_strict_mode(strict_mode)
            changes = True
        if max_decisions is not None:
            set_max_decisions_inject(max_decisions)
            changes = True
            
    if changes:
        console.print(Panel("[bold green]Configuration saved successfully[/bold green]", title="[bold green]DONE[/bold green]", border_style="green", expand=False))
    else:
        console.print("[yellow]Usage:[/yellow] kl config --api-key ... --openai-key ... --model ... --injection-template ... --strict-mode ... --max-decisions ...\nOr use --show to display current configuration.")

@cli.command(context_settings={"ignore_unknown_options": True})
@click.option('--no-inject', is_flag=True, help='Skip generating injection file')
@click.option('--relevance-mode', type=click.Choice(['balanced', 'strict']), default='balanced', help='Relevance scoring mode')
@click.argument('cmd', nargs=-1, type=click.UNPROCESSED)
def run(no_inject, relevance_mode, cmd):
    """Run an agent with injected architectural memory."""
    if not cmd:
        console.print("Usage: kl run <agent> [args...]")
        return
        
    klyd_dir = Path('.klyd')
    inj_path = klyd_dir / 'injection.txt'
    
    if not no_inject:
        with console.status("[bold cyan]Preparing injection context...[/bold cyan]", spinner="dots12"):
            try:
                ctx = click.get_current_context()
                # Pass relevance_mode to prepare_injection
                ctx.invoke(prepare_injection, relevance_mode=relevance_mode)
            except Exception as e:
                pass
            
    run_cmd = list(cmd)
    agent_name = run_cmd[0].lower()
    
    if inj_path.exists() and inj_path.stat().st_size > 0:
        if agent_name == 'aider':
            run_cmd.extend(['--message-file', str(inj_path)])
        elif agent_name == 'opencode':
            run_cmd.extend(['-m', inj_path.read_text()])
            
    console.print(Panel(f"[bold cyan]Launching[/bold cyan] {agent_name}...", title="[bold blue]AGENT START[/bold blue]", border_style="blue", expand=False))
    
    try:
        subprocess.run(run_cmd)
        console.print(Panel("[bold green]Agent session ended[/bold green]", title="[bold green]DONE[/bold green]", border_style="green", expand=False))
    except FileNotFoundError:
        console.print(f"[red]Command not found: {cmd[0]}[/red]")

@cli.command()
def extract_commit():
    """Extract decisions from the last commit."""
    from .extractor import extract_decisions
    from .db import get_decisions_for_files, store_decision_with_embedding, reinforce_decision, flag_decision
    
    klyd_dir = Path('.klyd')
    if not (klyd_dir / 'memory.db').exists():
        return
    
    try:
        diff = subprocess.check_output(['git', 'show', 'HEAD'], text=True)
        files_out = subprocess.check_output(['git', 'show', '--name-only', '--format=', 'HEAD'], text=True)
        msg = subprocess.check_output(['git', 'log', '-1', '--format=%B'], text=True)
        files = [f for f in files_out.strip().split('\n') if f]
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except subprocess.CalledProcessError as e:
        err_log = klyd_dir / 'errors.log'
        with open(err_log, 'a') as f:
            f.write(f"Error running git command: {e}\n")
        return

    if not files:
        return

    try:
        with console.status("[bold cyan]Extracting architectural decisions via LLM...[/bold cyan]", spinner="dots12"):
            db_path = str(klyd_dir / 'memory.db')
            existing = get_decisions_for_files(db_path, files, top_k=20)
            existing_json = json.dumps(existing, indent=2)

            config_data = get_all_config()
            model = config_data.get('model', 'claude-sonnet-4-6')
            
            decisions = extract_decisions(diff, msg, existing_json, config_data, model)

            new_count = 0
            reinforced_count = 0

            for d in decisions:
                event = d.get('event_type')
                d['last_seen_commit'] = commit_hash
                emb_bytes = d.pop('embedding_bytes', None)
                
                if event == 'REINFORCE':
                    match = next((e for e in existing if e['module'] == d['module'] and e['decision'] == d['decision']), None)
                    if match:
                        reinforce_decision(db_path, match['id'], commit_hash)
                        reinforced_count += 1
                    else:
                        d['event_type'] = 'NEW'
                        store_decision_with_embedding(db_path, d, embedding_bytes=emb_bytes)
                        new_count += 1
                elif event == 'CONTRADICT':
                    did = store_decision_with_embedding(db_path, d, embedding_bytes=emb_bytes)
                    flag_decision(db_path, did)
                    new_count += 1
                else:
                    store_decision_with_embedding(db_path, d, embedding_bytes=emb_bytes)
                    new_count += 1
                    
        if new_count > 0 or reinforced_count > 0:
            console.print(Panel(
                f"[bold green]{new_count}[/bold green] new decisions extracted\n[bold yellow]{reinforced_count}[/bold yellow] decisions reinforced",
                title="[bold cyan]EXTRACTION COMPLETE[/bold cyan]", border_style="cyan", expand=False
            ))
        else:
            console.print(Panel(
                "[dim]No architectural decisions found in this commit[/dim]",
                title="[dim]EXTRACTION COMPLETE[/dim]", border_style="dim", expand=False
            ))

    except Exception as e:
        err_log = klyd_dir / 'errors.log'
        with open(err_log, 'a') as f:
            f.write(f"Error extracting commit:\n{traceback.format_exc()}\n")
        return

@cli.command()
@click.option('--relevance-mode', type=click.Choice(['balanced', 'strict']), default='balanced', help='Relevance scoring mode')
def prepare_injection(relevance_mode):
    """Prepare injection file for agent sessions."""
    from .db import get_decisions_for_files
    from .injector import format_injection
    
    klyd_dir = Path('.klyd')
    if not (klyd_dir / 'memory.db').exists():
        return
        
    try:
        with console.status("[bold cyan]Preparing injection context...[/bold cyan]", spinner="dots12"):
            files_out = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True)
            files = [f for f in files_out.strip().split('\n') if f]
            
            if not files:
                with open(klyd_dir / 'injection.txt', 'w') as f:
                    f.write('')
                return
                
            db_path = str(klyd_dir / 'memory.db')
            decisions = get_decisions_for_files(db_path, files, top_k=20)
            
            # Get task description from git commit message? For now, use None.
            # In future, could be passed via --message flag.
            task_description = None
            
            injection = format_injection(
                decisions,
                db_path=db_path,
                task_description=task_description,
                relevance_mode=relevance_mode,
                top_k=None  # will use config default
            )
            with open(klyd_dir / 'injection.txt', 'w') as f:
                f.write(injection)
                
        console.print(Panel("[green]Injection file ready at .klyd/injection.txt[/green]", border_style="green", expand=False))
            
    except Exception as e:
        with open(klyd_dir / 'errors.log', 'a') as f:
            f.write(f"Error preparing injection:\n{traceback.format_exc()}\n")
        return

@cli.command()
@click.option('--relevance-mode', type=click.Choice(['balanced', 'strict']), default='balanced', help='Relevance scoring mode')
def preview_injection(relevance_mode):
    """Preview the injection file without writing it."""
    from .db import get_decisions_for_files
    from .injector import format_injection
    
    klyd_dir = Path('.klyd')
    if not (klyd_dir / 'memory.db').exists():
        console.print("[bold red]klyd is not initialized. Run `kl init`.[/bold red]")
        return
        
    try:
        with console.status("[bold cyan]Generating preview...[/bold cyan]", spinner="dots12"):
            files_out = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True)
            files = [f for f in files_out.strip().split('\n') if f]
            
            if not files:
                console.print("[yellow]No staged files. Injection would be empty.[/yellow]")
                return
                
            db_path = str(klyd_dir / 'memory.db')
            decisions = get_decisions_for_files(db_path, files, top_k=20)
            
            task_description = None
            
            injection = format_injection(
                decisions,
                db_path=db_path,
                task_description=task_description,
                relevance_mode=relevance_mode,
                top_k=None
            )
            
        console.print(Panel(
            injection,
            title="[bold cyan]Injection Preview[/bold cyan]",
            border_style="cyan",
            expand=False
        ))
            
    except Exception as e:
        console.print(f"[red]Error generating preview: {e}[/red]")
        return

@cli.command()
@click.argument('decision_id', type=int)
def pin(decision_id):
    """Pin a decision so it always appears in injection."""
    from .db import get_decision_by_id
    klyd_dir = Path('.klyd')
    db_path = klyd_dir / 'memory.db'
    if not db_path.exists():
        console.print("[bold red]klyd is not initialized. Run `kl init`.[/bold red]")
        return
    
    decision = get_decision_by_id(str(db_path), decision_id)
    if not decision:
        console.print(f"[red]Decision with id {decision_id} not found.[/red]")
        return
    
    add_pinned_decision_id(decision_id)
    console.print(Panel(
        f"[bold green]Pinned decision {decision_id}:[/bold green] {decision['decision'][:60]}...",
        border_style="green", expand=False
    ))

@cli.command()
@click.argument('decision_id', type=int, required=False)
def unpin(decision_id):
    """Unpin a decision. If no id given, unpin all."""
    if decision_id is None:
        clear_pinned_decision_ids()
        console.print("[bold yellow]All decisions unpinned.[/bold yellow]")
    else:
        remove_pinned_decision_id(decision_id)
        console.print(f"[bold yellow]Unpinned decision {decision_id}.[/bold yellow]")

@cli.command()
def review():
    """Review flagged conflicting decisions with enhanced conflict resolution."""
    from .db import get_flagged_decisions, get_active_decisions_by_module, resolve_decision
    
    klyd_dir = Path('.klyd')
    db_path = klyd_dir / 'memory.db'
    if not db_path.exists():
        console.print("[bold red]klyd is not initialized. Run `kl init`.[/bold red]")
        return

    db_str = str(db_path)
    flagged = get_flagged_decisions(db_str)
    
    if not flagged:
        console.print(Panel(
            "[bold green]All conflicts resolved. Memory is clean.[/bold green]",
            title="[bold green]NO CONFLICTS[/bold green]", border_style="green", expand=False
        ))
        return

    config_data = get_all_config()

    for i, d in enumerate(flagged):
        console.print()
        console.print(f"[bold cyan]Conflict {i+1} of {len(flagged)}[/bold cyan] in module: [bold white]{d['module']}[/bold white]")
        
        commit_ref = d.get('last_seen_commit') or "unknown commit"
        
        active = get_active_decisions_by_module(db_str, d['module'])
        old_id = None
        old_decision_text = ""
        if active:
            old = active[0]
            old_id = old['id']
            old_decision_text = old['decision']
            conf_color = "bold bright_green" if old['confidence'] == 'HIGH' else ("bold bright_yellow" if old['confidence'] == 'MEDIUM' else "dim white")
            old_panel = Panel(
                f"{old['decision']}\n\n[dim]([/dim][{conf_color}]{old['confidence']}[/{conf_color}][dim] | reinforced x{old['reinforcement_count']})[/dim]",
                title="[bold yellow]Existing Memory[/bold yellow]",
                border_style="yellow",
                expand=True
            )
        else:
            old_panel = Panel(
                "\n[dim]No existing memory found.[/dim]\n",
                title="[bold yellow]Existing Memory[/bold yellow]",
                border_style="yellow",
                expand=True
            )

        new_conf_color = "bold bright_green" if d['confidence'] == 'HIGH' else ("bold bright_yellow" if d['confidence'] == 'MEDIUM' else "dim white")
        new_panel = Panel(
            f"{d['decision']}\n\n[dim]([/dim][{new_conf_color}]{d['confidence']}[/{new_conf_color}][dim] | commit {commit_ref[:7]})[/dim]",
            title="[bold red]New Conflicting Extraction[/bold red]",
            border_style="red",
            expand=True
        )

        console.print(Columns([old_panel, new_panel]))
        
        # Show side-by-side diff
        console.print()
        console.print("[bold cyan]Side-by-side diff:[/bold cyan]")
        diff_text = _format_diff(old_decision_text, d['decision'])
        console.print(Panel(diff_text, title="[bold]Old | New[/bold]", border_style="blue", expand=False))
        
        console.print()
        console.print("  [bold green][ a ][/bold green] Accept+Merge (LLM suggests unified decision)")
        console.print("  [bold yellow][ o ][/bold yellow] ArchiveOld+AcceptNew (keep new, archive old)")
        console.print("  [bold blue][ e ][/bold blue] ManualEdit (edit the new decision)")
        console.print("  [bold red][ r ][/bold red] Reject new (keep old)")
        console.print("  [bold magenta][ w ][/bold magenta] AutoArchiveWeak (archive old if weak confidence)")
        console.print("  [bold dim][ s ][/bold dim] Skip for now")
        console.print()
        
        while True:
            choice = Prompt.ask("[bold cyan]Select action[/bold cyan]", choices=["a", "o", "e", "r", "w", "s"], show_choices=False).lower()
            if choice == 's':
                console.print("[dim]Skipped.[/dim]")
                break
            elif choice == 'a':
                # Accept+Merge: call LLM to suggest unified decision
                with console.status("[bold cyan]Asking LLM to suggest a merge...[/bold cyan]", spinner="dots12"):
                    merged_text = _call_llm_for_merge(old_decision_text, d['decision'], config_data)
                if merged_text:
                    console.print(Panel(
                        f"[bold green]LLM suggests:[/bold green]\n{merged_text}",
                        title="[bold]Merge Suggestion[/bold]",
                        border_style="green",
                        expand=False
                    ))
                    confirm = Prompt.ask("[bold cyan]Accept this merge?[/bold cyan]", choices=["y", "n"], default="y")
                    if confirm == 'y':
                        # Create a new version from the old decision with the merged text
                        new_version_id = create_decision_version(db_str, old_id, {
                            'decision': merged_text,
                            'confidence': 'MEDIUM',
                            'event_type': 'NEW',
                            'last_seen_commit': d.get('last_seen_commit')
                        })
                        # Archive the old decision
                        merge_decisions(db_str, new_version_id, old_id)
                        # Archive the new conflicting decision
                        merge_decisions(db_str, new_version_id, d['id'])
                        console.print(Panel(
                            "[bold green]Merged decision created. Old and conflicting decisions archived.[/bold green]",
                            border_style="green", expand=False
                        ))
                        break
                    else:
                        console.print("[dim]Merge cancelled. Choose another option.[/dim]")
                        continue
                else:
                    console.print("[red]LLM merge failed. Please choose another option.[/red]")
                    continue
            elif choice == 'o':
                # ArchiveOld+AcceptNew
                if old_id:
                    merge_decisions(db_str, d['id'], old_id)
                resolve_decision(db_str, d['id'], 'accept', old_id=old_id)
                console.print(Panel(
                    "[bold yellow]Old decision archived. New decision accepted.[/bold yellow]",
                    border_style="yellow", expand=False
                ))
                break
            elif choice == 'e':
                new_text = click.edit(d['decision'])
                if new_text is not None:
                    new_text = new_text.strip()
                    resolve_decision(db_str, d['id'], 'edit', old_id=old_id, new_text=new_text)
                    console.print(Panel(
                        "[bold blue]Saved edited decision. Memory updated.[/bold blue]",
                        border_style="blue", expand=False
                    ))
                    break
                else:
                    console.print("[red]Edit cancelled. Please choose an option.[/red]")
            elif choice == 'r':
                resolve_decision(db_str, d['id'], 'reject')
                console.print(Panel(
                    "[bold red]Rejected new decision. Existing memory preserved.[/bold red]",
                    border_style="red", expand=False
                ))
                break
            elif choice == 'w':
                # AutoArchiveWeak: archive old if its confidence is LOW
                if old_id:
                    old_decision = get_decision_by_id(db_str, old_id)
                    if old_decision and old_decision['confidence'] == 'LOW':
                        merge_decisions(db_str, d['id'], old_id)
                        resolve_decision(db_str, d['id'], 'accept', old_id=old_id)
                        console.print(Panel(
                            "[bold magenta]Old decision (LOW confidence) archived. New decision accepted.[/bold magenta]",
                            border_style="magenta", expand=False
                        ))
                    else:
                        console.print("[yellow]Old decision confidence is not LOW. Skipping auto-archive.[/yellow]")
                        continue
                else:
                    console.print("[yellow]No old decision to archive.[/yellow]")
                    continue
                break

@cli.command()
def status():
    """Show the current memory store status."""
    klyd_dir = Path('.klyd')
    db_path = klyd_dir / 'memory.db'
    if not db_path.exists():
        console.print("[bold red]klyd is not initialized. Run `kl init`.[/bold red]")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM decisions WHERE archived = 0")
    all_decisions = [dict(r) for r in cur.fetchall()]
    conn.close()

    active = [d for d in all_decisions if d['flagged'] == 0]
    flagged = [d for d in all_decisions if d['flagged'] == 1]
    
    active.sort(key=lambda x: x['reinforcement_count'], reverse=True)

    table = Table(box=rich.box.SIMPLE, header_style="bold cyan", show_edge=False, padding=(0, 2))
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Decision", style="white")
    table.add_column("Module", style="bold blue")
    table.add_column("Confidence")
    table.add_column("Event")
    table.add_column("Reinforcements", justify="right")

    pinned_ids = get_pinned_decision_ids()

    for d in active:
        conf_color = "bold bright_green" if d['confidence'] == 'HIGH' else ("bold bright_yellow" if d['confidence'] == 'MEDIUM' else "dim white")
        conf_styled = f"[{conf_color}]{d['confidence']}[/{conf_color}]"
        
        event_color = "bright_blue" if d.get('event_type') == 'NEW' else ("bright_magenta" if d.get('event_type') == 'REINFORCE' else "bold bright_red")
        event_styled = f"[{event_color}]{d.get('event_type', 'NEW')}[/{event_color}]"
        
        id_str = str(d['id'])
        if d['id'] in pinned_ids:
            id_str = f"[bold yellow]📌 {id_str}[/bold yellow]"
        
        table.add_row(
            id_str,
            d['decision'],
            d['module'],
            conf_styled,
            event_styled,
            f"x{d['reinforcement_count']}"
        )
    
    if not active:
        table.add_row("", "[dim]No active architectural decisions stored yet.[/dim]", "", "", "", "")

    console.print(Panel(table, title="[bold cyan] Architectural Memory [/bold cyan]", border_style="cyan", expand=False, padding=(1, 2)))

    if flagged:
        console.print()
        flagged_table = Table(box=rich.box.SIMPLE, show_edge=False, padding=(0, 2))
        flagged_table.add_column("ID", style="dim", max_width=8)
        flagged_table.add_column("Decision", style="bold white")
        flagged_table.add_column("Module", style="bold blue")
        flagged_table.add_column("Confidence")
        
        for d in flagged:
            conf_color = "bold bright_green" if d['confidence'] == 'HIGH' else ("bold bright_yellow" if d['confidence'] == 'MEDIUM' else "dim white")
            conf_styled = f"[{conf_color}]{d['confidence']}[/{conf_color}]"
            
            flagged_table.add_row(
                str(d['id'])[:8],
                d['decision'],
                d['module'],
                conf_styled
            )

        warning_panel = Panel(
            flagged_table,
            title="[bold white on red] NEEDS REVIEW - CONFLICTS DETECTED [/bold white on red]",
            border_style="red",
            expand=False,
            padding=(1, 2)
        )
        console.print(warning_panel)

    console.print()
    console.print(f"[dim]Summary:[/dim] [cyan]{len(active)} active[/cyan] | [red]{len(flagged)} pending review[/red]")
