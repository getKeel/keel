import click
import subprocess
import json
import traceback
from .hooks import install_hooks
from .config import init_config, set_config, get_config, get_all_config
from .db import get_schema_path, migrate_db
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
def config(api_key, openai_key, openrouter_key, gemini_key, groq_key, model, show):
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
            
    if changes:
        console.print(Panel("[bold green]Configuration saved successfully[/bold green]", title="[bold green]DONE[/bold green]", border_style="green", expand=False))
    else:
        console.print("[yellow]Usage:[/yellow] kl config --api-key ... --openai-key ... --model ...\nOr use --show to display current configuration.")

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
                top_k=10
            )
            with open(klyd_dir / 'injection.txt', 'w') as f:
                f.write(injection)
                
        console.print(Panel("[green]Injection file ready at .klyd/injection.txt[/green]", border_style="green", expand=False))
            
    except Exception as e:
        with open(klyd_dir / 'errors.log', 'a') as f:
            f.write(f"Error preparing injection:\n{traceback.format_exc()}\n")
        return

@cli.command()
def review():
    """Review flagged conflicting decisions."""
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

    for i, d in enumerate(flagged):
        console.print()
        console.print(f"[bold cyan]Conflict {i+1} of {len(flagged)}[/bold cyan] in module: [bold white]{d['module']}[/bold white]")
        
        commit_ref = d.get('last_seen_commit') or "unknown commit"
        
        active = get_active_decisions_by_module(db_str, d['module'])
        old_id = None
        if active:
            old = active[0]
            old_id = old['id']
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
        
        console.print()
        console.print("  [bold green][ a ][/bold green] Accept new (archive old)    [bold red][ r ][/bold red] Reject new (keep old)")
        console.print("  [bold yellow][ e ][/bold yellow] Edit manually               [bold blue][ s ][/bold blue] Skip for now")
        console.print()
        
        while True:
            choice = Prompt.ask("[bold cyan]Select action[/bold cyan]", choices=["a", "r", "e", "s"], show_choices=False).lower()
            if choice == 's':
                console.print("[dim]Skipped.[/dim]")
                break
            elif choice == 'a':
                resolve_decision(db_str, d['id'], 'accept', old_id=old_id)
                console.print(Panel(
                    "[bold green]Accepted new decision. Memory updated.[/bold green]",
                    border_style="green", expand=False
                ))
                break
            elif choice == 'r':
                resolve_decision(db_str, d['id'], 'reject')
                console.print(Panel(
                    "[bold red]Rejected new decision. Existing memory preserved.[/bold red]",
                    border_style="red", expand=False
                ))
                break
            elif choice == 'e':
                new_text = click.edit(d['decision'])
                if new_text is not None:
                    new_text = new_text.strip()
                    resolve_decision(db_str, d['id'], 'edit', old_id=old_id, new_text=new_text)
                    console.print(Panel(
                        "[bold yellow]Saved edited decision. Memory updated.[/bold yellow]",
                        border_style="yellow", expand=False
                    ))
                    break
                else:
                    console.print("[red]Edit cancelled. Please choose an option.[/red]")

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

    for d in active:
        conf_color = "bold bright_green" if d['confidence'] == 'HIGH' else ("bold bright_yellow" if d['confidence'] == 'MEDIUM' else "dim white")
        conf_styled = f"[{conf_color}]{d['confidence']}[/{conf_color}]"
        
        event_color = "bright_blue" if d.get('event_type') == 'NEW' else ("bright_magenta" if d.get('event_type') == 'REINFORCE' else "bold bright_red")
        event_styled = f"[{event_color}]{d.get('event_type', 'NEW')}[/{event_color}]"
        
        table.add_row(
            str(d['id'])[:8],
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
