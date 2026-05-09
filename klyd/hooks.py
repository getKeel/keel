import os
import shutil
import stat
from pathlib import Path

def get_template_path(hook_name):
    # Try to find the hooks directory relative to this file
    base_dir = Path(__file__).resolve().parent.parent
    hook_path = base_dir / 'hooks' / f"{hook_name}.sh"
    return hook_path

def install_hooks():
    git_dir = Path('.git')
    if not git_dir.is_dir():
        raise RuntimeError("Not a git repository. Please run in the root of a git repo.")

    git_hooks_dir = git_dir / 'hooks'
    git_hooks_dir.mkdir(exist_ok=True)

    for hook in ['post-commit', 'pre-commit']:
        src = get_template_path(hook)
        if not src.exists():
            raise RuntimeError(f"Hook template not found: {src}")
        
        dest = git_hooks_dir / hook
        shutil.copy2(src, dest)
        
        # chmod +x
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

def uninstall_hooks():
    git_dir = Path('.git')
    if not git_dir.is_dir():
        return
        
    git_hooks_dir = git_dir / 'hooks'
    for hook in ['post-commit', 'pre-commit']:
        dest = git_hooks_dir / hook
        if dest.exists():
            # Read as bytes to avoid UnicodeDecodeError on binary files
            content = dest.read_bytes()
            if b"klyd" in content:
                dest.unlink()
