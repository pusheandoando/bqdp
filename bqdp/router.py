# bqdp/router.py
from pathlib import Path





def resolve_state_dir(name: str="default") -> str:
    path = Path.home() / ".bqdp" / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)