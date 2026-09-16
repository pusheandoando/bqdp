# bqdp/router.py
from pathlib import Path





def resolve_state_dir(name: str="default") -> str:
    candidate = Path(name).expanduser()

    # bare names stay namespaced under the user home, explicit paths are honoured as given
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = Path.home() / ".bqdp" / candidate

    candidate.mkdir(parents=True, exist_ok=True)
    
    return str(candidate)