

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def load_env() -> None:
    
    global _loaded
    if _loaded:
        return
    _loaded = True

    from dotenv import load_dotenv

    env_file = os.getenv("MMSHOPBENCH_EVAL_ENV_FILE")
    if env_file:
        load_dotenv(env_file, override=False)
        return

    for candidate in [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return
