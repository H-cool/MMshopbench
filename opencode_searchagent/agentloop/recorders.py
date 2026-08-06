from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .types import to_jsonable


class Recorder(Protocol):
    def record(self, run_record: dict[str, Any]) -> None:
        ...


class JsonlRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, run_record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(to_jsonable(run_record), ensure_ascii=False) + "\n")
