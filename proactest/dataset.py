from __future__ import annotations
import json
from typing import Iterable, List
from dataclasses import asdict
from pathlib import Path
from .types import TestCase

class JSONLDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def __iter__(self) -> Iterable[TestCase]:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                yield TestCase(**obj)

    def to_list(self) -> List[TestCase]:
        return list(iter(self))

def write_jsonl(cases: List[TestCase], path: str | Path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in cases:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
