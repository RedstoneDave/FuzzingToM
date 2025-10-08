from __future__ import annotations
from typing import Tuple, Optional, List
from ..types import TestCase
from .base import BaseAgent

def _cover_slots_prompt(unresolved: List[str], slot_keywords: dict) -> str:
    parts = []
    for slot in unresolved:
        kws = slot_keywords.get(slot, [slot])
        parts.append(f"{kws[0] if kws else slot}?")
    return "To proceed, could you clarify: " + "; ".join(parts)

class HeuristicAgent(BaseAgent):
    def __init__(self, max_len: int = 160) -> None:
        self.max_len = max_len

    def decide_and_ask(self, tc: TestCase) -> Tuple[int, Optional[str]]:
        if not tc.unresolved_items: return 0, None
        q = _cover_slots_prompt(tc.unresolved_items, tc.slot_keywords)
        if len(q) > self.max_len: q = q[: self.max_len-3] + "..."
        return 1, q
