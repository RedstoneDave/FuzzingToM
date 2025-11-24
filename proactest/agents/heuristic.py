from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
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

    def decide_and_ask(self, tc: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        unresolved = tc.get("unresolved_items", [])
        slot_keywords = tc.get("slot_keywords", {})
        if not unresolved: return 0, None
        q = _cover_slots_prompt(unresolved, slot_keywords)
        if len(q) > self.max_len: q = q[: self.max_len-3] + "..."
        return 1, q
