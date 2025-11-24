from __future__ import annotations
from typing import Dict, List, Tuple, Any

class LLMUserSim:
    def __init__(self, seed: int = 0) -> None:
        import random; random.seed(seed)

    def _slot_hit(self, question: str, slot: str, slot_keywords: Dict[str, List[str]]) -> bool:
        kws = slot_keywords.get(slot, [slot])
        q = question.lower()
        return any(kw.lower() in q for kw in kws)

    def answer(self, tc: Dict[str, Any], question: str) -> Tuple[str, List[str]]:
        revealed = []
        for slot in tc.get("unresolved_items", []):
            if self._slot_hit(question, slot, tc.get("slot_keywords", {})):
                revealed.append(slot)
        parts = []
        for s in revealed:
            val = tc.get("resolved_items", {}).get(s, f"{s}_value")
            parts.append(f"{s}: {val}")
        if parts:
            return ", ".join(parts), revealed
        return "no preference", []
