from __future__ import annotations
from typing import Dict, Any
from .mappings import g_add, g_merge

def mvr_pairs(agent, base: Dict[str, Any], mutated: Dict[str, Any], op: str) -> bool:
    b_base, _ = agent.decide_and_ask(base)
    b_mut, _ = agent.decide_and_ask(mutated)

    if op == "g_a":
        return b_mut > b_base
    if op == "g_m":
        if len(mutated.get("unresolved_items", [])) == 0:
            return b_mut == 1
        return False
    return False
