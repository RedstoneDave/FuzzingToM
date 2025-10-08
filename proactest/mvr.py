from __future__ import annotations
from .types import TestCase
from .mappings import g_add, g_remove, g_change, g_merge

def mvr_pairs(agent, base: TestCase, mutated: TestCase, op: str) -> bool:
    b_base, _ = agent.decide_and_ask(base)
    b_mut, _ = agent.decide_and_ask(mutated)

    if op == "g_a":
        return b_mut > b_base
    if op == "g_r":
        return b_mut < b_base
    if op == "g_c":
        if len(base.unresolved_items) == 0:
            return b_mut != b_base
        return False
    if op == "g_m":
        if len(mutated.unresolved_items) == 0:
            return b_mut == 1
        return False
    return False
