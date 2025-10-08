from __future__ import annotations
import random
from copy import deepcopy
from typing import List
from .types import TestCase

def g_add(tc: TestCase) -> TestCase:
    out = deepcopy(tc)
    if not out.unresolved_items: return out
    pick = random.choice(out.unresolved_items)
    out.resolved_items[pick] = out.resolved_items.get(pick, f"{pick}_val")
    out.unresolved_items = [s for s in out.unresolved_items if s != pick]
    return out

def g_remove(tc: TestCase, extra_slots: List[str] | None = None) -> TestCase:
    out = deepcopy(tc)
    candidates = list(out.resolved_items.keys())
    if not candidates and extra_slots:
        add = random.choice(extra_slots)
        if add not in out.unresolved_items: out.unresolved_items.append(add)
        return out
    if not candidates: return out
    pick = random.choice(candidates)
    out.resolved_items.pop(pick, None)
    if pick not in out.unresolved_items: out.unresolved_items.append(pick)
    return out

def g_change(tc: TestCase) -> TestCase:
    out = deepcopy(tc)
    if not out.resolved_items: return out
    pick = random.choice(list(out.resolved_items.keys()))
    out.resolved_items[pick] = out.resolved_items[pick] + "_alt"
    return out

def g_merge(tc1: TestCase, tc2: TestCase) -> TestCase:
    out = deepcopy(tc1)
    if isinstance(tc1.conversation, list) and isinstance(tc2.conversation, list):
        seen, merged = set(), []
        for c in tc1.conversation + tc2.conversation:
            key = (c.get("role"), c.get("text"))
            if key in seen: continue
            seen.add(key); merged.append(c)
        out.conversation = merged
    out.resolved_items = {**tc1.resolved_items, **tc2.resolved_items}
    unresolved = set(tc1.unresolved_items) | set(tc2.unresolved_items)
    out.unresolved_items = sorted([u for u in unresolved if u not in out.resolved_items])
    sk = {}
    for d in [tc1.slot_keywords, tc2.slot_keywords]:
        for k, v in d.items():
            sk.setdefault(k, [])
            for x in v:
                if x not in sk[k]: sk[k].append(x)
    out.slot_keywords = sk
    return out
