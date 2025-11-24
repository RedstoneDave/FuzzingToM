from __future__ import annotations
import random
from copy import deepcopy
from typing import Dict, Any

def g_add(tc: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(tc)
    unresolved = out.get("unresolved_items", [])
    resolved = out.get("resolved_items", {})
    if not unresolved: return out
    pick = random.choice(unresolved)
    resolved[pick] = resolved.get(pick, f"{pick}_val")
    out["resolved_items"] = resolved
    out["unresolved_items"] = [s for s in unresolved if s != pick]
    return out

def g_merge(tc1: Dict[str, Any], tc2: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(tc1)
    conv1, conv2 = str(tc1.get("conversation", "")), str(tc2.get("conversation", ""))
    out["conversation"] = (conv1 + "\n" + conv2).strip() if conv2 else conv1
    resolved1 = tc1.get("resolved_items", {})
    resolved2 = tc2.get("resolved_items", {})
    out["resolved_items"] = {**resolved1, **resolved2}
    unresolved = set(tc1.get("unresolved_items", [])) | set(tc2.get("unresolved_items", []))
    out["unresolved_items"] = sorted([u for u in unresolved if u not in out["resolved_items"]])
    sk = {}
    for d in [tc1.get("slot_keywords", {}), tc2.get("slot_keywords", {})]:
        for k, v in d.items():
            sk.setdefault(k, [])
            for x in v:
                if x not in sk[k]: sk[k].append(x)
    out["slot_keywords"] = sk
    return out
