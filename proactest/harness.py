from __future__ import annotations
import csv, json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from .types import TestCase
from .lsu import LLMUserSim
from .metrics import MetricAccumulators

def ground_truth_should_ask(tc: TestCase) -> int:
    return 1 if len(tc.unresolved_items) > 0 else 0

def is_on_target(question: str, tc: TestCase) -> bool:
    q = question.lower()
    for slot in tc.unresolved_items:
        for kw in tc.slot_keywords.get(slot, [slot]):
            if kw.lower() in q:
                return True
    return False

def run_suite(cases: List[TestCase], agent, lsu: LLMUserSim, out_dir: str | Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    acc = MetricAccumulators()
    for tc in tqdm(cases, desc="Running test suite"):
        should = ground_truth_should_ask(tc)
        b, a = agent.decide_and_ask(tc)
        acc.update_b(correct=(b == should))

        delta_t = 0
        on_target = False
        a_text = a or ""
        if b == 1 and a:
            on_target = is_on_target(a, tc)
            _, revealed = lsu.answer(tc, a)
            delta_t = len(revealed)
            acc.update_question(on_target=on_target, delta_t=delta_t, q_len=len(a_text))

        trace_rows.append({
            "conversation_id": tc.conversation_id,
            "turn_index": tc.turn_index,
            "should_ask": should,
            "b": b,
            "a": a_text,
            "delta_t": delta_t,
            "on_target": int(on_target),
            "unresolved": len(tc.unresolved_items),
        })

    if trace_rows:
        with (out_dir / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            w.writerows(trace_rows)
    summary = acc.finalize()
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary
