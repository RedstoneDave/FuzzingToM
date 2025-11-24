from __future__ import annotations
import csv, json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from .metrics import MetricAccumulators

def run_suite(cases: List[Dict[str, Any]], agent, out_dir: str | Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_rows = []
    acc = MetricAccumulators()
    for tc in tqdm(cases, desc="Running test suite"):
        b, a = agent.decide_and_ask(tc)
        n_missing = len(tc.get("unresolved_items", []))
        m_known = len(tc.get("resolved_items", {}))
        acc.update(asked=(b == 1), n_missing=n_missing, m_known=m_known)

        trace_rows.append({
            "conversation_id": tc.get("conversation_id", ""),
            "turn_index": tc.get("turn_index", 0),
            "b": b,
            "a": a or "",
            "n_missing": n_missing,
            "m_known": m_known,
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
