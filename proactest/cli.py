from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
from typing import List
from .dataset import JSONLDataset
from .types import TestCase
from .agents import HeuristicAgent, ReactiveNeverAsk, LLMAgent
from .lsu import LLMUserSim
from .harness import run_suite
from .mappings import g_add, g_remove, g_change, g_merge
from .mvr import mvr_pairs
from .metrics import MetricAccumulators

def _load_cases(path: str | Path) -> List[TestCase]:
    return JSONLDataset(path).to_list()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/toy.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    agent_name = cfg.get("agent", "heuristic")
    if agent_name == "heuristic":
        agent = HeuristicAgent(max_len=cfg.get("L_max", 160))
    elif agent_name == "never":
        agent = ReactiveNeverAsk()
    elif agent_name == "llm":
        prov = cfg.get("llm", {}).get("provider", "openai")
        model = cfg.get("llm", {}).get("model", "gpt-4o-mini")
        temp = float(cfg.get("llm", {}).get("temperature", 0.2))
        agent = LLMAgent(provider=prov, model=model, temperature=temp)
    else:
        agent = HeuristicAgent(max_len=cfg.get("L_max", 160))

    lsu = LLMUserSim(seed=cfg.get("seed", 0))
    out_dir = cfg.get("output_dir", "outputs/run")
    base_cases = _load_cases(cfg["dataset"]["test"])

    summary = run_suite(base_cases, agent, lsu, out_dir)

    acc = MetricAccumulators(l_max=cfg.get("L_max", 160))
    for base in base_cases:
        acc.update_mvr("g_a", mvr_pairs(agent, base, g_add(base), "g_a"))
        acc.update_mvr("g_r", mvr_pairs(agent, base, g_remove(base, extra_slots=list(base.slot_keywords.keys())), "g_r"))
        acc.update_mvr("g_c", mvr_pairs(agent, base, g_change(base), "g_c"))
    by_cid = {}
    for tc in base_cases:
        by_cid.setdefault(tc.conversation_id, []).append(tc)
    for cid, group in by_cid.items():
        if len(group) >= 2:
            m = g_merge(group[0], group[1])
            acc.update_mvr("g_m", mvr_pairs(agent, group[0], m, "g_m"))

    mvr_summary = acc.finalize()
    combined = {**summary, **{k: v for k, v in mvr_summary.items() if k.startswith("MVR_")}}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "summary_with_mvr.json").write_text(json.dumps(combined, indent=2))
    print(json.dumps(combined, indent=2))

if __name__ == "__main__":
    main()
