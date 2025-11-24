from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict

@dataclass
class MetricAccumulators:
    """
    Simplified metrics:
    - AskRate: proportion of turns where the agent asks something (b==1).
    - AskRate_by_difficulty: ask rate bucketed by (N_missing, M_known), where
      lower N and higher M are considered harder.
    """
    total_cases: int = 0
    asked_cases: int = 0
    diff_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    diff_asked: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mvr_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mvr_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, asked: bool, n_missing: int, m_known: int):
        self.total_cases += 1
        key = f"N{n_missing}_M{m_known}"
        self.diff_totals[key] += 1
        if asked:
            self.asked_cases += 1
            self.diff_asked[key] += 1

    def update_mvr(self, op: str, violated: bool):
        self.mvr_totals[op] += 1
        if violated:
            self.mvr_counts[op] += 1

    def finalize(self) -> Dict[str, float | Dict[str, float]]:
        res: Dict[str, float | Dict[str, float]] = {}
        res["AskRate"] = (self.asked_cases / self.total_cases) if self.total_cases else 0.0
        by_diff = {}
        for key, tot in self.diff_totals.items():
            by_diff[key] = (self.diff_asked.get(key, 0) / tot) if tot else 0.0
        res["AskRate_by_difficulty"] = by_diff
        for op, tot in self.mvr_totals.items():
            res[f"MVR_{op}"] = (self.mvr_counts[op] / tot) if tot else 0.0
        return res
