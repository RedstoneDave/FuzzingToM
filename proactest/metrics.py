from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
from collections import defaultdict

@dataclass
class MetricAccumulators:
    total_cases: int = 0
    correct_b: int = 0
    asked_turns: int = 0
    on_target_questions: int = 0
    low_utility_or_overlong: int = 0
    l_max: int = 160
    mvr_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    mvr_totals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update_b(self, correct: bool):
        self.total_cases += 1
        if correct:
            self.correct_b += 1

    def update_question(self, on_target: bool, delta_t: int, q_len: int):
        self.asked_turns += 1
        if on_target:
            self.on_target_questions += 1
        if delta_t == 0 or q_len > self.l_max:
            self.low_utility_or_overlong += 1

    def update_mvr(self, op: str, violated: bool):
        self.mvr_totals[op] += 1
        if violated:
            self.mvr_counts[op] += 1

    def finalize(self) -> Dict[str, float]:
        res = {}
        res["Acc_b"] = (self.correct_b / self.total_cases) if self.total_cases else 0.0
        res["AP"] = (self.on_target_questions / self.asked_turns) if self.asked_turns else 0.0
        res["LOR"] = (self.low_utility_or_overlong / self.asked_turns) if self.asked_turns else 0.0
        for op, tot in self.mvr_totals.items():
            res[f"MVR_{op}"] = (self.mvr_counts[op] / tot) if tot else 0.0
        return res
