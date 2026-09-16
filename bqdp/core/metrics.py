# bqdp/core/metrics.py
import math
from collections import deque
from typing import Dict

import torch





class StreamMetrics:
    def __init__(self, window: int=512):
        if window < 1:
            raise ValueError("window must be >= 1")

        self.steps = 0
        self.hits = 0
        self.reciprocal_rank_total = 0.0
        self.discounted_gain_total = 0.0
        self.rank_total = 0
        self.set_size_total = 0
        self.covered = 0
        self.evaluation_k = 0
        self.recent_ranks = deque(maxlen=window)


    def observe(self, distribution: torch.Tensor, actual: int, k: int, set_size: int, covered: bool) -> None:
        rank = int((distribution > distribution[actual]).sum()) + 1

        self.steps += 1
        self.evaluation_k = k
        self.rank_total += rank
        self.reciprocal_rank_total += 1.0 / rank
        
        if rank <= k:
            self.hits += 1
            self.discounted_gain_total += 1.0 / math.log2(rank + 1)
        
        self.set_size_total += set_size
        self.covered += int(covered)
        self.recent_ranks.append(rank)


    def snapshot(self) -> Dict[str, float]:
        if self.steps == 0:
            return {"steps": 0.0}

        steps = float(self.steps)
        recent = len(self.recent_ranks)
        recent_hits = sum(1 for rank in self.recent_ranks if rank <= self.evaluation_k)
        
        return {
            "steps": steps,
            "evaluation_k": float(self.evaluation_k),
            "recall_at_k": self.hits / steps,
            "mrr": self.reciprocal_rank_total / steps,
            "ndcg_at_k": self.discounted_gain_total / steps,
            "mean_rank": self.rank_total / steps,
            "mean_set_size": self.set_size_total / steps,
            "empirical_coverage": self.covered / steps,
            "recent_recall_at_k": recent_hits / recent,
            "recent_mean_rank": sum(self.recent_ranks) / recent,
        }


    def state_dict(self) -> dict:
        return {
            "steps": self.steps,
            "hits": self.hits,
            "reciprocal_rank_total": self.reciprocal_rank_total,
            "discounted_gain_total": self.discounted_gain_total,
            "rank_total": self.rank_total,
            "set_size_total": self.set_size_total,
            "covered": self.covered,
            "evaluation_k": self.evaluation_k,
            "recent_ranks": list(self.recent_ranks),
        }


    def load_state_dict(self, state: dict) -> None:
        self.steps = state["steps"]
        self.hits = state["hits"]
        self.reciprocal_rank_total = state["reciprocal_rank_total"]
        self.discounted_gain_total = state["discounted_gain_total"]
        self.rank_total = state["rank_total"]
        self.set_size_total = state["set_size_total"]
        self.covered = state["covered"]
        self.evaluation_k = state["evaluation_k"]
        self.recent_ranks.clear()
        self.recent_ranks.extend(state["recent_ranks"])