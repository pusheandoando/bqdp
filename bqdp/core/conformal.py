# bqdp/core/conformal.py
import math
from collections import deque
from typing import List, Optional

import torch





class ConformalCalibrator:
    def __init__(self, target_alpha: float=0.05, step_size: float=0.01, capacity: int=4096):
        if not 0.0 < target_alpha < 1.0:
            raise ValueError("target_alpha must be in (0.0, 1.0)")
        if step_size <= 0.0:
            raise ValueError("step_size must be > 0.0")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")

        self.target_alpha = target_alpha
        self.step_size = step_size
        self.scores = deque(maxlen=capacity)
        self.effective_alpha = target_alpha
        self.covered = 0
        self.observed = 0


    @property
    def coverage(self) -> float:
        if self.observed == 0:
            return 0.0
        
        return self.covered / self.observed


    def prediction_set(self, distribution: torch.Tensor, max_size: int, alpha: Optional[float]=None) -> List[int]:
        threshold = self._threshold(alpha)
        candidates = distribution[1:]
        ordered, indices = torch.sort(candidates, descending=True)
        cumulative = torch.cumsum(ordered, dim=0)
        reached = torch.nonzero(cumulative >= threshold)

        size = int(reached[0]) + 1 if reached.numel() > 0 else int(ordered.numel())
        size = max(1, min(size, max_size))
        
        return [int(index) + 1 for index in indices[:size]]


    def observe(self, distribution: torch.Tensor, actual: int) -> bool:
        score = self._score(distribution, actual)
        covered = score <= self._threshold()

        self.observed += 1
        self.covered += int(covered)

        # adaptive conformal inference keeps long run coverage under distribution shift
        error = 0.0 if covered else 1.0
        adjusted = self.effective_alpha + self.step_size * (self.target_alpha - error)
        self.effective_alpha = min(1.0, max(0.0, adjusted))
        self.scores.append(score)
        
        return covered


    def _threshold(self, alpha: Optional[float]=None) -> float:
        level = self.effective_alpha if alpha is None else alpha
        level = min(max(level, 0.0), 1.0)

        count = len(self.scores)
        if count == 0:
            return 1.0 - level

        rank = max(1, math.ceil((count + 1) * (1.0 - level)))
        if rank > count:
            return 1.0
        
        return sorted(self.scores)[rank - 1]


    @staticmethod
    def _score(distribution: torch.Tensor, actual: int) -> float:
        probability = distribution[actual]
        
        return float((distribution * (distribution >= probability)).sum())


    def state_dict(self) -> dict:
        return {
            "scores": list(self.scores),
            "effective_alpha": self.effective_alpha,
            "covered": self.covered,
            "observed": self.observed,
        }


    def load_state_dict(self, state: dict) -> None:
        self.scores.clear()
        self.scores.extend(state["scores"])
        self.effective_alpha = state["effective_alpha"]
        self.covered = state["covered"]
        self.observed = state["observed"]