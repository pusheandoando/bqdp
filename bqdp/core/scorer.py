# bqdp/core/scorer.py
from typing import Dict, List

import torch

from .mixture import ExpertMixer





class Scorer:
    def __init__(self, initial_weights: Dict[str, float], smoothing: float=1e-3, temperature: float=1.0, share: float=0.005, floor: float=1e-4):
        if smoothing <= 0.0:
            raise ValueError("smoothing must be > 0.0")

        self.smoothing = smoothing
        self.mixer = ExpertMixer(initial_weights, temperature, share, floor)


    @property
    def weights(self) -> Dict[str, float]:
        return dict(self.mixer.weights)


    @property
    def mean_log_loss(self) -> Dict[str, float]:
        return self.mixer.mean_log_loss


    def normalize(self, experts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = {}

        for name, values in experts.items():
            prior = values.clamp_min(0.0) + self.smoothing
            prior[0] = 0.0
            normalized[name] = prior / prior.sum()

        return normalized


    def blend(self, normalized: Dict[str, torch.Tensor]) -> torch.Tensor:
        names = sorted(normalized)
        blended = torch.zeros_like(normalized[names[0]])

        # a linear mixture bounds the result below by every weighted expert, so no expert can veto
        for name in names:
            blended.add_(normalized[name], alpha=self.mixer.weights[name])

        blended[0] = 0.0

        return blended / blended.sum()


    def observe(self, normalized: Dict[str, torch.Tensor], actual: int) -> None:
        self.mixer.observe(normalized, actual)


    @staticmethod
    def top_k(distribution: torch.Tensor, k: int) -> List[int]:
        _, indices = torch.topk(distribution[1:], k)
        
        return [int(index) + 1 for index in indices]


    def state_dict(self) -> dict:
        return {"mixer": self.mixer.state_dict()}


    def load_state_dict(self, state: dict) -> None:
        self.mixer.load_state_dict(state["mixer"])