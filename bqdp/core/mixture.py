# bqdp/core/mixture.py
import math
from typing import Dict

import torch





_MINIMUM_LIKELIHOOD = 1e-12





class ExpertMixer:
    def __init__(self, initial_weights: Dict[str, float], temperature: float=1.0, share: float=0.005, floor: float=1e-4):
        if not initial_weights:
            raise ValueError("initial_weights must contain at least one expert")
        
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.0")
        
        if not 0.0 <= share < 1.0:
            raise ValueError("share must be in [0.0, 1.0)")
        
        if not 0.0 <= floor < 1.0 / len(initial_weights):
            raise ValueError("floor must leave room for every expert")

        total = sum(max(weight, 0.0) for weight in initial_weights.values())
        
        if total <= 0.0:
            raise ValueError("at least one initial weight must be > 0.0")

        self.weights = {name: max(weight, 0.0) / total for name, weight in initial_weights.items()}
        self.temperature = temperature
        self.share = share
        self.floor = floor
        self.log_loss = {name: 0.0 for name in self.weights}
        self.observed = 0


    @property
    def mean_log_loss(self) -> Dict[str, float]:
        if self.observed == 0:
            return {name: 0.0 for name in self.log_loss}
        
        return {name: value / self.observed for name, value in self.log_loss.items()}


    def observe(self, distributions: Dict[str, torch.Tensor], actual: int) -> None:
        likelihoods = {}
        
        for name in self.weights:
            probability = max(float(distributions[name][actual]), _MINIMUM_LIKELIHOOD)
            likelihoods[name] = probability
            self.log_loss[name] -= math.log(probability)

        updated = {
            name: self.weights[name] * (likelihoods[name] ** self.temperature)
            for name in self.weights
        }
        total = sum(updated.values())
        
        if total <= 0.0:
            return

        # fixed share keeps a dormant expert recoverable when the process drifts back toward it
        count = len(updated)
        
        for name in updated:
            weight = updated[name] / total
            weight = (1.0 - self.share) * weight + self.share / count
            updated[name] = max(weight, self.floor)

        total = sum(updated.values())
        
        self.weights = {name: weight / total for name, weight in updated.items()}
        self.observed += 1


    def state_dict(self) -> dict:
        return {
            "weights": dict(self.weights),
            "log_loss": dict(self.log_loss),
            "observed": self.observed,
        }


    def load_state_dict(self, state: dict) -> None:
        self.weights = dict(state["weights"])
        self.log_loss = dict(state["log_loss"])
        self.observed = state["observed"]