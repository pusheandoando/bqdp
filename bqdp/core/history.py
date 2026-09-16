# bqdp/core/history.py
import math
from typing import List, Tuple

import torch





class History:
    def __init__(self, table_size: int, decay: float = 0.1, capacity: int = 200000):
        if table_size < 2:
            raise ValueError("table_size must be >= 2")
        
        if decay < 0.0:
            raise ValueError("decay must be >= 0.0")
        
        if capacity < 2:
            raise ValueError("capacity must be >= 2")

        self.decay = decay
        self.capacity = capacity
        self.sequence: List[int] = []
        self.total_observed = 0
        self._recency = torch.zeros(table_size, dtype=torch.float32)
        self._decay_factor = math.exp(-decay)


    def append(self, item_id: int) -> None:
        # equivalent to the exponential sum over positions, maintained incrementally
        self._recency.mul_(self._decay_factor)
        self._recency[item_id] += 1.0

        self.sequence.append(item_id)
        self.total_observed += 1
        
        if len(self.sequence) > 2 * self.capacity:
            del self.sequence[: len(self.sequence) - self.capacity]


    def temporal_weights(self) -> torch.Tensor:
        return self._recency


    def recent(self, window: int) -> List[int]:
        if window <= 0:
            return []
        
        return self.sequence[-window:]


    def tail(self, length: int) -> List[int]:
        return self._padded_window(len(self.sequence), length)


    def training_pairs(self, batch_size: int, length: int, rng) -> Tuple[List[List[int]], List[List[int]]]:
        available = len(self.sequence)
        
        if available < 2:
            return [], []

        anchors = [available]
        for _ in range(batch_size - 1):
            anchors.append(rng.randint(2, available))

        inputs, targets = [], []
        for end in anchors:
            window = self._padded_window(end, length + 1)
            inputs.append(window[:-1])
            targets.append(window[1:])
        
        return inputs, targets


    def _padded_window(self, end: int, length: int) -> List[int]:
        start = max(0, end - length)
        chunk = self.sequence[start:end]
        
        if len(chunk) < length:
            return [0] * (length - len(chunk)) + chunk
        
        return chunk


    def state_dict(self) -> dict:
        return {
            "sequence": list(self.sequence),
            "total_observed": self.total_observed,
            "recency": self._recency,
        }


    def load_state_dict(self, state: dict) -> None:
        self.sequence = list(state["sequence"])
        self.total_observed = state["total_observed"]
        self._recency = state["recency"].to(self._recency.device)