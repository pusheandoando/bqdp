# bqdp/core/history.py
import math





class History:
    def __init__(self, decay: float = 0.1):
        self.decay = decay
        self.sequence: list[int] = []

    def append(self, item_id: int):
        self.sequence.append(item_id)

    def temporal_weights(self, universe_items: list[int]) -> dict[int, float]:
        T = len(self.sequence)
        weights: dict[int, float] = {}

        for item in universe_items:
            weights[item] = 0.0
        
        for pos, item_id in enumerate(self.sequence):
            if item_id in weights:
                weights[item_id] += math.exp(-self.decay * (T - 1 - pos))
        return weights

    def to_dict(self) -> dict:
        return {"decay": self.decay, "sequence": self.sequence}

    @classmethod
    def from_dict(cls, data: dict) -> "History":
        h = cls(decay=data["decay"])
        h.sequence = data["sequence"]
        return h