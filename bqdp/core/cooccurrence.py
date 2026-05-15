# bqdp/core/cooccurrence.py
from collections import defaultdict





class Cooccurrence:
    def __init__(self, window: int=5):
        self.window = window
        self.matrix: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def update(self, sequence: list[int], new_item: int):
        recent = sequence[-(self.window):]
        for other in recent:
            if other != new_item:
                self.matrix[new_item][other] += 1
                self.matrix[other][new_item] += 1

    def scores(self, universe_items: list[int], recent_history: list[int]) -> dict[int, float]:
        result: dict[int, float] = {}

        for item in universe_items:
            result[item] = 0.0
        
        for item in universe_items:
            for recent in recent_history:
                result[item] += self.matrix[item].get(recent, 0.0)
        return result

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "matrix": {k: dict(v) for k, v in self.matrix.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cooccurrence":
        c = cls(window=data["window"])
        
        for k, v in data["matrix"].items():
            temp = defaultdict(float)

            for kk, vv in v.items():
                temp[int(kk)] = vv
            
            c.matrix[int(k)] = temp
        return c