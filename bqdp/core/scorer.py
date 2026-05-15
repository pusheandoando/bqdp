# bqdp/core/scorer.py





class Scorer:
    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        if round(alpha + beta, 10) != 1.0:
            raise ValueError("alpha + beta must equal 1.0")
        
        self.alpha = alpha
        self.beta = beta

    def rank(self, universe_items: list[int], temporal_weights: dict[int, float], cooccurrence_scores: dict[int, float], k: int) -> list[int]:
        max_t = max(temporal_weights.values(), default=1) or 1
        max_c = max(cooccurrence_scores.values(), default=1) or 1

        scored = []
        for item in universe_items:
            t = temporal_weights.get(item, 0.0) / max_t
            c = cooccurrence_scores.get(item, 0.0) / max_c
            scored.append((item, self.alpha * t + self.beta * c))

        result = []
        scored.sort(key=lambda x: x[1], reverse=True)
        for item, _ in scored[:k]:
            result.append(item)
        return result

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: dict) -> "Scorer":
        return cls(alpha=data["alpha"], beta=data["beta"])