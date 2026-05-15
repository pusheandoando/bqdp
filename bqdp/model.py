# bqdp/model.py
import os
import json

from .router import resolve_state_dir
from .core import Universe, History, Cooccurrence, Scorer





class BQDP:
    def __init__(self, n: int, decay: float=0.1, alpha: float=0.6, beta: float=0.4, window: int=5, state_dir: str=None):
        if state_dir is not None:
            self.state_dir = resolve_state_dir(state_dir)
        else:
            self.state_dir = resolve_state_dir()
        self.universe = Universe(n)
        self.history = History(decay=decay)
        self.cooccurrence = Cooccurrence(window=window)
        self.scorer = Scorer(alpha=alpha, beta=beta)

    def predict(self, k: int) -> list[int]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > self.universe.n:
            raise ValueError("k cannot exceed universe size")

        recent = self.history.sequence[-(self.cooccurrence.window):]
        temporal = self.history.temporal_weights(self.universe.items)
        cooc = self.cooccurrence.scores(self.universe.items, recent)
        return self.scorer.rank(self.universe.items, temporal, cooc, k)

    def update(self, correct: int):
        if not self.universe.validate(correct):
            raise ValueError(f"{correct} is outside universe range [1, {self.universe.n}]")

        self.cooccurrence.update(self.history.sequence, correct)
        self.history.append(correct)
        self.save()

    def save(self):
        os.makedirs(self.state_dir, exist_ok=True)

        with open(os.path.join(self.state_dir, "history.json"), "w") as f:
            json.dump(self.history.to_dict(), f, indent=4)
        with open(os.path.join(self.state_dir, "cooccurrence.json"), "w") as f:
            json.dump(self.cooccurrence.to_dict(), f, indent=4)
        with open(os.path.join(self.state_dir, "universe.json"), "w") as f:
            json.dump(self.universe.to_dict(), f, indent=4)
        with open(os.path.join(self.state_dir, "scorer.json"), "w") as f:
            json.dump(self.scorer.to_dict(), f, indent=4)

    @classmethod
    def load(cls, state_dir: str=None) -> "BQDP":
        resolved = resolve_state_dir(state_dir) if state_dir is not None else resolve_state_dir()

        def read(name):
            with open(os.path.join(resolved, name)) as f:
                return json.load(f)

        universe = Universe.from_dict(read("universe.json"))
        history = History.from_dict(read("history.json"))
        cooccurrence = Cooccurrence.from_dict(read("cooccurrence.json"))
        scorer = Scorer.from_dict(read("scorer.json"))

        instance = cls.__new__(cls)
        instance.state_dir = resolved
        instance.universe = universe
        instance.history = history
        instance.cooccurrence = cooccurrence
        instance.scorer = scorer
        return instance