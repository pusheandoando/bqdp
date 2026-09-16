# bqdp/core/markov.py
from collections import OrderedDict
from typing import Dict, Sequence, Tuple

import torch





class MarkovChain:
    def __init__(self, table_size: int, order: int=4, discount: float=0.75, max_contexts: int=200000, max_successors: int=64):
        if table_size < 2:
            raise ValueError("table_size must be >= 2")
        
        if order < 1:
            raise ValueError("order must be >= 1")
        
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must be in [0.0, 1.0)")
        
        if max_contexts < 1:
            raise ValueError("max_contexts must be >= 1")
        
        if max_successors < 1:
            raise ValueError("max_successors must be >= 1")

        self.table_size = table_size
        self.order = order
        self.discount = discount
        self.max_contexts = max_contexts
        self.max_successors = max_successors
        self.contexts: "OrderedDict[Tuple[int, ...], Dict[int, float]]" = OrderedDict()
        self.unigram = torch.zeros(table_size, dtype=torch.float32)
        self.observed = 0


    def update(self, recent: Sequence[int], target: int) -> None:
        self.unigram[target] += 1.0
        self.observed += 1

        for length in range(1, self.order + 1):
            if len(recent) < length:
                break
            
            self._increment(tuple(recent[-length:]), target)


    def distribution(self, recent: Sequence[int]) -> torch.Tensor:
        prediction = self._base()

        for length in range(1, self.order + 1):
            if len(recent) < length:
                break
            
            row = self.contexts.get(tuple(recent[-length:]))
            
            # a missing context invalidates every longer one, so the backoff stops here
            if not row:
                break
            
            prediction = self._interpolate(row, prediction)

        return prediction


    def _base(self) -> torch.Tensor:
        prior = self.unigram + 1.0
        prior[0] = 0.0
        
        return prior / prior.sum()


    def _interpolate(self, row: Dict[int, float], backoff: torch.Tensor) -> torch.Tensor:
        total = sum(row.values())
        
        if total <= 0.0:
            return backoff

        # absolute discounting reserves exactly the mass it subtracts, so the result stays normalized
        reserved = self.discount * len(row) / total
        blended = backoff * reserved

        index = torch.tensor(list(row.keys()), dtype=torch.long)
        value = torch.tensor(
            [max(count - self.discount, 0.0) / total for count in row.values()],
            dtype=torch.float32,
        )
        blended.index_add_(0, index, value)

        return blended


    def _increment(self, context: Tuple[int, ...], target: int) -> None:
        row = self.contexts.get(context)

        if row is None:
            row = {}
            self.contexts[context] = row
            
            if len(self.contexts) > self.max_contexts:
                self.contexts.popitem(last=False)
        else:
            self.contexts.move_to_end(context)

        row[target] = row.get(target, 0.0) + 1.0

        if len(row) > 2 * self.max_successors:
            self.contexts[context] = self._prune(row)


    def _prune(self, row: Dict[int, float]) -> Dict[int, float]:
        ordered = sorted(row.items(), key=lambda entry: entry[1], reverse=True)
        
        return dict(ordered[: self.max_successors])


    def state_dict(self) -> dict:
        entries = []
        
        for context, row in self.contexts.items():
            entries.append([list(context), list(row.keys()), list(row.values())])
        
        return {"entries": entries, "unigram": self.unigram, "observed": self.observed}


    def load_state_dict(self, state: dict) -> None:
        self.contexts = OrderedDict()
        
        for context, keys, values in state["entries"]:
            self.contexts[tuple(context)] = dict(zip(keys, values))
        
        self.unigram = state["unigram"].to(self.unigram.device)
        self.observed = state["observed"]