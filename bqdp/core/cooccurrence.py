# bqdp/core/cooccurrence.py
import math
from typing import Dict, List, Sequence

import torch





_RESCALE_THRESHOLD = 1e-12





class Cooccurrence:
    def __init__(self, table_size: int, window: int=5, decay: float=0.0, max_neighbors: int=64):
        if table_size < 2:
            raise ValueError("table_size must be >= 2")
        
        if window < 1:
            raise ValueError("window must be >= 1")
        
        if decay < 0.0:
            raise ValueError("decay must be >= 0.0")
        
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be >= 1")

        self.table_size = table_size
        self.window = window
        self.decay = decay
        self.max_neighbors = max_neighbors
        self.matrix: Dict[int, Dict[int, float]] = {}
        self._scale = 1.0
        self._decay_factor = math.exp(-decay)


    def update(self, recent: Sequence[int], new_item: int) -> None:
        # decay is folded into a global scale so stored rows are never swept
        self._scale *= self._decay_factor
        if self._scale < _RESCALE_THRESHOLD:
            self._rescale()

        increment = 1.0 / self._scale
        
        for other in recent[-self.window :]:
            if other <= 0:
                continue
            
            # only the forward edge is evidence of succession, the reverse direction is noise
            self._increment(other, new_item, increment)


    def scores(self, recent: Sequence[int]) -> torch.Tensor:
        scores = torch.zeros(self.table_size, dtype=torch.float32)
        
        for item_id in recent[-self.window :]:
            row = self.matrix.get(item_id)
            
            if not row:
                continue
            
            index = torch.tensor(list(row.keys()), dtype=torch.long)
            value = torch.tensor(list(row.values()), dtype=torch.float32)
            scores.index_add_(0, index, value)

        if self._scale != 1.0:
            scores.mul_(self._scale)
        
        return scores


    def _increment(self, row_id: int, column_id: int, increment: float) -> None:
        row = self.matrix.setdefault(row_id, {})
        row[column_id] = row.get(column_id, 0.0) + increment
        
        if len(row) > 2 * self.max_neighbors:
            self.matrix[row_id] = self._prune(row)


    def _prune(self, row: Dict[int, float]) -> Dict[int, float]:
        ordered = sorted(row.items(), key=lambda entry: entry[1], reverse=True)
        
        return dict(ordered[: self.max_neighbors])


    def _rescale(self) -> None:
        for row_id, row in self.matrix.items():
            self.matrix[row_id] = {
                column_id: value * self._scale for column_id, value in row.items()
            }
        
        self._scale = 1.0


    def state_dict(self) -> dict:
        return {"matrix": self.matrix, "scale": self._scale}


    def load_state_dict(self, state: dict) -> None:
        self.matrix = {
            int(row_id): {int(column_id): float(value) for column_id, value in row.items()}
            for row_id, row in state["matrix"].items()
        }
        
        self._scale = state["scale"]