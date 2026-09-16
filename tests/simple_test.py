# tests/simple_test.py
import random
from typing import Dict, List

from bqdp import BQDP





SUCCESSORS_PER_ITEM = 3
TRANSITION_PROBABILITY = 0.9





def build_successors(universe_size: int, rng: random.Random) -> Dict[int, List[int]]:
    return {
        item: rng.sample(range(1, universe_size + 1), SUCCESSORS_PER_ITEM)
        for item in range(1, universe_size + 1)
    }


def synthetic_stream(successors: Dict[int, List[int]], universe_size: int, length: int, rng: random.Random) -> List[int]:
    current = rng.randint(1, universe_size)
    stream = []
    
    for _ in range(length):
        stream.append(current)
        
        if rng.random() < TRANSITION_PROBABILITY:
            current = rng.choice(successors[current])
        else:
            current = rng.randint(1, universe_size)
    
    return stream


def oracle_recall(universe_size: int, k: int) -> float:
    background = (1.0 - TRANSITION_PROBABILITY) / universe_size
    informed = min(k, SUCCESSORS_PER_ITEM)
    remaining = max(0, k - SUCCESSORS_PER_ITEM)
    successor_probability = TRANSITION_PROBABILITY / SUCCESSORS_PER_ITEM + background
    
    return informed * successor_probability + remaining * background





def main() -> None:
    universe_size = 50
    candidates = 10
    rng = random.Random(7)
    successors = build_successors(universe_size, rng)
    stream = synthetic_stream(successors, universe_size, 2000, rng)

    model = BQDP(
        n = universe_size,
        evaluation_k = candidates,
        autosave_interval = 0,
        max_sequence_length = 32,
        batch_size = 16,
        embedding_dim = 32,
        feedforward_dim = 128,
    )

    for correct in stream:
        model.update(correct)

    report = model.evaluate()
    for name in sorted(report):
        print(f"{name}: {report[name]:.4f}")

    ceiling = oracle_recall(universe_size, candidates)
    print(f"oracle_recall_at_k: {ceiling:.4f}")
    print(f"gap_to_oracle: {ceiling - report['recent_recall_at_k']:.4f}")
    print(f"top {candidates} candidates: {model.predict(k=candidates)}")
    print(f"conformal set size: {len(model.predict_set())}")
    print(f"state saved in: {model.save()}")





if __name__ == "__main__":
    main()