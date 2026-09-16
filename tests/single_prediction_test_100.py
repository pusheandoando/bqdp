# tests/single_prediction_test_100.py
from typing import List

from bqdp import BQDP





def build_training_history(pattern: List[int], repetitions: int) -> List[int]:
    history = []

    for _ in range(repetitions):
        history.extend(pattern)

    return history


def rank_of(candidates_by_score: List[int], target: int) -> int:
    if target not in candidates_by_score:
        return -1

    return candidates_by_score.index(target) + 1





def main() -> None:
    universe_size = 100
    candidates = 40

    # the training history repeats this pattern so the target below is genuinely learnable
    context_before_target = [7, 63, 7, 63, 7, 63, 7, 63]
    pattern_repetitions = 320
    hidden_item = 63

    training_history = build_training_history(context_before_target, pattern_repetitions)

    model = BQDP(
        n = universe_size,
        evaluation_k = candidates,
        autosave_interval = 0,
        max_sequence_length = 32,
        batch_size = 16,
        embedding_dim = 32,
        feedforward_dim = 128,
        markov_weight = 0.55,
        model_weight = 0.15,
        temporal_weight = 0.15,
        cooccurrence_weight = 0.15,
        mixture_temperature = 2.0,
    )

    for item in training_history:
        model.update(item)

    top_candidates = model.predict(k=universe_size)
    conformal_set = model.predict_set()

    hit_in_top_k = hidden_item in top_candidates[:candidates]
    hit_in_conformal_set = hidden_item in conformal_set
    position = rank_of(top_candidates, hidden_item)
    
    output_sorted = sorted(top_candidates[:candidates])

    print(f"hidden item (defined by hand): {hidden_item}")
    print(f"predicted top {candidates}: {output_sorted}")
    print(f"hit in top {candidates}: {hit_in_top_k}")
    print(f"rank of hidden item: {position} out of {universe_size}")
    print(f"conformal set size: {len(conformal_set)}")
    print(f"hit in conformal set: {hit_in_conformal_set}")
    print(f"expert weights: {model.scorer.weights}")

    model.update(hidden_item)
    print(f"state saved in: {model.save()}")





if __name__ == "__main__":
    main()