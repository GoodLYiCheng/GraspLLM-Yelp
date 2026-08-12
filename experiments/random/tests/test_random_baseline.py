from __future__ import annotations

import numpy as np

from experiments.random.run import generate_random_scores


def test_uniform_scores_are_deterministic_and_label_blind():
    nodes = np.asarray([2, 5, 9, 12], dtype=np.int64)
    first = generate_random_scores(nodes, method="uniform", seed=42, alignment_fraud_rate=0.2)
    second = generate_random_scores(nodes, method="uniform", seed=42, alignment_fraud_rate=0.8)
    assert np.array_equal(first, second)
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_alignment_prior_is_constant():
    scores = generate_random_scores(
        np.arange(5), method="alignment_prior", seed=42, alignment_fraud_rate=0.125
    )
    assert scores.tolist() == [0.125] * 5
