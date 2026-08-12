from __future__ import annotations

import numpy as np

from .split import Split


def sample_balanced_support(
    labels: np.ndarray,
    assignments: np.ndarray,
    *,
    shots_per_class: int,
    seed: int,
) -> dict[int, list[int]]:
    if shots_per_class <= 0:
        raise ValueError("shots_per_class must be positive")
    rng = np.random.default_rng(seed)
    result: dict[int, list[int]] = {}
    for label in (0, 1):
        pool = np.flatnonzero((assignments == int(Split.ALIGNMENT)) & (labels == label))
        if pool.size < shots_per_class:
            raise ValueError(
                f"alignment split has {pool.size} examples for label {label}, "
                f"but {shots_per_class} are required"
            )
        result[label] = sorted(rng.choice(pool, size=shots_per_class, replace=False).tolist())
    return result

