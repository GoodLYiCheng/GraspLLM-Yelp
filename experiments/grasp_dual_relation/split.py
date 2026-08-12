from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Split(IntEnum):
    STRUCTURE = 0
    ALIGNMENT = 1
    VALIDATION = 2
    TEST = 3


@dataclass(frozen=True)
class TemporalSplit:
    assignments: np.ndarray
    boundaries: tuple[int, int, int]

    def mask(self, split: Split) -> np.ndarray:
        return self.assignments == int(split)


def temporal_split(
    timestamps: np.ndarray,
    ratios: tuple[float, float, float, float] = (0.60, 0.10, 0.15, 0.15),
) -> TemporalSplit:
    """Split sorted events without ever dividing an equal-timestamp group."""
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size < 4:
        raise ValueError("timestamps must be a one-dimensional array with at least four rows")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("timestamps must already be sorted")
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must be positive and sum to one")

    _, first_indices, counts = np.unique(timestamps, return_index=True, return_counts=True)
    if counts.size < 4:
        raise ValueError("at least four unique timestamps are required")
    cumulative = np.cumsum(counts)
    targets = np.cumsum(ratios[:-1]) * timestamps.size
    boundaries: list[int] = []
    min_group = 1
    for boundary_no, target in enumerate(targets):
        group = int(np.searchsorted(cumulative, target, side="left")) + 1
        remaining = len(counts) - (len(targets) - boundary_no)
        group = min(max(group, min_group), remaining)
        boundaries.append(int(first_indices[group]))
        min_group = group + 1

    b0, b1, b2 = boundaries
    assignments = np.empty(timestamps.size, dtype=np.uint8)
    assignments[:b0] = Split.STRUCTURE
    assignments[b0:b1] = Split.ALIGNMENT
    assignments[b1:b2] = Split.VALIDATION
    assignments[b2:] = Split.TEST
    return TemporalSplit(assignments=assignments, boundaries=(b0, b1, b2))


def stratified_time_sample(
    node_ids: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    seed: int,
    time_bins: int = 10,
) -> np.ndarray:
    """Deterministic label-by-time-bin sample from an already ordered split."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    labels = np.asarray(labels)
    if sample_size <= 0 or sample_size > node_ids.size:
        raise ValueError("sample_size must be in [1, number of candidate nodes]")
    rng = np.random.default_rng(seed)
    strata = []
    for time_bin, positions in enumerate(np.array_split(np.arange(node_ids.size), time_bins)):
        for label in np.unique(labels[node_ids[positions]]):
            members = node_ids[positions][labels[node_ids[positions]] == label]
            if members.size:
                strata.append((time_bin, int(label), members))
    exact = np.asarray([sample_size * members.size / node_ids.size for _, _, members in strata])
    quotas = np.floor(exact).astype(int)
    remainder = sample_size - int(quotas.sum())
    order = np.argsort(-(exact - quotas), kind="stable")
    for offset in order[:remainder]:
        quotas[offset] += 1
    selected = []
    for quota, (_, _, members) in zip(quotas, strata):
        if quota:
            selected.extend(rng.choice(members, size=quota, replace=False).tolist())
    result = np.asarray(sorted(selected), dtype=np.int64)
    if result.size != sample_size:
        raise RuntimeError(f"stratified sampler selected {result.size}, expected {sample_size}")
    return result
