from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np

from utils.constants import DEFAULT_GRAPH_PAD_ID


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def pad_nodes(nodes: Iterable[int], size: int, pad_id: int = DEFAULT_GRAPH_PAD_ID) -> list[int]:
    selected = [int(node) for node in nodes][:size]
    return selected + [int(pad_id)] * (size - len(selected))


def text_topk_select(center: int, candidates: Iterable[int], embeddings: np.ndarray, *, k: int) -> list[int]:
    candidates = np.asarray(list(dict.fromkeys(int(x) for x in candidates)), dtype=np.int64)
    if candidates.size == 0:
        return pad_nodes([], k)
    vectors = _normalized_rows(np.vstack([embeddings[center], embeddings[candidates]]))
    scores = vectors[1:] @ vectors[0]
    order = np.lexsort((candidates, -scores))
    return pad_nodes(candidates[order].tolist(), k)


def random_select(center: int, candidates: Iterable[int], *, k: int, seed: int) -> list[int]:
    unique = list(dict.fromkeys(int(x) for x in candidates))
    digest = hashlib.sha256(f"{seed}:{center}".encode("ascii")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    rng.shuffle(unique)
    return pad_nodes(unique, k)


def ocs_select(
    center: int,
    candidates: Iterable[int],
    structure_embeddings: np.ndarray,
    predecessor_graph,
    *,
    k: int,
    beta: float = 0.55,
) -> list[int]:
    """Deterministic one-hop OCS ranking over an already-causal candidate set."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    candidates = np.asarray(list(dict.fromkeys(int(x) for x in candidates)), dtype=np.int64)
    if candidates.size == 0:
        return pad_nodes([], k)

    normalized = _normalized_rows(np.asarray(structure_embeddings, dtype=np.float32))
    center_scores = np.maximum(0.0, normalized[candidates] @ normalized[center])
    candidate_set = set(candidates.tolist())
    coherence = np.zeros(candidates.size, dtype=np.float32)
    for offset, node in enumerate(candidates):
        local = [int(x) for x in predecessor_graph.predecessors(int(node)) if int(x) in candidate_set]
        if local:
            coherence[offset] = float(
                np.maximum(0.0, normalized[local] @ normalized[int(node)]).mean()
            )
    scores = (1.0 - beta) * center_scores + beta * coherence
    order = np.lexsort((candidates, -scores))
    return pad_nodes(candidates[order].tolist(), k)

