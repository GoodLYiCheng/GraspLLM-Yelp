from __future__ import annotations

import numpy as np

from experiments.grasp_dual_relation.graph import CausalRelationGraph
from experiments.grasp_dual_relation.sampling import ocs_select, random_select, text_topk_select
from utils.constants import DEFAULT_GRAPH_PAD_ID


def _graph(predecessors):
    flat = []
    indptr = [0]
    for row in predecessors:
        flat.extend(row)
        indptr.append(len(flat))
    return CausalRelationGraph(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(flat, dtype=np.int32),
        relation="user",
        max_history=8,
    )


def test_selectors_pad_and_are_deterministic():
    embeddings = np.eye(4, dtype=np.float32)
    assert text_topk_select(0, [], embeddings, k=2) == [DEFAULT_GRAPH_PAD_ID] * 2
    first = random_select(3, [0, 1, 2], k=4, seed=42)
    second = random_select(3, [0, 1, 2], k=4, seed=42)
    assert first == second
    assert first[-1] == DEFAULT_GRAPH_PAD_ID


def test_beta_changes_ocs_tradeoff():
    graph = _graph([[], [], [3], [], [1, 2, 3]])
    embeddings = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    center_first = ocs_select(4, [1, 2, 3], embeddings, graph, k=1, beta=0.0)
    coherence_first = ocs_select(4, [1, 2, 3], embeddings, graph, k=1, beta=1.0)
    assert center_first == [1]
    assert coherence_first == [2]

