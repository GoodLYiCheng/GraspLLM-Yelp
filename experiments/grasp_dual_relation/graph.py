from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CausalRelationGraph:
    """CSR predecessor graph; every stored edge is history -> current."""

    indptr: np.ndarray
    indices: np.ndarray
    relation: str
    max_history: int

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def num_edges(self) -> int:
        return int(self.indices.size)

    def predecessors(self, node_id: int) -> np.ndarray:
        start, end = int(self.indptr[node_id]), int(self.indptr[node_id + 1])
        return self.indices[start:end]

    def edge_index(self):
        import torch

        dst = np.repeat(np.arange(self.num_nodes, dtype=np.int64), np.diff(self.indptr))
        src = self.indices.astype(np.int64, copy=False)
        return torch.from_numpy(np.stack([src, dst], axis=0))


def build_causal_relation_graph(
    entity_ids: Iterable[str],
    timestamps: np.ndarray,
    *,
    max_history: int,
    relation: str,
) -> CausalRelationGraph:
    """Build a bounded causal graph while excluding equal timestamps."""
    entity_ids = np.asarray(list(entity_ids), dtype=object)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if entity_ids.shape[0] != timestamps.shape[0]:
        raise ValueError("entity_ids and timestamps must have equal length")
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("events must be sorted by timestamp")

    history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=max_history))
    flat: list[int] = []
    indptr = np.zeros(timestamps.size + 1, dtype=np.int64)
    start = 0
    while start < timestamps.size:
        end = start + 1
        while end < timestamps.size and timestamps[end] == timestamps[start]:
            end += 1

        # Query histories before publishing this timestamp, preventing ties.
        for node in range(start, end):
            flat.extend(history[str(entity_ids[node])])
            indptr[node + 1] = len(flat)
        for node in range(start, end):
            history[str(entity_ids[node])].append(node)
        start = end

    return CausalRelationGraph(
        indptr=indptr,
        indices=np.asarray(flat, dtype=np.int32),
        relation=relation,
        max_history=max_history,
    )


def causal_dfs(
    center: int,
    graph: CausalRelationGraph,
    timestamps: np.ndarray,
    *,
    max_depth: int = 1,
) -> list[int]:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    timestamps = np.asarray(timestamps, dtype=np.int64)
    query_time = int(timestamps[center])
    visited = {int(center)}
    result: list[int] = []
    stack: list[tuple[int, int]] = [(int(center), 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        current_time = int(timestamps[current])
        predecessors = graph.predecessors(current)
        # Reverse keeps the most recent predecessors first with a LIFO stack.
        for predecessor in predecessors[::-1]:
            node = int(predecessor)
            node_time = int(timestamps[node])
            if node in visited or node_time >= current_time or node_time >= query_time:
                continue
            visited.add(node)
            result.append(node)
            stack.append((node, depth + 1))
    return result


def merge_causal_relation_graphs(
    first: CausalRelationGraph,
    second: CausalRelationGraph,
    *,
    max_history: int,
) -> CausalRelationGraph:
    """Union two predecessor graphs without changing their causal direction."""
    if first.num_nodes != second.num_nodes:
        raise ValueError("relation graphs must have the same nodes")
    flat: list[int] = []
    indptr = np.zeros(first.num_nodes + 1, dtype=np.int64)
    for node in range(first.num_nodes):
        predecessors = sorted(
            set(map(int, first.predecessors(node))) | set(map(int, second.predecessors(node)))
        )[-max_history:]
        flat.extend(predecessors)
        indptr[node + 1] = len(flat)
    return CausalRelationGraph(
        indptr=indptr,
        indices=np.asarray(flat, dtype=np.int32),
        relation="merged",
        max_history=max_history,
    )
