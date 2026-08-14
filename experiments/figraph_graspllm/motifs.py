from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import scipy.sparse as sp
import torch


MOTIF_NAMES = ("edge", "triangle", "4-cycle", "4-clique")


@dataclass(frozen=True)
class MotifResult:
    channels: dict[str, torch.Tensor]
    audit: dict[str, object]


def _simple_undirected_edges(edges: np.ndarray, num_nodes: int) -> list[tuple[int, int]]:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return []
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [E,2]")
    result: set[tuple[int, int]] = set()
    for raw_u, raw_v in edges:
        u, v = int(raw_u), int(raw_v)
        if not 0 <= u < num_nodes or not 0 <= v < num_nodes:
            raise ValueError(f"edge ({u},{v}) is outside [0,{num_nodes})")
        if u != v:
            result.add((u, v) if u < v else (v, u))
    return sorted(result)


def _adjacency(pairs: list[tuple[int, int]], num_nodes: int) -> sp.csr_matrix:
    if not pairs:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.int32)
    undirected = np.asarray(pairs, dtype=np.int64)
    row = np.concatenate([undirected[:, 0], undirected[:, 1]])
    col = np.concatenate([undirected[:, 1], undirected[:, 0]])
    return sp.csr_matrix(
        (np.ones(row.size, dtype=np.int32), (row, col)), shape=(num_nodes, num_nodes)
    )


def _to_edge_index(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo(copy=False)
    if coo.nnz == 0:
        return torch.empty((2, 0), dtype=torch.long)
    order = np.lexsort((coo.col, coo.row))
    values = np.stack([coo.row[order], coo.col[order]], axis=0).astype(np.int64, copy=False)
    return torch.from_numpy(values).long()


def _subset_edge_index(pairs: list[tuple[int, int]], keep: set[tuple[int, int]]) -> torch.Tensor:
    selected = [pair for pair in pairs if pair in keep]
    if not selected:
        return torch.empty((2, 0), dtype=torch.long)
    values = np.asarray(selected, dtype=np.int64)
    directed = np.concatenate([values, values[:, ::-1]], axis=0)
    order = np.lexsort((directed[:, 1], directed[:, 0]))
    return torch.from_numpy(directed[order].T.copy()).long()


def _exact_membership(
    pairs: list[tuple[int, int]], adjacency: sp.csr_matrix
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    neighbors = [set(adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]]) for i in range(adjacency.shape[0])]
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.int64)
    a3 = (adjacency @ adjacency @ adjacency).tocsr()

    triangle_edges: set[tuple[int, int]] = set()
    cycle_edges: set[tuple[int, int]] = set()
    clique_edges: set[tuple[int, int]] = set()
    triangle_instances = 0
    four_clique_instances: set[tuple[int, int, int, int]] = set()
    for u, v in pairs:
        common = neighbors[u] & neighbors[v]
        if common:
            triangle_edges.add((u, v))
            triangle_instances += len(common)
        # A^3[u,v] includes deg(u)+deg(v)-1 backtracking walks for every edge.
        if int(a3[u, v]) - int(degree[u]) - int(degree[v]) + 1 > 0:
            cycle_edges.add((u, v))
        common_sorted = sorted(common)
        found_clique = False
        for a, b in combinations(common_sorted, 2):
            if b in neighbors[a]:
                found_clique = True
                four_clique_instances.add(tuple(sorted((u, v, a, b))))
        if found_clique:
            clique_edges.add((u, v))

    # Number of simple undirected 4-cycles: each opposite-node pair is counted twice.
    common_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for center, local in enumerate(neighbors):
        del center
        for pair in combinations(sorted(local), 2):
            common_pair_counts[pair] += 1
    four_cycle_instances = sum(count * (count - 1) // 2 for count in common_pair_counts.values()) // 2

    channels = {
        "edge": _to_edge_index(adjacency),
        "triangle": _subset_edge_index(pairs, triangle_edges),
        "4-cycle": _subset_edge_index(pairs, cycle_edges),
        "4-clique": _subset_edge_index(pairs, clique_edges),
    }
    counts = {
        "triangle_instances": int(triangle_instances // 3),
        "four_cycle_instances": int(four_cycle_instances),
        "four_clique_instances": int(len(four_clique_instances)),
    }
    return channels, counts


def _legacy_channels(adjacency: sp.csr_matrix) -> dict[str, torch.Tensor]:
    # This intentionally mirrors gnn/get_matrix.py::compute_motifs_torch.
    a2 = (adjacency @ adjacency).tocsr()
    a3 = (a2 @ adjacency).tocsr()
    a2.data[:] = 1
    a3.data[:] = 1
    edge = adjacency.copy()
    edge.data[:] = 1
    triangle = edge.multiply(a2)
    cycle = edge.multiply(a3)
    clique = a2.multiply(a3)
    for matrix in (triangle, cycle, clique):
        matrix.eliminate_zeros()
    return {
        "edge": _to_edge_index(edge),
        "triangle": _to_edge_index(triangle),
        "4-cycle": _to_edge_index(cycle),
        "4-clique": _to_edge_index(clique),
    }


def compute_motifs(
    edges: np.ndarray,
    num_nodes: int,
    *,
    mode: str = "grasp_legacy",
) -> MotifResult:
    if mode not in {"grasp_legacy", "exact_edge_membership"}:
        raise ValueError("mode must be grasp_legacy or exact_edge_membership")
    pairs = _simple_undirected_edges(edges, num_nodes)
    adjacency = _adjacency(pairs, num_nodes)
    exact_channels, instance_counts = _exact_membership(pairs, adjacency)
    channels = _legacy_channels(adjacency) if mode == "grasp_legacy" else exact_channels
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)

    def coverage(edge_index: torch.Tensor) -> float:
        if num_nodes == 0 or edge_index.numel() == 0:
            return 0.0
        return float(torch.unique(edge_index).numel() / num_nodes)

    audit = {
        "mode": mode,
        "nodes": int(num_nodes),
        "undirected_edges": int(len(pairs)),
        "isolated_nodes": int((degree == 0).sum()),
        "max_degree": int(degree.max()) if degree.size else 0,
        "selected_channels": {
            name: {
                "directed_pairs": int(channels[name].shape[1]),
                "node_coverage": coverage(channels[name]),
            }
            for name in MOTIF_NAMES
        },
        "exact_edge_membership": {
            name: {
                "directed_pairs": int(exact_channels[name].shape[1]),
                "node_coverage": coverage(exact_channels[name]),
            }
            for name in MOTIF_NAMES
        },
        **instance_counts,
    }
    return MotifResult(channels=channels, audit=audit)


def offset_channels(channels: dict[str, torch.Tensor], offset: int) -> dict[str, torch.Tensor]:
    return {
        name: value + int(offset) if value.numel() else value.clone()
        for name, value in channels.items()
    }
