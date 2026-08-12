from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import CausalRelationGraph, causal_dfs
from .split import Split, TemporalSplit


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, object]


def audit_causal_motif_coverage(
    graph: CausalRelationGraph,
    *,
    sample_edges: int = 20000,
    seed: int = 42,
) -> dict[str, object]:
    """Estimate motif-channel diversity using only historical predecessors.

    The audit never creates a new message-passing edge. It asks whether sampled
    base edges participate in a causal triangle, four-cycle, or four-clique.
    """
    if graph.num_edges == 0:
        return {
            "sampled_edges": 0,
            "coverage": {"edge": 0.0, "triangle": 0.0, "4-cycle": 0.0, "4-clique": 0.0},
            "recommended_mode": "edge_only",
            "reason": "empty relation graph",
        }
    rng = np.random.default_rng(seed)
    count = min(int(sample_edges), graph.num_edges)
    edge_offsets = rng.choice(graph.num_edges, size=count, replace=False)
    destinations = np.searchsorted(graph.indptr, edge_offsets, side="right") - 1
    cache: dict[int, set[int]] = {}

    def pred(node: int) -> set[int]:
        if node not in cache:
            cache[node] = set(map(int, graph.predecessors(node)))
        return cache[node]

    hits = {"triangle": 0, "4-cycle": 0, "4-clique": 0}
    for offset, dst in zip(edge_offsets, destinations):
        src = int(graph.indices[int(offset)])
        dst = int(dst)
        src_pred, dst_pred = pred(src), pred(dst)
        common = src_pred & dst_pred
        if common:
            hits["triangle"] += 1

        clique = False
        common_list = sorted(common)
        for right_pos, right in enumerate(common_list):
            right_pred = pred(right)
            if any(left in right_pred for left in common_list[:right_pos]):
                clique = True
                break
        if clique:
            hits["4-clique"] += 1

        cycle = False
        for left in src_pred:
            for right in dst_pred:
                if left == right:
                    continue
                newer, older = (left, right) if left > right else (right, left)
                if older in pred(newer):
                    cycle = True
                    break
            if cycle:
                break
        if cycle:
            hits["4-cycle"] += 1

    coverage = {"edge": 1.0, **{name: value / count for name, value in hits.items()}}
    degenerate = [
        name for name, value in coverage.items()
        if name != "edge" and (value < 0.01 or value > 0.95)
    ]
    return {
        "sampled_edges": count,
        "coverage": coverage,
        "jaccard_with_edge": {name: value for name, value in coverage.items() if name != "edge"},
        "recommended_mode": "edge_only" if degenerate else "four_motif",
        "degenerate_channels": degenerate,
    }


def audit_temporal_contract(
    timestamps: np.ndarray,
    split: TemporalSplit,
    graphs: tuple[CausalRelationGraph, ...],
    *,
    labels: np.ndarray | None = None,
    max_depth: int,
    dfs_sample_size: int = 5000,
    seed: int = 42,
) -> AuditReport:
    timestamps = np.asarray(timestamps, dtype=np.int64)
    assignments = split.assignments
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    unique_times, timestamp_counts = np.unique(timestamps, return_counts=True)
    positive_steps = np.diff(unique_times)
    details["temporal_data_quality"] = {
        "unique_timestamps": int(unique_times.size),
        "minimum_positive_step": int(positive_steps.min()) if positive_steps.size else None,
        "equal_timestamp_group_size": {
            "median": float(np.median(timestamp_counts)),
            "p95": float(np.quantile(timestamp_counts, 0.95)),
            "max": int(timestamp_counts.max()),
        },
        "policy": "equal timestamps are unordered and excluded from predecessor context",
    }

    boundaries_ok = True
    ranges: dict[str, tuple[int, int]] = {}
    previous_max = None
    for part in Split:
        values = timestamps[assignments == int(part)]
        if values.size == 0:
            boundaries_ok = False
            continue
        ranges[part.name.lower()] = (int(values.min()), int(values.max()))
        if previous_max is not None and int(values.min()) <= previous_max:
            boundaries_ok = False
        previous_max = int(values.max())
    checks["strict_split_time_order"] = boundaries_ok
    details["timestamp_ranges"] = ranges

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != timestamps.shape:
            raise ValueError("labels and timestamps must have identical shapes")
        class_stats = {}
        both_classes = True
        for part in Split:
            values = labels[assignments == int(part)]
            observed = np.unique(values)
            both_classes = both_classes and observed.size == 2
            class_stats[part.name.lower()] = {
                "rows": int(values.size),
                "fraud_rate": float(values.mean()),
                "classes": [int(value) for value in observed],
            }
        checks["both_classes_in_every_split"] = both_classes
        details["split_class_stats"] = class_stats

    graph_checks = {}
    for graph in graphs:
        causal = True
        max_degree = 0
        for dst in range(graph.num_nodes):
            pred = graph.predecessors(dst)
            max_degree = max(max_degree, int(pred.size))
            if pred.size and np.any(timestamps[pred] >= timestamps[dst]):
                causal = False
                break
        graph_checks[graph.relation] = causal and max_degree <= graph.max_history
        details[f"{graph.relation}_edges"] = graph.num_edges
        details[f"{graph.relation}_max_history_seen"] = max_degree
        degree = np.diff(graph.indptr)
        details[f"{graph.relation}_history_coverage"] = {
            part.name.lower(): {
                "nonempty_rate": float((degree[assignments == int(part)] > 0).mean()),
                "mean_history": float(degree[assignments == int(part)].mean()),
                "at_least_8_rate": float((degree[assignments == int(part)] >= 8).mean()),
            }
            for part in Split
        }
    checks["causal_relation_edges"] = all(graph_checks.values())
    details["relation_checks"] = graph_checks

    rng = np.random.default_rng(seed)
    sample_size = min(int(dfs_sample_size), timestamps.size)
    sample = rng.choice(timestamps.size, size=sample_size, replace=False)
    dfs_ok = True
    for graph in graphs:
        for center in sample:
            nodes = causal_dfs(int(center), graph, timestamps, max_depth=max_depth)
            if center in nodes or any(timestamps[node] >= timestamps[center] for node in nodes):
                dfs_ok = False
                break
        if not dfs_ok:
            break
    checks["causal_dfs"] = dfs_ok
    details["dfs_nodes_audited"] = sample_size * len(graphs)
    details["motif_audit"] = {
        graph.relation: audit_causal_motif_coverage(graph, seed=seed)
        for graph in graphs
    }
    return AuditReport(passed=all(checks.values()), checks=checks, details=details)
