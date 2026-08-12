from __future__ import annotations

import numpy as np

from experiments.grasp_dual_relation.audit import audit_temporal_contract
from experiments.grasp_dual_relation.graph import (
    build_causal_relation_graph,
    causal_dfs,
    merge_causal_relation_graphs,
)
from experiments.grasp_dual_relation.split import Split, stratified_time_sample, temporal_split


def test_split_never_divides_equal_timestamp():
    timestamps = np.repeat(np.arange(12, dtype=np.int64), [2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 2, 2])
    split = temporal_split(timestamps)
    for timestamp in np.unique(timestamps):
        assert np.unique(split.assignments[timestamps == timestamp]).size == 1
    assert set(split.assignments.tolist()) == {int(part) for part in Split}


def test_relation_graph_excludes_ties_and_future():
    timestamps = np.asarray([1, 2, 2, 3, 4], dtype=np.int64)
    entities = np.asarray(["u", "u", "u", "u", "u"])
    graph = build_causal_relation_graph(
        entities, timestamps, max_history=16, relation="user"
    )
    assert graph.predecessors(1).tolist() == [0]
    assert graph.predecessors(2).tolist() == [0]
    assert graph.predecessors(3).tolist() == [0, 1, 2]
    for node in range(len(timestamps)):
        assert all(timestamps[pred] < timestamps[node] for pred in graph.predecessors(node))


def test_dfs_moves_strictly_backwards_at_every_hop():
    timestamps = np.asarray([1, 2, 3, 4, 5], dtype=np.int64)
    entities = np.asarray(["u"] * 5)
    graph = build_causal_relation_graph(entities, timestamps, max_history=1, relation="user")
    assert causal_dfs(4, graph, timestamps, max_depth=3) == [3, 2, 1]


def test_appending_future_does_not_change_past_graph_or_dfs():
    past_times = np.asarray([1, 2, 3, 4], dtype=np.int64)
    full_times = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    past_entities = np.asarray(["u", "u", "v", "u"])
    full_entities = np.asarray(["u", "u", "v", "u", "u", "v"])
    past = build_causal_relation_graph(past_entities, past_times, max_history=2, relation="user")
    full = build_causal_relation_graph(full_entities, full_times, max_history=2, relation="user")
    for node in range(len(past_times)):
        assert past.predecessors(node).tolist() == full.predecessors(node).tolist()
        assert causal_dfs(node, past, past_times, max_depth=2) == causal_dfs(
            node, full, full_times, max_depth=2
        )


def test_merged_graph_is_causal_union():
    timestamps = np.arange(1, 7, dtype=np.int64)
    user = build_causal_relation_graph(
        np.asarray(["u", "u", "v", "u", "v", "u"]), timestamps,
        max_history=2, relation="user",
    )
    business = build_causal_relation_graph(
        np.asarray(["a", "b", "a", "a", "b", "a"]), timestamps,
        max_history=3, relation="business",
    )
    merged = merge_causal_relation_graphs(user, business, max_history=4)
    for node in range(len(timestamps)):
        expected = sorted(
            set(user.predecessors(node).tolist()) | set(business.predecessors(node).tolist())
        )[-4:]
        assert merged.predecessors(node).tolist() == expected
        assert all(timestamps[pred] < timestamps[node] for pred in expected)


def test_gate_zero_audit_passes_valid_graphs():
    timestamps = np.arange(1, 21, dtype=np.int64)
    split = temporal_split(timestamps)
    user = build_causal_relation_graph(
        np.asarray([f"u{i % 3}" for i in range(20)]), timestamps, max_history=2, relation="user"
    )
    business = build_causal_relation_graph(
        np.asarray([f"b{i % 4}" for i in range(20)]), timestamps, max_history=3, relation="business"
    )
    report = audit_temporal_contract(
        timestamps,
        split,
        (user, business),
        labels=np.asarray(([0, 1] * 10), dtype=np.int8),
        max_depth=2,
        dfs_sample_size=20,
    )
    assert report.passed, report
    assert report.checks["both_classes_in_every_split"]
    assert report.details["temporal_data_quality"]["unique_timestamps"] == 20
    assert "validation" in report.details["user_history_coverage"]


def test_stratified_time_sample_is_exact_and_deterministic():
    nodes = np.arange(100, dtype=np.int64)
    labels = np.asarray(([0] * 8 + [1] * 2) * 10, dtype=np.int8)
    first = stratified_time_sample(nodes, labels, sample_size=30, seed=42, time_bins=5)
    second = stratified_time_sample(nodes, labels, sample_size=30, seed=42, time_bins=5)
    assert first.tolist() == second.tolist()
    assert first.size == 30
    assert set(labels[first]) == {0, 1}
