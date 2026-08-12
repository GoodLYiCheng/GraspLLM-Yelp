from __future__ import annotations

import pytest
import numpy as np

from experiments.yelpzip_fewshot.prepare import build_fewshot_splits


def _record(node_id: int, label: int) -> dict:
    return {
        "id": node_id,
        "graph": [node_id],
        "conversations": [
            {"from": "human", "value": "review <graph>"},
            {"from": "gpt", "value": "Fraudulent" if label else "Legitimate"},
        ],
    }


def test_support_is_balanced_deterministic_and_removed_from_validation():
    labels = np.asarray([0] * 12 + [1] * 12, dtype=np.int8)
    val_mask = np.ones(labels.size, dtype=bool)
    records = [_record(node, int(label)) for node, label in enumerate(labels)]
    support, validation, support_ids = build_fewshot_splits(
        records, labels, val_mask, shots=5, seed=42
    )
    again, _, again_ids = build_fewshot_splits(records, labels, val_mask, shots=5, seed=42)
    support_nodes = {row["id"] for row in support}
    assert len(support) == 10
    assert len(validation) == 14
    assert len(support_ids["Legitimate"]) == len(support_ids["Fraudulent"]) == 5
    assert support_nodes.isdisjoint({row["id"] for row in validation})
    assert [row["id"] for row in support] == [row["id"] for row in again]
    assert support_ids == again_ids


def test_support_rejects_records_outside_validation_mask():
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    val_mask = np.asarray([True, True, True, False])
    records = [_record(node, int(label)) for node, label in enumerate(labels)]
    with pytest.raises(ValueError, match="validation mask"):
        build_fewshot_splits(records, labels, val_mask, shots=1, seed=42)
