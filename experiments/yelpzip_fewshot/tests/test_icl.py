from __future__ import annotations

import numpy as np
import pytest

from experiments.yelpzip_fewshot.icl import (
    build_icl_conversations, stack_graph_node_ids, support_entries_from_records,
    stratified_record_subset, truncate_query_review_text, validate_icl_query_records,
)


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return type("Batch", (), {"input_ids": [ord(char) for char in text]})()

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)


def _record(node_id: int, label: int) -> dict:
    return {
        "id": node_id,
        "conversations": [
            {"from": "human", "value": "contains <graph> but must not enter ICL support"},
            {"from": "gpt", "value": "Fraudulent" if label else "Legitimate"},
        ],
    }


def test_icl_support_is_validation_only_and_text_only():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    val_mask = np.asarray([True, True, False, False])
    support = support_entries_from_records([_record(0, 0), _record(1, 1)], labels, val_mask)
    conversations = build_icl_conversations(
        support=support, raw_texts=["short text", "fraud text", "x", "y"],
        query_user_text="Target <graph>", tokenizer=_Tokenizer(), support_max_tokens=5,
    )
    support_users = conversations[:-1:2]
    assert all("<graph>" not in turn["value"] for turn in support_users)
    assert conversations[-1]["value"] == "Target <graph>"
    assert [turn["value"] for turn in conversations[1:-1:2]] == ["Legitimate", "Fraudulent"]


def test_graph_aware_icl_has_one_graph_marker_per_support_then_query():
    support = [(0, 0), (1, 1)]
    conversations = build_icl_conversations(
        support=support, raw_texts=["normal", "fraud"],
        query_user_text="Target <graph>", tokenizer=_Tokenizer(), support_max_tokens=5,
        include_support_graphs=True,
    )
    user_turns = conversations[::2]
    assert [turn["value"].count("<graph>") for turn in user_turns] == [1, 1, 1]
    assert [turn["value"] for turn in conversations[1:-1:2]] == ["Legitimate", "Fraudulent"]


def test_graph_rows_preserve_support_then_query_order_and_pad():
    rows = stack_graph_node_ids(
        [{"graph": [10, 11]}, {"graph": [20]}, {"graph": [30, 31, 32]}], pad_id=-500,
    )
    assert rows.tolist() == [[10, 11, -500], [20, -500, -500], [30, 31, 32]]


def test_query_truncation_preserves_graph_and_classification_instruction():
    text = (
        "Given a review-centered graph: <graph>. The target review text is: abcdef "
        "Classify the target review as exactly one of: Fraudulent, Legitimate."
    )
    truncated = truncate_query_review_text(_Tokenizer(), text, 3)
    assert "<graph>" in truncated
    assert "The target review text is: abc" in truncated
    assert truncated.endswith("Classify the target review as exactly one of: Fraudulent, Legitimate.")


def test_icl_rejects_support_reused_as_validation_query():
    labels = np.asarray([0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="reused"):
        validate_icl_query_records(
            [_record(0, 0)], support_node_ids=[0], labels=labels,
            allowed_mask=np.asarray([True, True]), split_name="validation",
        )


def test_validation_subset_is_deterministic_stratified_and_keeps_input_order():
    records = [_record(node, int(node >= 8)) for node in range(10)]
    first = stratified_record_subset(records, 5, seed=42)
    second = stratified_record_subset(records, 5, seed=42)
    assert [record["id"] for record in first] == [record["id"] for record in second]
    assert [record["id"] for record in first] == sorted(record["id"] for record in first)
    assert sum(record["conversations"][1]["value"] == "Fraudulent" for record in first) == 1


def test_validation_subset_none_keeps_all_and_invalid_size_rejected():
    records = [_record(0, 0), _record(1, 1)]
    assert stratified_record_subset(records, None) == records
    with pytest.raises(ValueError, match="positive"):
        stratified_record_subset(records, 0)
