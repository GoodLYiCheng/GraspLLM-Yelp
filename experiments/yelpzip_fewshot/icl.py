from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch


LABEL_TEXT = {0: "Legitimate", 1: "Fraudulent"}
YELP_REVIEW_PREFIX = "The target review text is: "
YELP_REVIEW_SUFFIX = " Classify the target review as exactly one of: Fraudulent, Legitimate."


def stratified_record_subset(records: list[dict], max_records: int | None, *, seed: int = 42) -> list[dict]:
    """Select a deterministic subset while preserving the observed label ratio."""
    if max_records is None or max_records >= len(records):
        return list(records)
    if max_records <= 0:
        raise ValueError("max_records must be positive")

    by_label: dict[str, list[tuple[int, dict]]] = {}
    for position, record in enumerate(records):
        conversations = record.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError("validation records must contain a prompt and label")
        label = str(conversations[1].get("value", "")).strip()
        if label not in LABEL_TEXT.values():
            raise ValueError(f"unexpected Yelp label text: {label!r}")
        by_label.setdefault(label, []).append((position, record))

    total = len(records)
    exact = {label: max_records * len(group) / total for label, group in by_label.items()}
    quotas = {label: int(value) for label, value in exact.items()}
    remaining = max_records - sum(quotas.values())
    for label in sorted(by_label, key=lambda key: (-(exact[key] - quotas[key]), key))[:remaining]:
        quotas[label] += 1

    selected: list[tuple[int, dict]] = []
    for label, group in by_label.items():
        ranked = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{seed}:{label}:{item[1].get('id', item[0])}".encode("utf-8")
            ).digest(),
        )
        selected.extend(ranked[:quotas[label]])
    return [record for _, record in sorted(selected, key=lambda item: item[0])]


def stack_graph_node_ids(records: Iterable[dict], *, pad_id: int) -> torch.Tensor:
    """Stack variable-length graph node lists without changing record order."""
    rows = [torch.as_tensor(record["graph"], dtype=torch.long).flatten() for record in records]
    if not rows:
        raise ValueError("at least one graph record is required")
    if any(row.numel() == 0 for row in rows):
        raise ValueError("graph records must contain at least one node or padding token")
    width = max(int(row.numel()) for row in rows)
    output = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
    for index, row in enumerate(rows):
        output[index, :row.numel()] = row
    return output


def support_entries_from_records(
    records: Iterable[dict], labels: np.ndarray, val_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Validate ICL support provenance and return stable ``(node_id, label)`` pairs."""
    entries: list[tuple[int, int]] = []
    seen: set[int] = set()
    for record in records:
        node_id = int(record["id"])
        if node_id in seen:
            raise ValueError(f"duplicate ICL support node id: {node_id}")
        if node_id < 0 or node_id >= labels.size or not bool(val_mask[node_id]):
            raise ValueError(f"ICL support node {node_id} is not in the validation mask")
        turns = record.get("conversations", [])
        if len(turns) != 2:
            raise ValueError("ICL support records require one user turn and one labelled assistant turn")
        label_text = str(turns[1].get("value", "")).strip()
        if label_text not in LABEL_TEXT.values():
            raise ValueError(f"unexpected ICL support label: {label_text!r}")
        label = int(label_text == LABEL_TEXT[1])
        if label != int(labels[node_id]):
            raise ValueError(f"ICL support node {node_id} label disagrees with processed_data.pt")
        entries.append((node_id, label))
        seen.add(node_id)
    if not entries:
        raise ValueError("ICL support set is empty")
    return entries


def validate_icl_query_records(
    records: Iterable[dict], *, support_node_ids: Iterable[int], labels: np.ndarray,
    allowed_mask: np.ndarray, split_name: str,
) -> None:
    """Reject support reuse and wrong-split records before ICL scoring."""
    support = {int(node) for node in support_node_ids}
    seen: set[int] = set()
    for record in records:
        node_id = int(record["id"])
        if node_id in seen:
            raise ValueError(f"duplicate {split_name} query id: {node_id}")
        if node_id in support:
            raise ValueError(f"ICL support node {node_id} was reused as a {split_name} query")
        if node_id < 0 or node_id >= labels.size or not bool(allowed_mask[node_id]):
            raise ValueError(f"ICL query node {node_id} is not in the declared {split_name} mask")
        seen.add(node_id)


def truncate_tokens(tokenizer, text: str, max_tokens: int | None) -> str:
    """Return the complete text unless an explicit positive token cap is supplied."""
    if max_tokens is None:
        return str(text)
    if max_tokens <= 0:
        raise ValueError("ICL support max_tokens must be positive")
    ids = tokenizer(str(text), add_special_tokens=False).input_ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_query_review_text(tokenizer, user_text: str, max_tokens: int | None) -> str:
    """Preserve the full Yelp review unless an explicit positive cap is supplied."""
    start = user_text.find(YELP_REVIEW_PREFIX)
    if start < 0:
        raise ValueError("Yelp query is missing the target-review prefix")
    body_start = start + len(YELP_REVIEW_PREFIX)
    end = user_text.find(YELP_REVIEW_SUFFIX, body_start)
    if end < 0:
        raise ValueError("Yelp query is missing the classification suffix")
    if max_tokens is None:
        return user_text
    if max_tokens <= 0:
        raise ValueError("query review max_tokens must be positive")
    review = user_text[body_start:end]
    truncated = truncate_tokens(tokenizer, review, max_tokens)
    return user_text[:body_start] + truncated + user_text[end:]


def build_icl_conversations(
    *,
    support: Iterable[tuple[int, int]],
    raw_texts: Iterable[str],
    query_user_text: str,
    tokenizer,
    support_max_tokens: int | None,
    include_support_graphs: bool = False,
) -> list[dict[str, str]]:
    """Build demonstrations followed by the original graph query.

    When ``include_support_graphs`` is enabled, each support user turn contains
    one ``<graph>`` marker.  The evaluator must pass graph tensors in the same
    support order, followed by the query graph.
    """
    if not isinstance(raw_texts, list):
        raw_texts = list(raw_texts)
    conversations: list[dict[str, str]] = []
    for node_id, label in support:
        review = truncate_tokens(tokenizer, raw_texts[int(node_id)], support_max_tokens)
        conversations.extend((
            {
                "from": "human",
                "value": (
                    "Example Yelp review"
                    + (" with its review-centered graph <graph>" if include_support_graphs else "")
                    + ":\n"
                    f"{review}\n\n"
                    "Classify this review as exactly one of: Fraudulent, Legitimate."
                ),
            },
            {"from": "gpt", "value": LABEL_TEXT[int(label)]},
        ))
    conversations.append({"from": "human", "value": query_user_text})
    return conversations
