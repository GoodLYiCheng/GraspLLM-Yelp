from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_support_records(alignment_records: list[dict], support_nodes: dict) -> list[dict]:
    wanted = {int(node) for nodes in support_nodes.values() for node in nodes}
    by_node = {int(record["node_id"]): record for record in alignment_records}
    missing = sorted(wanted - by_node.keys())
    if missing:
        raise ValueError(f"support nodes are absent from alignment contexts: {missing[:10]}")
    return [by_node[node] for node in sorted(wanted)]


def make_fewshot_query(support_records: list[dict], query_record: dict) -> dict:
    conversations = []
    graphs = []
    for record in support_records:
        if len(record["conversations"]) != 2:
            raise ValueError("support records must contain one user and one labelled assistant turn")
        conversations.extend(record["conversations"])
        graphs.extend(record["graphs"])
    if len(query_record["conversations"]) != 1:
        raise ValueError("evaluation query must contain only its user turn")
    conversations.extend(query_record["conversations"])
    graphs.extend(query_record["graphs"])
    return {
        "id": query_record["id"],
        "node_id": query_record["node_id"],
        "timestamp": query_record["timestamp"],
        "split": query_record["split"],
        "ground_truth": query_record["ground_truth"],
        "dataset": "yelpzip_grasp",
        "graphs": graphs,
        "conversations": conversations,
        "support_node_ids": [record["node_id"] for record in support_records],
    }

