from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import file_sha256


_MATCH_KEYS = (
    "encoder",
    "tokenizer_hash",
    "processed_data_sha256",
    "requested_max_length",
    "final_max_length",
    "dtype",
    "attention_backend",
    "num_nodes",
    "num_shards",
)


def merge_shards(paths: list[Path], output: Path) -> dict:
    if not paths:
        raise ValueError("at least one embedding shard is required")
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    reference = loaded[0]["metadata"]
    num_nodes = int(reference["num_nodes"])
    num_shards = int(reference["num_shards"])
    if len(paths) != num_shards:
        raise ValueError(f"expected {num_shards} shard files, got {len(paths)}")
    dimensions = {int(value["emb"].shape[1]) for value in loaded}
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions differ across shards: {dimensions}")
    merged = torch.zeros((num_nodes, dimensions.pop()), dtype=loaded[0]["emb"].dtype)
    seen = torch.zeros(num_nodes, dtype=torch.bool)
    records = []
    shard_ids = set()
    for path, value in zip(paths, loaded):
        metadata = value["metadata"]
        for key in _MATCH_KEYS:
            if metadata.get(key) != reference.get(key):
                raise ValueError(
                    f"shard metadata mismatch for {key}: "
                    f"{path}={metadata.get(key)!r}, expected={reference.get(key)!r}"
                )
        shard_id = int(metadata["shard_id"])
        if shard_id in shard_ids:
            raise ValueError(f"duplicate shard_id={shard_id}")
        shard_ids.add(shard_id)
        indices = torch.as_tensor(value["node_indices"], dtype=torch.long)
        if indices.numel() != value["emb"].shape[0]:
            raise ValueError(f"row/index mismatch in {path}")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= num_nodes):
            raise ValueError(f"out-of-range node index in {path}")
        if bool(seen[indices].any()):
            raise ValueError(f"overlapping node indices in {path}")
        merged[indices] = value["emb"]
        seen[indices] = True
        records.extend(value.get("token_records", []))
    if shard_ids != set(range(num_shards)):
        raise ValueError(f"shard IDs are incomplete: {sorted(shard_ids)}")
    missing = torch.where(~seen)[0]
    if missing.numel():
        raise ValueError(f"embedding shards omit {missing.numel()} nodes")
    records.sort(key=lambda row: int(row["node_id"]))
    if [int(row["node_id"]) for row in records] != list(range(num_nodes)):
        raise ValueError("token records do not cover nodes exactly once in node order")
    metadata = dict(reference)
    metadata.update({
        "merged_from_shards": num_shards,
        "shard_ids": sorted(shard_ids),
        "shard_paths": [str(path.resolve()) for path in paths],
        "shard_sha256": {str(path.resolve()): file_sha256(path) for path in paths},
        "shard_rows": num_nodes,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"emb": merged, "metadata": metadata, "token_records": records}, output)
    return {
        "status": "COMPLETED",
        "output": str(output.resolve()),
        "nodes": num_nodes,
        "dimension": int(merged.shape[1]),
        "shards": num_shards,
        "context_length": int(metadata["final_max_length"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge deterministic FiGraph embedding shards")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_shards(args.inputs, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
