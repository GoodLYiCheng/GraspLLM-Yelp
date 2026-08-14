from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import PROTOCOL_NAME
from .artifacts import sha256_lines, write_json


def sample_support(data, *, k: int, seed: int) -> list[int]:
    if k < 1:
        raise ValueError("k must be positive")
    pool = torch.where(data.support_mask)[0].cpu().numpy()
    labels = data.y[pool].cpu().numpy()
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        candidates = np.sort(pool[labels == label])
        if candidates.size < k:
            raise ValueError(f"support pool has {candidates.size} label={label} rows, needs {k}")
        selected.extend(map(int, rng.choice(candidates, size=k, replace=False)))
    # Interleave Normal/Fraud so every prefix does not privilege a class.
    return [value for pair in zip(selected[:k], selected[k:]) for value in pair]


def support_manifest(data, *, k: int, seed: int) -> dict:
    nodes = sample_support(data, k=k, seed=seed)
    records = [
        {
            "node_index": node,
            "node_key": data.node_keys[node],
            "year": int(data.years[node]),
            "label": int(data.y[node]),
        }
        for node in nodes
    ]
    counts = {str(label): sum(row["label"] == label for row in records) for label in (0, 1)}
    if counts != {"0": k, "1": k}:
        raise RuntimeError(f"unbalanced support sample: {counts}")
    return {
        "protocol": PROTOCOL_NAME,
        "definition": "K Normal + K Fraud",
        "k_per_class": k,
        "total": 2 * k,
        "seed": seed,
        "support_year": 2019,
        "counts": counts,
        "node_hash": sha256_lines(str(node) for node in nodes),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic balanced 2019 support manifests")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    args = parser.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for k in args.k:
        for seed in args.seeds:
            manifest = support_manifest(data, k=k, seed=seed)
            path = args.output_dir / f"support_k{k}_seed{seed}.json"
            write_json(path, manifest)
            index.append(str(path.resolve()))
    print(json.dumps({"status": "COMPLETED", "files": len(index), "paths": index}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
