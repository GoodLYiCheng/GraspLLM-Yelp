from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.paths import STAGE1_SOURCE_DATASETS


PAIR_NAMES = (
    ("yelpzip_rur", "yelpzip_rbr"),
    ("amazon_cellphones_rur", "amazon_cellphones_rpr"),
    ("amazon_clothing_rur", "amazon_clothing_rpr"),
    ("amazon_electronics_rur", "amazon_electronics_rpr"),
    ("amazon_home_rur", "amazon_home_rpr"),
    ("amazon_sports_rur", "amazon_sports_rpr"),
    ("amazon_toys_rur", "amazon_toys_rpr"),
)


def _load(dataset_root: Path, name: str):
    path = dataset_root / name / "processed_data.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_prepared(dataset_root: Path) -> dict:
    rows = {}
    loaded = {}
    for name in STAGE1_SOURCE_DATASETS:
        data = _load(dataset_root, name)
        loaded[name] = data
        metadata = getattr(data, "metadata", {}) or {}
        masks = [data.train_mask.bool(), data.val_mask.bool(), data.test_mask.bool()]
        if torch.any(masks[0] & masks[1]) or torch.any(masks[0] & masks[2]) or torch.any(masks[1] & masks[2]):
            raise ValueError(f"{name} masks overlap")
        if not torch.equal(data.pretrain_mask.bool(), data.train_mask.bool()):
            raise ValueError(f"{name} pretrain_mask differs from train_mask")
        degree = torch.bincount(data.edge_index[0], minlength=int(data.num_nodes))
        if degree.numel() and int(degree.max()) > int(metadata.get("max_neighbors", 32)):
            raise ValueError(f"{name} exceeds max degree")
        rows[name] = {
            "nodes": int(data.num_nodes),
            "edges": int(data.edge_index.size(1)),
            "train": int(data.train_mask.sum()),
            "validation": int(data.val_mask.sum()),
            "test": int(data.test_mask.sum()),
            "class_counts": {
                str(label): int((data.y == label).sum()) for label in (0, 1)
            },
        }
    for left, right in PAIR_NAMES:
        left_meta = getattr(loaded[left], "metadata", {}) or {}
        right_meta = getattr(loaded[right], "metadata", {}) or {}
        for key in ("review_id_hash", "text_hash", "mask_hash", "embedding_group"):
            if not left_meta.get(key) or left_meta.get(key) != right_meta.get(key):
                raise ValueError(f"{left}/{right} disagree on {key}")
        if not torch.equal(loaded[left].train_mask, loaded[right].train_mask):
            raise ValueError(f"{left}/{right} train masks differ")
    return {"stage": "prepared", "datasets": rows}


def validate_embedded(dataset_root: Path) -> dict:
    prepared = validate_prepared(dataset_root)
    rows = {}
    for name in STAGE1_SOURCE_DATASETS:
        data = _load(dataset_root, name)
        path = dataset_root / name / "qwen3_emb_x.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        emb = obj["emb"] if isinstance(obj, dict) else obj
        if emb.ndim != 2 or emb.size(0) != int(data.num_nodes) or emb.size(1) != 4096:
            raise ValueError(f"{name} invalid embedding shape {tuple(emb.shape)}")
        rows[name] = {"shape": list(emb.shape), "dtype": str(emb.dtype)}
    return {**prepared, "stage": "embedded", "embeddings": rows}


def _audit_ocs(path: Path, train_mask: torch.Tensor, expected: int) -> dict:
    count = 0
    labels = {"Legitimate": 0, "Fraudulent": 0}
    centers = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            center = int(record["id"])
            if not bool(train_mask[center]):
                raise ValueError(f"{path}:{line_number} center {center} is held out")
            invalid = [
                int(node) for node in record["graph"]
                if int(node) >= 0 and not bool(train_mask[int(node)])
            ]
            if invalid:
                raise ValueError(f"{path}:{line_number} leaks held-out nodes {invalid[:10]}")
            label = record["conversations"][1]["value"]
            if label not in labels:
                raise ValueError(f"{path}:{line_number} invalid label {label!r}")
            labels[label] += 1
            centers.append(center)
            count += 1
    if count != expected:
        raise ValueError(f"{path} has {count} records, expected {expected}")
    if labels["Legitimate"] != expected // 2 or labels["Fraudulent"] != expected // 2:
        raise ValueError(f"{path} is not class-balanced: {labels}")
    if len(centers) != len(set(centers)):
        raise ValueError(f"{path} contains duplicate centers")
    return {"records": count, "labels": labels, "centers": centers}


def validate_ocs(dataset_root: Path) -> dict:
    prepared = validate_prepared(dataset_root)
    rows = {}
    centers_by_name = {}
    for name in STAGE1_SOURCE_DATASETS:
        data = _load(dataset_root, name)
        expected = 60000 if name.startswith("yelpzip_") else 10000
        result = _audit_ocs(
            dataset_root / name / "ocs_train.jsonl", data.train_mask.bool(), expected
        )
        centers_by_name[name] = result.pop("centers")
        rows[name] = result
        manifest_path = dataset_root / name / "ocs_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        leakage = manifest.get("leakage_audit") or {}
        if int(leakage.get("leakage", -1)) != 0:
            raise ValueError(f"{name} manifest does not confirm zero leakage")
    for left, right in PAIR_NAMES:
        if centers_by_name[left] != centers_by_name[right]:
            raise ValueError(f"{left}/{right} use different projector centers")
    total = sum(value["records"] for value in rows.values())
    if total != 240000:
        raise ValueError(f"combined OCS records={total}, expected=240000")
    return {**prepared, "stage": "ocs", "ocs": rows, "total_ocs_records": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Yelp/Amazon pretraining artifacts")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepared", "embedded", "ocs"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fn = {
        "prepared": validate_prepared,
        "embedded": validate_embedded,
        "ocs": validate_ocs,
    }[args.stage]
    report = {"status": "VERIFIED", **fn(args.dataset_root)}
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
