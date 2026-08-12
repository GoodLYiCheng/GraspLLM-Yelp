from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


LABEL_TEXT = {0: "Legitimate", 1: "Fraudulent"}


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _record_label(record: dict) -> int:
    conversations = record.get("conversations", [])
    if len(conversations) != 2:
        raise ValueError("Yelp support/evaluation records must contain exactly one prompt and one label")
    label = str(conversations[1].get("value", "")).strip()
    if label not in LABEL_TEXT.values():
        raise ValueError(f"unexpected Yelp label text: {label!r}")
    return int(label == LABEL_TEXT[1])


def build_fewshot_splits(
    records: list[dict], labels: np.ndarray, val_mask: np.ndarray, *, shots: int, seed: int
) -> tuple[list[dict], list[dict], dict[str, list[int]]]:
    """Return balanced labelled support and the disjoint validation remainder."""
    if shots not in (1, 5, 10):
        raise ValueError("shots must be one of 1, 5, 10")
    by_id: dict[int, dict] = {}
    for record in records:
        node_id = int(record["id"])
        if node_id in by_id:
            raise ValueError(f"duplicate validation record id: {node_id}")
        if node_id < 0 or node_id >= labels.size or not bool(val_mask[node_id]):
            raise ValueError(f"record {node_id} is not in the declared validation mask")
        if _record_label(record) != int(labels[node_id]):
            raise ValueError(f"record {node_id} label disagrees with processed_data.pt")
        by_id[node_id] = record

    expected = set(np.flatnonzero(val_mask).tolist())
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise ValueError(f"ocs_val coverage mismatch; missing={missing[:5]}, extra={extra[:5]}")

    rng = np.random.default_rng(seed)
    support_ids: dict[str, list[int]] = {}
    selected: set[int] = set()
    for label in (0, 1):
        candidates = np.asarray(sorted(node for node in by_id if int(labels[node]) == label), dtype=np.int64)
        if candidates.size < shots:
            raise ValueError(f"validation pool has only {candidates.size} examples for label={label}; need {shots}")
        chosen = sorted(rng.choice(candidates, size=shots, replace=False).astype(int).tolist())
        support_ids[LABEL_TEXT[label]] = chosen
        selected.update(chosen)

    support = [by_id[node] for node in sorted(selected)]
    validation = [by_id[node] for node in sorted(set(by_id) - selected)]
    return support, validation, support_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Create strict 1/5/10-shot YelpZip support and validation files")
    parser.add_argument("--processed-data", required=True, type=Path)
    parser.add_argument("--ocs-validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shots", nargs="+", type=int, choices=(1, 5, 10), default=(1, 5, 10))
    parser.add_argument("--seeds", nargs="+", type=int, default=tuple(range(42, 47)))
    args = parser.parse_args()

    data = torch.load(args.processed_data, map_location="cpu", weights_only=False)
    labels = np.asarray(data.y, dtype=np.int8)
    val_mask = np.asarray(data.val_mask, dtype=bool)
    records = _read_jsonl(args.ocs_validation)
    metadata = getattr(data, "metadata", {})

    manifest = {
        "protocol": "static_transductive_few_shot_support_pool",
        "support_source": "yelpzip_validation_mask",
        "support_labels_used_for": "in-context demonstrations",
        "validation_source": "validation_mask excluding support ids",
        "test_labels_used": False,
        "review_id_hash": metadata.get("review_id_hash"),
        "mask_hash": metadata.get("mask_hash"),
        "records_hash": _json_hash([{ "id": int(record["id"]), "label": _record_label(record) } for record in records]),
        "splits": {},
    }
    for shot in args.shots:
        for seed in args.seeds:
            support, validation, support_ids = build_fewshot_splits(
                records, labels, val_mask, shots=shot, seed=seed
            )
            output = args.output_dir / f"k{shot}_seed{seed}"
            output.mkdir(parents=True, exist_ok=True)
            support_path = output / "support_train.jsonl"
            validation_path = output / "validation_holdout.jsonl"
            with support_path.open("w", encoding="utf-8") as handle:
                for record in support:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            with validation_path.open("w", encoding="utf-8") as handle:
                for record in validation:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            entry = {
                "shots_per_class": shot,
                "seed": seed,
                "support_ids": support_ids,
                "support_rows": len(support),
                "validation_rows": len(validation),
                "support_path": str(support_path),
                "validation_path": str(validation_path),
                "support_hash": _json_hash(support_ids),
            }
            (output / "support_manifest.json").write_text(
                json.dumps({**manifest, **entry}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            manifest["splits"][f"k{shot}_seed{seed}"] = entry
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fewshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "COMPLETED", "output_dir": str(args.output_dir), "splits": len(manifest["splits"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
