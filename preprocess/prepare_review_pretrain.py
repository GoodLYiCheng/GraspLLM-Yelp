from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from preprocess.prepare_yelpzip import (
    _hash_lines,
    build_bounded_relation_graph,
    prepare as prepare_yelpzip,
)
from preprocess.review_sources import AMAZON_SOURCES, STANDARD_FIELDS, sha256_file


DERIVED_NAMES = (
    "processed_data.pt",
    "qwen3_emb_x.pt",
    "ocs_train.jsonl",
    "ocs_val.jsonl",
    "ocs_test.jsonl",
    "ocs_manifest.json",
    "run_manifest.json",
)


def _read_standard_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if tuple(record.keys()) != STANDARD_FIELDS:
                raise ValueError(
                    f"{path}:{line_number} schema mismatch: got={tuple(record.keys())}"
                )
            if int(record["label"]) not in (0, 1):
                raise ValueError(f"{path}:{line_number} invalid label={record['label']!r}")
            records.append(record)
    ids = [str(record["review_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate review_id values")
    return records


def _clear_or_reject(output_dirs: list[Path], *, overwrite: bool) -> None:
    for output_dir in output_dirs:
        processed = output_dir / "processed_data.pt"
        if processed.exists() and not overwrite:
            raise FileExistsError(f"{processed} exists; pass --overwrite to rebuild")
    if overwrite:
        for output_dir in output_dirs:
            for name in DERIVED_NAMES:
                path = output_dir / name
                if path.is_file():
                    path.unlink()


def _prepare_amazon(args: argparse.Namespace) -> dict:
    manifests = {}
    for source in AMAZON_SOURCES:
        raw_path = args.raw_root / "amazon" / f"amazon_{source.slug}_100k.jsonl"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        records = _read_standard_jsonl(raw_path)
        if len(records) != args.amazon_rows_per_category:
            raise ValueError(
                f"{raw_path} must contain {args.amazon_rows_per_category} rows, got {len(records)}"
            )
        labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
        class_counts = {str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)}
        expected_per_class = args.amazon_rows_per_category // 2
        if any(value != expected_per_class for value in class_counts.values()):
            raise ValueError(
                f"{raw_path} must be class-balanced at {expected_per_class}/class, got {class_counts}"
            )
        review_ids = [str(record["review_id"]) for record in records]
        raw_texts = [str(record["review_text"]) for record in records]
        train_mask = torch.ones(len(records), dtype=torch.bool)
        val_mask = torch.zeros(len(records), dtype=torch.bool)
        test_mask = torch.zeros(len(records), dtype=torch.bool)
        common = {
            "y": torch.from_numpy(labels).long(),
            "train_mask": train_mask,
            "pretrain_mask": train_mask.clone(),
            "val_mask": val_mask,
            "test_mask": test_mask,
            "raw_texts": raw_texts,
            "review_ids": review_ids,
            "label_texts": ["Legitimate", "Fraudulent"],
        }
        relation_specs = (
            (f"{source.dataset_prefix}_rur", "reviewer_id", "user", "RUR"),
            (f"{source.dataset_prefix}_rpr", "product_id", "product", "RPR"),
        )
        output_dirs = [args.dataset_root / spec[0] for spec in relation_specs]
        _clear_or_reject(output_dirs, overwrite=args.overwrite)
        review_id_hash = _hash_lines(review_ids)
        text_hash = _hash_lines(raw_texts)
        mask_hash = _hash_lines(np.flatnonzero(train_mask.numpy()).tolist())
        for dataset_name, entity_field, relation_text, relation_code in relation_specs:
            entity_ids = np.asarray([str(record[entity_field]) for record in records], dtype=object)
            edge_index, motif_adj, graph_stats = build_bounded_relation_graph(
                entity_ids, max_neighbors=args.max_neighbors, seed=args.seed
            )
            metadata = {
                "dataset": dataset_name,
                "source_domain": "amazon",
                "source_category": source.category,
                "raw_path": str(raw_path.resolve()),
                "raw_sha256": sha256_file(raw_path),
                "relation": relation_code,
                "relation_text": relation_text,
                "entity_field": entity_field,
                "node_feature": "review_text_only",
                "uses_labels_for_graph": False,
                "training_scope": "all_sampled_nodes",
                "evaluation_scope": "source_training_only",
                "seed": args.seed,
                "max_neighbors": args.max_neighbors,
                "review_id_hash": review_id_hash,
                "text_hash": text_hash,
                "mask_hash": mask_hash,
                "embedding_group": source.dataset_prefix,
                "train_rows": len(records),
                "validation_rows": 0,
                "test_rows": 0,
                "class_counts": class_counts,
                **graph_stats,
            }
            data = Data(edge_index=edge_index, num_nodes=len(records), **common)
            data.motif_adj = motif_adj
            data.metadata = metadata
            output_dir = args.dataset_root / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(data, output_dir / "processed_data.pt")
            (output_dir / "run_manifest.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            manifests[dataset_name] = metadata
    return manifests


def prepare(args: argparse.Namespace) -> None:
    yelp_args = argparse.Namespace(
        raw_path=args.raw_root / "Yelp-Dataset" / "yelpzip.csv",
        dataset_root=args.dataset_root,
        max_neighbors=args.max_neighbors,
        val_size=args.yelp_val_size,
        test_size=args.yelp_test_size,
        seed=args.seed,
        max_rows=None,
        overwrite=args.overwrite,
    )
    prepare_yelpzip(yelp_args)
    amazon = _prepare_amazon(args)
    summary = {
        "status": "COMPLETED",
        "datasets": ["yelpzip_rur", "yelpzip_rbr", *amazon.keys()],
        "dataset_count": 2 + len(amazon),
        "raw_root": str(args.raw_root.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "seed": args.seed,
    }
    (args.dataset_root / "yelp_amazon_pretrain_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Yelp/Amazon review graphs")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--amazon-rows-per-category", type=int, default=100000)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--yelp-val-size", type=int, default=10000)
    parser.add_argument("--yelp-test-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
