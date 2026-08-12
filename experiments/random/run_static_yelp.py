from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.grasp_dual_relation.artifacts import stable_hash, write_json
from experiments.grasp_dual_relation.evaluation import binary_metrics, select_f1_threshold
from experiments.random.run import generate_random_scores


def _write(path: Path, data, nodes: np.ndarray, probabilities: np.ndarray, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for node, probability in zip(nodes, probabilities):
            node = int(node)
            handle.write(json.dumps({
                "id": node,
                "review_id": str(data.review_ids[node]),
                "ground_truth": int(data.y[node]),
                "fraud_probability": float(probability),
                "method": "uniform",
                "seed": seed,
            }) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Random baseline on exact static Yelp masks")
    parser.add_argument("--processed-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    args = parser.parse_args()
    data = torch.load(args.processed_data, map_location="cpu", weights_only=False)
    val_nodes = torch.where(data.val_mask)[0].numpy()
    test_nodes = torch.where(data.test_mask)[0].numpy()
    labels = data.y.numpy().astype(np.int8)
    results = []
    for seed in args.seeds:
        val_p = generate_random_scores(
            val_nodes, method="uniform", seed=seed, alignment_fraud_rate=0.5
        )
        test_p = generate_random_scores(
            test_nodes, method="uniform", seed=seed + 10_000, alignment_fraud_rate=0.5
        )
        threshold = select_f1_threshold(labels[val_nodes], val_p)
        result = {
            "seed": seed,
            "validation": binary_metrics(labels[val_nodes], val_p, threshold=threshold),
            "test": binary_metrics(labels[test_nodes], test_p, threshold=threshold),
        }
        results.append(result)
        _write(args.output_dir / f"uniform_seed{seed}_validation.jsonl", data, val_nodes, val_p, seed)
        _write(args.output_dir / f"uniform_seed{seed}_test.jsonl", data, test_nodes, test_p, seed)
        write_json(args.output_dir / f"uniform_seed{seed}_metrics.json", result)
    metric_names = sorted(results[0]["test"])
    summary = {}
    for name in metric_names:
        values = np.asarray([result["test"][name] for result in results], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "values": values.tolist(),
        }
    metadata = getattr(data, "metadata", {})
    manifest = {
        "dataset": metadata.get("dataset"),
        "relation": metadata.get("relation"),
        "protocol": "exact processed_data val_mask/test_mask",
        "score_generation_uses_labels": False,
        "threshold_source": "validation",
        "counts": {"validation": int(val_nodes.size), "test": int(test_nodes.size)},
        "mask_hash": metadata.get("mask_hash"),
        "test_node_ids_hash": stable_hash(test_nodes.tolist()),
        "test_fraud_prevalence": float(labels[test_nodes].mean()),
        "chance_expectation": {"roc_auc": 0.5, "pr_auc": float(labels[test_nodes].mean())},
        "uniform": {"runs": len(results), "test": summary},
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
