from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.grasp_dual_relation.artifacts import stable_hash, write_json
from experiments.grasp_dual_relation.data import ReviewEventTable, load_yelpzip
from experiments.grasp_dual_relation.evaluation import binary_metrics, select_f1_threshold
from experiments.grasp_dual_relation.split import Split, stratified_time_sample


def generate_random_scores(
    node_ids: np.ndarray,
    *,
    method: str,
    seed: int,
    alignment_fraud_rate: float,
) -> np.ndarray:
    """Generate label-blind scores for fixed query IDs."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    if method == "uniform":
        rng = np.random.default_rng(seed)
        return rng.random(node_ids.size, dtype=np.float64)
    if method == "alignment_prior":
        return np.full(node_ids.size, alignment_fraud_rate, dtype=np.float64)
    raise ValueError(f"unknown random baseline method: {method}")


def _write_predictions(
    path: Path,
    events: ReviewEventTable,
    node_ids: np.ndarray,
    probabilities: np.ndarray,
    *,
    method: str,
    seed: int,
    split_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for node, probability in zip(node_ids, probabilities):
            node = int(node)
            handle.write(json.dumps({
                "id": str(events.review_ids[node]),
                "node_id": node,
                "timestamp": int(events.timestamps[node]),
                "ground_truth": int(events.labels[node]),
                "fraud_probability": float(probability),
                "method": method,
                "seed": seed,
                "split": split_name,
            }) + "\n")


def _aggregate(seed_results: list[dict]) -> dict:
    names = sorted(seed_results[0]["test"])
    summary = {}
    for name in names:
        values = np.asarray([result["test"][name] for result in seed_results], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run label-blind temporal random baselines on YelpZip")
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--methods", nargs="+", choices=["uniform", "alignment_prior"],
        default=["uniform", "alignment_prior"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--test-sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="smoke-only raw row cap")
    args = parser.parse_args()

    events = load_yelpzip(args.raw_path, max_rows=args.max_rows)
    bundle = np.load(args.graph_bundle)
    for key, expected in (("timestamps", events.timestamps), ("labels", events.labels)):
        if not np.array_equal(bundle[key], expected):
            raise ValueError(f"raw data and graph bundle disagree on {key}")
    assignments = bundle["assignments"]
    alignment_nodes = np.flatnonzero(assignments == int(Split.ALIGNMENT))
    validation_nodes = np.flatnonzero(assignments == int(Split.VALIDATION))
    test_nodes = np.flatnonzero(assignments == int(Split.TEST))
    if args.test_sample_size is not None:
        test_nodes = stratified_time_sample(
            test_nodes,
            events.labels,
            sample_size=args.test_sample_size,
            seed=args.sample_seed,
        )
    alignment_rate = float(events.labels[alignment_nodes].mean())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method in args.methods:
        method_results = []
        method_seeds = args.seeds if method == "uniform" else [args.seeds[0]]
        for seed in method_seeds:
            val_prob = generate_random_scores(
                validation_nodes,
                method=method,
                seed=seed,
                alignment_fraud_rate=alignment_rate,
            )
            test_prob = generate_random_scores(
                test_nodes,
                method=method,
                seed=seed + 10_000,
                alignment_fraud_rate=alignment_rate,
            )
            threshold = select_f1_threshold(events.labels[validation_nodes], val_prob)
            result = {
                "method": method,
                "seed": seed,
                "threshold_source": "validation",
                "validation": binary_metrics(
                    events.labels[validation_nodes], val_prob, threshold=threshold
                ),
                "test": binary_metrics(events.labels[test_nodes], test_prob, threshold=threshold),
            }
            method_results.append(result)
            _write_predictions(
                args.output_dir / f"{method}_seed{seed}_validation.jsonl",
                events,
                validation_nodes,
                val_prob,
                method=method,
                seed=seed,
                split_name="validation",
            )
            _write_predictions(
                args.output_dir / f"{method}_seed{seed}_test.jsonl",
                events,
                test_nodes,
                test_prob,
                method=method,
                seed=seed,
                split_name="test",
            )
            write_json(args.output_dir / f"{method}_seed{seed}_metrics.json", result)
        all_results[method] = {
            "runs": len(method_results),
            "test": _aggregate(method_results),
        }

    manifest = {
        "material_passport": {
            "origin": "YelpZip temporal random difficulty baseline",
            "verification_status": "VERIFIED",
            "version": "random_temporal_v1",
        },
        "protocol": {
            "split_source": str(args.graph_bundle),
            "temporal_split_preserved": True,
            "score_generation_uses_query_labels": False,
            "threshold_source": "validation",
            "test_labels_used_for_metrics_only": True,
            "test_sampling": "label-and-time stratified fixed sample" if args.test_sample_size else "full test",
        },
        "counts": {
            "alignment": int(alignment_nodes.size),
            "validation": int(validation_nodes.size),
            "test": int(test_nodes.size),
        },
        "alignment_fraud_rate": alignment_rate,
        "test_fraud_prevalence": float(events.labels[test_nodes].mean()),
        "chance_expectation": {"roc_auc": 0.5, "pr_auc": float(events.labels[test_nodes].mean())},
        "node_ids_hash": stable_hash(test_nodes.tolist()),
        "methods": all_results,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
