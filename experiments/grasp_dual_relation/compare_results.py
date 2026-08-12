from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from .artifacts import write_json


def _load(path: Path) -> dict[int, tuple[int, float]]:
    rows: dict[int, tuple[int, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            node = int(record["node_id"])
            if node in rows:
                raise ValueError(f"duplicate node_id {node} in {path}")
            rows[node] = (int(record["ground_truth"]), float(record["fraud_probability"]))
    if not rows:
        raise ValueError(f"no predictions in {path}")
    return rows


def _aligned(candidate: dict, baseline: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if candidate.keys() != baseline.keys():
        missing = sorted(candidate.keys() - baseline.keys())[:10]
        extra = sorted(baseline.keys() - candidate.keys())[:10]
        raise ValueError(f"{name} node IDs differ; missing={missing}, extra={extra}")
    nodes = sorted(candidate)
    labels = np.asarray([candidate[node][0] for node in nodes], dtype=np.int8)
    other_labels = np.asarray([baseline[node][0] for node in nodes], dtype=np.int8)
    if not np.array_equal(labels, other_labels):
        raise ValueError(f"{name} ground-truth labels differ from candidate")
    candidate_p = np.asarray([candidate[node][1] for node in nodes], dtype=np.float64)
    baseline_p = np.asarray([baseline[node][1] for node in nodes], dtype=np.float64)
    return labels, candidate_p, baseline_p


def _paired_stratified_deltas(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(index.size == 0 for index in by_class):
        raise ValueError("paired bootstrap requires both classes")
    deltas = np.empty(samples, dtype=np.float64)
    for offset in range(samples):
        draw = np.concatenate(
            [rng.choice(index, size=index.size, replace=True) for index in by_class]
        )
        deltas[offset] = average_precision_score(labels[draw], candidate[draw]) - average_precision_score(
            labels[draw], baseline[draw]
        )
    return deltas


def _holm_adjust(raw_p: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * raw_p[name]))
        adjusted[name] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired PR-AUC bootstrap against locked baselines")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--baseline", action="append", required=True, metavar="NAME=PATH",
        help="repeat for text and merged baselines",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--minimum-delta", type=float, default=0.01)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    baselines: dict[str, Path] = {}
    for specification in args.baseline:
        name, separator, raw_path = specification.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid baseline specification: {specification!r}")
        baselines[name] = Path(raw_path)

    candidate = _load(args.candidate)
    comparisons = {}
    raw_p = {}
    family_size = len(baselines)
    tail = args.alpha / (2.0 * family_size)
    for offset, (name, path) in enumerate(baselines.items()):
        labels, candidate_p, baseline_p = _aligned(candidate, _load(path), name)
        observed = float(
            average_precision_score(labels, candidate_p) - average_precision_score(labels, baseline_p)
        )
        deltas = _paired_stratified_deltas(
            labels,
            candidate_p,
            baseline_p,
            samples=args.bootstrap_samples,
            seed=args.seed + offset,
        )
        lower, upper = np.quantile(deltas, [tail, 1.0 - tail])
        p_value = float((1 + np.count_nonzero(deltas <= 0.0)) / (args.bootstrap_samples + 1))
        raw_p[name] = p_value
        comparisons[name] = {
            "rows": int(labels.size),
            "candidate_pr_auc": float(average_precision_score(labels, candidate_p)),
            "baseline_pr_auc": float(average_precision_score(labels, baseline_p)),
            "delta_pr_auc": observed,
            "simultaneous_ci": [float(lower), float(upper)],
            "raw_one_sided_p": p_value,
        }
    adjusted = _holm_adjust(raw_p)
    for name, result in comparisons.items():
        result["holm_adjusted_p"] = adjusted[name]
        result["passes"] = bool(
            result["delta_pr_auc"] >= args.minimum_delta
            and result["simultaneous_ci"][0] > 0.0
            and result["holm_adjusted_p"] < args.alpha
        )
    payload = {
        "metric": "PR-AUC",
        "bootstrap": "paired stratified by label",
        "simultaneous_ci": f"Bonferroni family-wise {1.0 - args.alpha:.1%}",
        "bootstrap_samples": args.bootstrap_samples,
        "minimum_delta": args.minimum_delta,
        "comparisons": comparisons,
        "expand_to_full_test": bool(all(item["passes"] for item in comparisons.values())),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
