from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .evaluation import macro_year_pr_auc


def _read(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = {str(row["node_key"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate node_key in {path}")
    return result


def paired_comparison(full_rows, baseline_rows, *, iterations: int, seed: int) -> dict:
    if set(full_rows) != set(baseline_rows):
        raise ValueError("paired Gate inputs have different node keys")
    keys = sorted(full_rows)
    full = [full_rows[key] for key in keys]
    base = [baseline_rows[key] for key in keys]
    for left, right in zip(full, base):
        if (int(left["year"]), int(left["ground_truth"])) != (int(right["year"]), int(right["ground_truth"])):
            raise ValueError("paired Gate rows disagree on year/label")
    years = np.asarray([int(row["year"]) for row in full])
    labels = np.asarray([int(row["ground_truth"]) for row in full])
    full_p = np.asarray([float(row["fraud_probability"]) for row in full])
    base_p = np.asarray([float(row["fraud_probability"]) for row in base])
    observed = macro_year_pr_auc(years, labels, full_p) - macro_year_pr_auc(years, labels, base_p)
    strata = [np.where((years == year) & (labels == label))[0] for year in np.unique(years) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in strata])
        deltas[iteration] = (
            macro_year_pr_auc(years[sample], labels[sample], full_p[sample])
            - macro_year_pr_auc(years[sample], labels[sample], base_p[sample])
        )
    return {
        "delta_macro_pr_auc": float(observed),
        "paired_bootstrap_iterations": iterations,
        "ci_95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "one_sided_p": float((1 + np.sum(deltas <= 0)) / (iterations + 1)),
    }


def _holm(results: dict[str, dict]) -> None:
    ordered = sorted(results, key=lambda name: results[name]["one_sided_p"])
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * results[name]["one_sided_p"])
        running = max(running, adjusted)
        results[name]["holm_adjusted_p"] = running


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the pre-registered FiGraph MVP Gate")
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--text-matched", required=True, type=Path)
    parser.add_argument("--random-graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20250814)
    args = parser.parse_args()
    full = _read(args.full)
    comparisons = {
        "full_vs_text_only_matched": paired_comparison(full, _read(args.text_matched), iterations=args.iterations, seed=args.seed),
        "full_vs_random_graph": paired_comparison(full, _read(args.random_graph), iterations=args.iterations, seed=args.seed + 1),
    }
    _holm(comparisons)
    for value in comparisons.values():
        value["passes"] = bool(
            value["delta_macro_pr_auc"] >= 0.01
            and value["ci_95"][0] > 0
            and value["holm_adjusted_p"] < 0.05
        )
    passed = all(value["passes"] for value in comparisons.values())
    payload = {
        "status": "PASS" if passed else "STOP",
        "metric": "macro mean of 2021 and 2022 PR-AUC",
        "requirements": {"minimum_delta": 0.01, "bootstrap_ci_lower_gt_zero": True, "holm_one_sided_p_lt": 0.05},
        "comparisons": comparisons,
        "on_failure": "stop 20-seed, K=16, YaRN, and higher-order motif extensions",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
