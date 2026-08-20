"""Combine no-ICL/baseline and ICL YelpZip metrics into one report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RELATIONS = ("yelpzip_rur", "yelpzip_rbr")
SHOTS = (1, 5, 10)
SEEDS = (42, 43, 44, 45, 46)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing required metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(records: list[dict]) -> dict:
    names = sorted(records[0]["test"])
    return {
        name: {
            "mean": float(np.mean([record["test"][name] for record in records])),
            "std": float(np.std([record["test"][name] for record in records], ddof=1)),
        }
        for name in names
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize static YelpZip no-ICL/ICL results")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--no-icl-dir-name", default="zero_shot")
    args = parser.parse_args()
    root = args.root
    baseline_key = "no_icl" if args.no_icl_dir_name == "no_icl" else "zero_shot"
    baseline_label = "no-ICL" if baseline_key == "no_icl" else "zero-shot"
    payload: dict = {
        "root": str(root.resolve()), baseline_key: {}, "in_context_few_shot": {}
    }

    baseline_paths = {
        relation: root / args.no_icl_dir_name / relation / "probability_metrics.json"
        for relation in RELATIONS
    }
    baseline_exists = {relation: path.is_file() for relation, path in baseline_paths.items()}
    if any(baseline_exists.values()) and not all(baseline_exists.values()):
        raise FileNotFoundError(
            f"partial {baseline_label} results found; both RUR and RBR metrics are required or neither"
        )

    for relation in RELATIONS:
        if all(baseline_exists.values()):
            payload[baseline_key][relation] = _read(baseline_paths[relation])
        payload["in_context_few_shot"][relation] = {}
        for shot in SHOTS:
            records = [_read(
                root / "few_shot" / relation / f"k{shot}_seed{seed}" / "probability_metrics.json"
            ) for seed in SEEDS]
            payload["in_context_few_shot"][relation][f"k{shot}"] = {
                "seeds": list(SEEDS),
                "test_mean_std": _mean_std(records),
                "runs": records,
            }

    (root / "all_results_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# YelpZip GraspLLM evaluation summary", "",
        "| Relation | Setting | PR-AUC | ROC-AUC | Fraud F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for relation in RELATIONS:
        if relation in payload[baseline_key]:
            baseline = payload[baseline_key][relation]["test"]
            lines.append(
                f"| {relation} | {baseline_label} | {baseline['pr_auc']:.6f} | "
                f"{baseline['roc_auc']:.6f} | {baseline['fraud_f1']:.6f} |"
            )
        for shot in SHOTS:
            metrics = payload["in_context_few_shot"][relation][f"k{shot}"]["test_mean_std"]
            lines.append(
                f"| {relation} | {shot}-shot ICL | "
                f"{metrics['pr_auc']['mean']:.6f} +/- {metrics['pr_auc']['std']:.6f} | "
                f"{metrics['roc_auc']['mean']:.6f} +/- {metrics['roc_auc']['std']:.6f} | "
                f"{metrics['fraud_f1']['mean']:.6f} +/- {metrics['fraud_f1']['std']:.6f} |"
            )
    (root / "all_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote {root / 'all_results_summary.json'}")
    print(f"[summary] wrote {root / 'all_results_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
