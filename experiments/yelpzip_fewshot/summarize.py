"""Combine zero-shot and ICL YelpZip probability metrics into one report."""
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
    parser = argparse.ArgumentParser(description="Summarize all static YelpZip zero/ICL results")
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    payload: dict = {"root": str(root.resolve()), "zero_shot": {}, "in_context_few_shot": {}}

    zero_paths = {
        relation: root / "zero_shot" / relation / "probability_metrics.json"
        for relation in RELATIONS
    }
    zero_exists = {relation: path.is_file() for relation, path in zero_paths.items()}
    if any(zero_exists.values()) and not all(zero_exists.values()):
        raise FileNotFoundError(
            "partial zero-shot results found; both RUR and RBR metrics are required or neither"
        )

    for relation in RELATIONS:
        if all(zero_exists.values()):
            payload["zero_shot"][relation] = _read(zero_paths[relation])
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
        if relation in payload["zero_shot"]:
            zero = payload["zero_shot"][relation]["test"]
            lines.append(
                f"| {relation} | zero-shot | {zero['pr_auc']:.6f} | "
                f"{zero['roc_auc']:.6f} | {zero['fraud_f1']:.6f} |"
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
