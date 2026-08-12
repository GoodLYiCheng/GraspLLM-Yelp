from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import build_manifest, write_json
from .audit import audit_temporal_contract
from .config import default_config
from .data import load_yelpzip
from .graph import build_causal_relation_graph
from .split import Split, temporal_split
from .support import sample_balanced_support


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare(args: argparse.Namespace) -> int:
    cfg = default_config(_repo_root())
    raw_path = Path(args.raw_path) if args.raw_path else cfg.raw_path
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    cfg = replace(cfg, raw_path=raw_path, output_dir=output_dir)
    events = load_yelpzip(raw_path, max_rows=args.max_rows)
    split = temporal_split(events.timestamps)
    user_graph = build_causal_relation_graph(
        events.user_ids, events.timestamps, max_history=cfg.max_user_history, relation="user"
    )
    business_graph = build_causal_relation_graph(
        events.business_ids,
        events.timestamps,
        max_history=cfg.max_business_history,
        relation="business",
    )
    report = audit_temporal_contract(
        events.timestamps,
        split,
        (user_graph, business_graph),
        labels=events.labels,
        max_depth=cfg.max_depth,
    )
    if not report.passed:
        raise RuntimeError(f"temporal audit failed: {report.checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "temporal_graphs.npz",
        timestamps=events.timestamps,
        labels=events.labels,
        assignments=split.assignments,
        user_indptr=user_graph.indptr,
        user_indices=user_graph.indices,
        business_indptr=business_graph.indptr,
        business_indices=business_graph.indices,
    )
    write_json(output_dir / "review_ids.json", events.review_ids.tolist())
    supports = {
        str(shot): {
            str(seed): sample_balanced_support(
                events.labels, split.assignments, shots_per_class=shot, seed=seed
            )
            for seed in range(42, 47)
        }
        for shot in (1, 5, 10)
    }
    write_json(output_dir / "support_ids.json", supports)
    write_json(output_dir / "leakage_audit.json", report)
    manifest = build_manifest(
        config=cfg.to_manifest_dict(),
        node_order_hash=events.node_order_hash(),
        artifacts={
            "graphs": "temporal_graphs.npz",
            "review_ids": "review_ids.json",
            "supports": "support_ids.json",
            "audit": "leakage_audit.json",
        },
    )
    manifest["counts"] = {
        "total": len(events),
        **{part.name.lower(): int((split.assignments == int(part)).sum()) for part in Split},
        "fraud": int(events.labels.sum()),
        "user_edges": user_graph.num_edges,
        "business_edges": business_graph.num_edges,
    }
    motif_modes = {
        relation: result["recommended_mode"]
        for relation, result in report.details["motif_audit"].items()
    }
    manifest["protocol"] = {
        "temporal_order": "strict timestamp; equal timestamps excluded",
        "message_direction": "history_to_current",
        "dfs_direction": "current_to_predecessor",
        "motif_audit": motif_modes,
        "selected_motif_mode": (
            "edge_only" if "edge_only" in motif_modes.values() else "four_motif"
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": "PASSED", "output_dir": str(output_dir), **manifest["counts"]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and audit temporal dual-relation YelpZip graphs")
    parser.add_argument("--raw-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="reserved for compatible long-running stages")
    parser.set_defaults(func=prepare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
