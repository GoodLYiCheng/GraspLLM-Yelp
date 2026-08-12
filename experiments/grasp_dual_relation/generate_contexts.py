from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .contexts import _select, build_dual_context, training_record
from .data import load_yelpzip
from .graph import CausalRelationGraph, causal_dfs, merge_causal_relation_graphs
from .split import Split, stratified_time_sample


def _load_embedding(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    value = obj["emb"] if isinstance(obj, dict) else obj
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} does not contain a tensor")
    return value.float().numpy()


def _graph(bundle, relation: str, max_history: int) -> CausalRelationGraph:
    return CausalRelationGraph(
        indptr=bundle[f"{relation}_indptr"],
        indices=bundle[f"{relation}_indices"],
        relation=relation,
        max_history=max_history,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate causal dual-relation GraspLLM JSONL")
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--graph-bundle", required=True, type=Path)
    parser.add_argument("--text-embedding", required=True, type=Path)
    parser.add_argument("--user-structure-embedding", required=True, type=Path)
    parser.add_argument("--business-structure-embedding", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", choices=["ocs", "random", "text_topk"], default="ocs")
    parser.add_argument(
        "--variant",
        choices=["dual", "user_only", "business_only", "merged", "text_only"],
        default="dual",
    )
    parser.add_argument("--splits", nargs="+", choices=["alignment", "validation", "test"], default=["alignment", "validation", "test"])
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--user-k", type=int, default=8)
    parser.add_argument("--business-k", type=int, default=8)
    parser.add_argument("--merged-k", type=int, default=16)
    parser.add_argument("--beta-user", type=float, default=0.55)
    parser.add_argument("--beta-business", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-sample-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="smoke-only raw row cap")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    events = load_yelpzip(args.raw_path, max_rows=args.max_rows)
    bundle = np.load(args.graph_bundle)
    if not np.array_equal(events.timestamps, bundle["timestamps"]):
        raise ValueError("raw YelpZip ordering/timestamps do not match graph bundle")
    if not np.array_equal(events.labels, bundle["labels"]):
        raise ValueError("raw YelpZip labels do not match graph bundle")
    text_embeddings = _load_embedding(args.text_embedding)
    user_structure = _load_embedding(args.user_structure_embedding)
    business_structure = _load_embedding(args.business_structure_embedding)
    expected_rows = len(events)
    if any(values.shape[0] != expected_rows for values in (text_embeddings, user_structure, business_structure)):
        raise ValueError("all embedding files must have one row per canonical review node")

    user_graph = _graph(bundle, "user", 16)
    business_graph = _graph(bundle, "business", 32)
    merged_graph = merge_causal_relation_graphs(user_graph, business_graph, max_history=48)
    merged_structure = np.concatenate([user_structure, business_structure], axis=1)
    split_map = {
        "alignment": Split.ALIGNMENT,
        "validation": Split.VALIDATION,
        "test": Split.TEST,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split_name in args.splits:
        nodes = np.flatnonzero(bundle["assignments"] == int(split_map[split_name]))
        if args.query_sample_size is not None:
            nodes = stratified_time_sample(
                nodes, events.labels, sample_size=args.query_sample_size, seed=args.seed
            )
        if args.max_queries is not None:
            nodes = nodes[: args.max_queries]
        prefix = args.method if args.variant == "dual" else f"{args.variant}_{args.method}"
        path = args.output_dir / f"{prefix}_{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            iterator = tqdm(nodes, desc=f"contexts-{args.method}-{split_name}", disable=args.no_progress)
            for center in iterator:
                context = build_dual_context(
                    int(center),
                    events,
                    user_graph,
                    business_graph,
                    text_embeddings,
                    user_structure,
                    business_structure,
                    method=args.method,
                    max_depth=args.max_depth,
                    user_k=args.user_k,
                    business_k=args.business_k,
                    beta_user=args.beta_user,
                    beta_business=args.beta_business,
                    seed=args.seed,
                )
                merged_nodes = None
                if args.variant == "merged":
                    candidates = causal_dfs(
                        int(center), merged_graph, events.timestamps, max_depth=args.max_depth
                    )
                    merged_nodes = _select(
                        args.method,
                        int(center),
                        candidates,
                        text_embeddings,
                        merged_structure,
                        merged_graph,
                        k=args.merged_k,
                        beta=(args.beta_user + args.beta_business) / 2.0,
                        seed=args.seed,
                    )
                record = training_record(
                    events, context, variant=args.variant, merged_nodes=merged_nodes
                )
                record["dataset"] = "yelpzip_grasp"
                record["split"] = split_name
                record["ground_truth"] = int(events.labels[int(center)])
                if split_name != "alignment":
                    record["conversations"] = record["conversations"][:1]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[split_name] = {"path": str(path), "rows": int(nodes.size)}
    print(json.dumps({
        "status": "COMPLETED", "variant": args.variant, "method": args.method, "outputs": outputs
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
