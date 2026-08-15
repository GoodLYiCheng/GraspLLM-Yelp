from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.grasp_dual_relation.evaluation import binary_metrics, select_f1_threshold
from experiments.grasp_dual_relation.scoring import binary_answer_probability
from model.builder import load_pretrained_model
from utils.chat_format import build_eval_prompt, build_multi_turn_eval_prompt, install_chat_template
from utils.constants import DEFAULT_GRAPH_PAD_ID
from utils.paths import dataset_dir
from utils.utils import get_model_name_from_path
from experiments.yelpzip_fewshot.icl import (
    build_icl_conversations, stack_graph_node_ids, support_entries_from_records,
    stratified_record_subset, truncate_query_review_text, validate_icl_query_records,
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_embedding(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    value = obj["emb"] if isinstance(obj, dict) else obj
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError("qwen3_emb_x.pt must contain a [N,D] tensor")
    return value.float()


def _load_jsonl(path: Path, max_queries: int | None) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return records if max_queries is None else records[:max_queries]


def _graph_inputs(record: dict, embeddings: torch.Tensor, device: torch.device, *,
                  support_records: list[dict] | None = None) -> dict:
    records = [*(support_records or []), record]
    nodes = stack_graph_node_ids(records, pad_id=DEFAULT_GRAPH_PAD_ID)
    mask = nodes != DEFAULT_GRAPH_PAD_ID
    graph_emb = torch.zeros((nodes.shape[0], nodes.shape[1], embeddings.shape[1]), dtype=torch.float32)
    graph_emb[mask] = embeddings[nodes[mask]]
    return {
        "graph": nodes.to(device),
        "graph_emb": graph_emb.to(device=device, dtype=torch.float16),
    }


def _score_split(
    records: list[dict], model, tokenizer, embeddings: torch.Tensor,
    device: torch.device, output_path: Path, *, max_length: int,
    icl_support: list[tuple[int, int]] | None = None,
    icl_support_records: list[dict] | None = None,
    icl_support_graphs: bool = False,
    raw_texts: list[str] | None = None,
    icl_support_max_tokens: int | None = None,
    query_max_tokens: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = [], []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in tqdm(records, desc=f"score-{output_path.stem}"):
            user_text = truncate_query_review_text(
                tokenizer, record["conversations"][0]["value"], query_max_tokens
            )
            label_text = record["conversations"][1]["value"].strip()
            if label_text not in ("Fraudulent", "Legitimate"):
                raise ValueError(f"unexpected Yelp label text: {label_text!r}")
            label = int(label_text == "Fraudulent")
            if icl_support is None:
                prompt_ids = build_eval_prompt(
                    tokenizer, user_text, has_graph=True, max_length=None
                )
            else:
                if raw_texts is None:
                    raise ValueError("raw_texts are required for ICL scoring")
                conversations = build_icl_conversations(
                    support=icl_support,
                    raw_texts=raw_texts,
                    query_user_text=user_text,
                    tokenizer=tokenizer,
                    support_max_tokens=icl_support_max_tokens,
                    include_support_graphs=icl_support_graphs,
                )
                prompt_ids = build_multi_turn_eval_prompt(
                    tokenizer, conversations, has_graph=True, max_length=None
                )
            graph_records = icl_support_records if icl_support_graphs else None
            graph_marker_count = int((prompt_ids < 0).sum().item())
            expected_graphs = 1 + (len(icl_support_records or []) if icl_support_graphs else 0)
            if graph_marker_count != expected_graphs:
                raise ValueError(
                    f"prompt/graph alignment mismatch: markers={graph_marker_count}, "
                    f"graph rows={expected_graphs}"
                )
            graph_kwargs = _graph_inputs(
                record, embeddings, device, support_records=graph_records,
            )
            expanded_length = (
                int(prompt_ids.numel()) - graph_marker_count
                # The original single-graph projector emits one token for every
                # stored graph position, including -500 padding positions.
                # Count the exact injected tensor size, not only valid nodes.
                + int(graph_kwargs["graph"].numel())
                + 8  # reserve space for either canonical answer sequence
            )
            if expanded_length > max_length:
                raise ValueError(
                    f"expanded graph-aware prompt has {expanded_length} tokens, exceeding "
                    f"max_length={max_length}; full Yelp text is never silently truncated. "
                    "Use explicit --icl-support-max-tokens/--query-max-tokens or raise --max-length."
                )
            probability = binary_answer_probability(
                model,
                tokenizer,
                prompt_ids,
                model_kwargs=graph_kwargs,
            )
            labels.append(label)
            probabilities.append(probability)
            output.write(json.dumps({
                "id": record["id"],
                "review_id": record.get("review_id"),
                "ground_truth": label,
                "fraud_probability": probability,
                "support_node_ids": [] if icl_support is None else [node for node, _ in icl_support],
                "prompt_tokens_expanded": expanded_length,
            }) + "\n")
    return np.asarray(labels, dtype=np.int8), np.asarray(probabilities, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequence-likelihood YelpZip evaluation")
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-base", default="/data/Qwen/Qwen3-8B")
    parser.add_argument("--dataset", required=True, choices=["yelpzip_rur", "yelpzip_rbr"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=32768,
                        help="Qwen3 prompt ceiling including expanded graph tokens (default: 32768)")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-validation-queries", type=int, default=None,
                        help="Deterministically stratify and cap validation only; test remains complete")
    parser.add_argument("--validation-subsample-seed", type=int, default=42,
                        help="Seed for --max-validation-queries (default: 42)")
    parser.add_argument("--validation-jsonl", type=Path, default=None,
                        help="Optional disjoint validation file, e.g. few-shot holdout")
    parser.add_argument("--test-jsonl", type=Path, default=None,
                        help="Optional test contexts; defaults to the dataset test split")
    parser.add_argument("--support-manifest", type=Path, default=None,
                        help="Record support provenance for a few-shot run")
    parser.add_argument("--icl-support-jsonl", type=Path, default=None,
                        help="Labelled support examples injected into each query prompt; no parameters are trained")
    parser.add_argument("--icl-support-max-tokens", type=int, default=None,
                        help="Optional cap for each support review; omitted keeps full text")
    parser.add_argument("--icl-support-graphs", action="store_true",
                        help="Attach each support record's own graph embedding to its <graph> marker")
    parser.add_argument("--query-max-tokens", type=int, default=None,
                        help="Optional cap for target review text; omitted keeps full text")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data_dir = Path(dataset_dir(args.dataset))
    processed_path = data_dir / "processed_data.pt"
    embedding_path = data_dir / "qwen3_emb_x.pt"
    processed = torch.load(processed_path, map_location="cpu", weights_only=False)
    embeddings = _load_embedding(embedding_path)
    if embeddings.shape[0] != int(processed.num_nodes):
        raise ValueError("embedding rows do not match processed Yelp nodes")
    if args.icl_support_jsonl and not args.support_manifest:
        parser.error("--icl-support-jsonl requires --support-manifest for provenance")
    if args.icl_support_jsonl:
        icl_records = _load_jsonl(args.icl_support_jsonl, max_queries=None)
        icl_support = support_entries_from_records(
            icl_records,
            np.asarray(processed.y, dtype=np.int8),
            np.asarray(processed.val_mask, dtype=bool),
        )
        raw_texts = [str(text) for text in processed.raw_texts]
    else:
        icl_records = None
        icl_support = None
        raw_texts = None
    if args.icl_support_graphs and not args.icl_support_jsonl:
        parser.error("--icl-support-graphs requires --icl-support-jsonl")

    model_name = get_model_name_from_path(str(args.model_path))
    tokenizer, model, _ = load_pretrained_model(
        str(args.model_path), args.model_base, model_name,
        cache_dir=str(args.model_path / "_eval_cache"),
    )
    install_chat_template(tokenizer, model_name_or_path=args.model_base)
    model = model.to(device=device, dtype=torch.float16).eval()

    val_path = args.validation_jsonl or data_dir / "ocs_val.jsonl"
    test_path = args.test_jsonl or data_dir / "ocs_test.jsonl"
    val_records = _load_jsonl(val_path, args.max_queries)
    validation_rows_available = len(val_records)
    val_records = stratified_record_subset(
        val_records, args.max_validation_queries, seed=args.validation_subsample_seed,
    )
    validation_was_subsampled = len(val_records) < validation_rows_available
    test_records = _load_jsonl(test_path, args.max_queries)
    if icl_support is not None:
        support_ids = [node for node, _ in icl_support]
        labels = np.asarray(processed.y, dtype=np.int8)
        validate_icl_query_records(
            val_records, support_node_ids=support_ids, labels=labels,
            allowed_mask=np.asarray(processed.val_mask, dtype=bool), split_name="validation",
        )
        validate_icl_query_records(
            test_records, support_node_ids=support_ids, labels=labels,
            allowed_mask=np.asarray(processed.test_mask, dtype=bool), split_name="test",
        )
    val_y, val_p = _score_split(
        val_records, model, tokenizer, embeddings, device,
        args.output_dir / "validation_predictions.jsonl", max_length=args.max_length,
        icl_support=icl_support, raw_texts=raw_texts,
        icl_support_records=icl_records if args.icl_support_graphs else None,
        icl_support_graphs=args.icl_support_graphs,
        icl_support_max_tokens=args.icl_support_max_tokens,
        query_max_tokens=args.query_max_tokens,
    )
    threshold = select_f1_threshold(val_y, val_p)
    test_y, test_p = _score_split(
        test_records, model, tokenizer, embeddings, device,
        args.output_dir / "test_predictions.jsonl", max_length=args.max_length,
        icl_support=icl_support, raw_texts=raw_texts,
        icl_support_records=icl_records if args.icl_support_graphs else None,
        icl_support_graphs=args.icl_support_graphs,
        icl_support_max_tokens=args.icl_support_max_tokens,
        query_max_tokens=args.query_max_tokens,
    )
    metadata = getattr(processed, "metadata", {})
    payload = {
        "dataset": args.dataset,
        "relation": metadata.get("relation"),
        "protocol": (
            "static_transductive_in_context_few_shot" if args.icl_support_jsonl else
            "static_transductive_projector_few_shot" if args.support_manifest else
            "static_transductive_zero_shot"
        ),
        "gnn_training_uses_yelp": False,
        "projector_source": "arxiv",
        "projector_training_uses_yelp": bool(args.support_manifest and not args.icl_support_jsonl),
        "threshold_source": "validation",
        "icl": {
            "enabled": bool(args.icl_support_jsonl),
            "support_labels_used_in_prompt": bool(args.icl_support_jsonl),
            "support_graphs_used_in_prompt": bool(args.icl_support_graphs),
            "support_text_max_tokens": args.icl_support_max_tokens if args.icl_support_jsonl else None,
            "query_text_max_tokens": args.query_max_tokens,
            "llm_frozen": True,
            "projector_frozen": True,
            "gnn_frozen": True,
        },
        "input_policy": {
            "max_context_tokens": args.max_length,
            "query_text_max_tokens": args.query_max_tokens,
            "support_text_max_tokens": args.icl_support_max_tokens if args.icl_support_jsonl else None,
            "default_text_policy": "full_text_no_truncation",
            "overflow_policy": "raise_error_no_silent_truncation",
        },
        "validation_context_path": str(val_path.resolve()),
        "test_context_path": str(test_path.resolve()),
        "validation": binary_metrics(val_y, val_p, threshold=threshold),
        "test": binary_metrics(test_y, test_p, threshold=threshold),
        "counts": {"validation": int(val_y.size), "test": int(test_y.size)},
        "validation_sampling": {
            "method": "deterministic_stratified_sha256" if validation_was_subsampled else "all",
            "seed": args.validation_subsample_seed if validation_was_subsampled else None,
            "available": validation_rows_available,
            "requested_max": args.max_validation_queries,
            "used": int(val_y.size),
        },
        "review_id_hash": metadata.get("review_id_hash"),
        "mask_hash": metadata.get("mask_hash"),
        "graph": {
            "nodes": metadata.get("nodes"),
            "directed_edges": metadata.get("directed_edges"),
            "max_degree": metadata.get("max_degree"),
            "isolated_nodes": metadata.get("isolated_nodes"),
            "max_neighbors": metadata.get("max_neighbors"),
        },
        "embedding_sha256": _file_hash(embedding_path),
        "checkpoint": str(args.model_path.resolve()),
        "model_base": str(Path(args.model_base).resolve()),
    }
    if args.support_manifest:
        support_payload = json.loads(args.support_manifest.read_text(encoding="utf-8"))
        payload["support"] = {
            "manifest_path": str(args.support_manifest.resolve()),
            "shots_per_class": support_payload.get("shots_per_class"),
            "seed": support_payload.get("seed"),
            "support_ids": support_payload.get("support_ids"),
            "support_hash": support_payload.get("support_hash"),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probability_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
