from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.builder import load_pretrained_model
from utils.constants import DEFAULT_GRAPH_PAD_ID
from utils.utils import get_model_name_from_path

from .artifacts import file_sha256, write_json
from .prompts import matched_text_prompt, pack_prompt
from .provenance import validate_projector_provenance
from .scoring import binary_answer_probability
from .text import FALLBACK_LENGTHS, tokenizer_hash


METHODS = ("text_only_matched", "text_only_maxcontext", "random_graph_llm", "full_graspllm")


def _load_contexts(path: Path) -> dict[int, list[int]]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            result[int(record["node_index"])] = list(map(int, record["nodes"]))
    return result


def _load_support(path: Path | None, k: int) -> list[int]:
    if k == 0:
        return []
    if path is None:
        raise ValueError("--support-manifest is required when K > 0")
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record["k_per_class"]) != k:
        raise ValueError("support manifest K mismatch")
    nodes = [int(value["node_index"]) for value in record["records"]]
    if len(nodes) != 2 * k:
        raise ValueError("support manifest must contain exactly 2K nodes")
    return nodes


def _stratified_subset(indices, years, labels, limit: int | None, seed: int):
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or len(indices) <= limit:
        return indices.tolist()
    rng = np.random.default_rng(seed)
    strata = {}
    for index in indices:
        strata.setdefault((int(years[index]), int(labels[index])), []).append(int(index))
    selected = []
    total = sum(len(values) for values in strata.values())
    remaining = limit
    items = sorted(strata.items())
    for offset, (_, values) in enumerate(items):
        if offset == len(items) - 1:
            count = remaining
        else:
            count = min(len(values), round(limit * len(values) / total))
            remaining -= count
        selected.extend(rng.choice(values, size=count, replace=False).tolist())
    return sorted(selected)


def _graph_inputs(rows, embeddings, device, dtype):
    graph = torch.tensor(rows, dtype=torch.long)
    mask = graph != DEFAULT_GRAPH_PAD_ID
    graph_emb = torch.zeros((*graph.shape, embeddings.shape[1]), dtype=torch.float32)
    graph_emb[mask] = embeddings[graph[mask]]
    return {"graph": graph.to(device), "graph_emb": graph_emb.to(device=device, dtype=dtype)}


def _load_model(args, graph_method: bool):
    projector_provenance = None
    if graph_method:
        projector_provenance = validate_projector_provenance(args.model_path, args.projector_provenance)
        name = get_model_name_from_path(str(args.model_path))
        tokenizer, model, _ = load_pretrained_model(
            str(args.model_path), args.model_base, name,
            cache_dir=str(args.model_path / "_figraph_eval_cache"),
            attn_implementation=None if args.attn_implementation == "auto" else args.attn_implementation,
            torch_dtype=torch.float16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        load_kwargs = {}
        if args.attn_implementation != "auto":
            load_kwargs["attn_implementation"] = args.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(
            args.model_base, torch_dtype=dtype, low_cpu_mem_usage=True,
            **load_kwargs,
        )
    model = model.to(args.device).eval()
    model_type = str(getattr(model.config, "model_type", "")).lower()
    if "qwen3" not in model_type and "qwen3" not in str(args.model_base).lower():
        raise ValueError(f"strict protocol requires Qwen3-8B, got model_type={model_type!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, projector_provenance


def _token_hash(ids) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _evaluate_once(args, tokenizer, model, cohort, nodes, support_nodes, contexts, embeddings, max_length, output_path):
    texts = cohort["raw_texts"]
    labels = cohort["labels"]
    years = cohort["years"]
    support_texts = [texts[index] for index in support_nodes]
    support_labels = [int(labels[index]) for index in support_nodes]
    graph_method = args.method in {"full_graspllm", "random_graph_llm"}
    with output_path.open("w", encoding="utf-8") as output:
        for node in tqdm(nodes, desc=f"{args.method}@{max_length}", disable=args.no_progress):
            if graph_method:
                rows = [contexts[index] for index in [*support_nodes, node]]
                packed = pack_prompt(
                    tokenizer, support_texts, support_labels, texts[node],
                    graph_rows=rows, max_length=max_length,
                )
                model_kwargs = _graph_inputs(rows, embeddings, torch.device(args.device), next(model.parameters()).dtype)
            elif args.method == "text_only_matched":
                # Reserve exactly 32 projected positions per document without
                # reading any graph artifact, then remove markers while keeping
                # the selected source-document token IDs unchanged.
                dummy = [[DEFAULT_GRAPH_PAD_ID] * 32 for _ in range(2 * args.k + 1)]
                full_budget = pack_prompt(
                    tokenizer, support_texts, support_labels, texts[node],
                    graph_rows=dummy, max_length=max_length,
                )
                packed = matched_text_prompt(tokenizer, full_budget, support_labels)
                model_kwargs = {}
            else:
                packed = pack_prompt(
                    tokenizer, support_texts, support_labels, texts[node],
                    graph_rows=None, max_length=max_length,
                )
                model_kwargs = {}
            probability = binary_answer_probability(model, tokenizer, packed.prompt_ids, model_kwargs=model_kwargs)
            truncation = [len(ids) < len(tokenizer(text, add_special_tokens=False).input_ids) for ids, text in zip(packed.document_token_ids, [*support_texts, texts[node]])]
            output.write(json.dumps({
                "node_index": int(node),
                "node_key": cohort["node_keys"][node],
                "year": int(years[node]),
                "ground_truth": int(labels[node]),
                "fraud_probability": probability,
                "support_node_indices": support_nodes,
                "context_length": max_length,
                "prompt_tokens": packed.prompt_tokens,
                "expanded_graph_tokens": packed.expanded_graph_tokens,
                "effective_total": packed.effective_total,
                "document_used_tokens": list(map(len, packed.document_token_ids)),
                "document_truncated": truncation,
                "document_segments": packed.document_segments,
                "document_token_hashes": [_token_hash(ids) for ids in packed.document_token_ids],
                "raw_text_hashes": [hashlib.sha256(str(text).encode("utf-8")).hexdigest() for text in [*support_texts, texts[node]]],
                "enable_thinking": packed.enable_thinking,
            }, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict-transfer FiGraph Qwen3 likelihood evaluation")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--text-cohort", required=True, type=Path)
    parser.add_argument("--model-base", default=os.environ.get("FIGRAPH_BASE_MODEL", "/data/Qwen/Qwen3-8B"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--projector-provenance", type=Path)
    parser.add_argument("--contexts", type=Path)
    parser.add_argument("--graph-embedding", type=Path)
    parser.add_argument("--support-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", type=int, choices=(0, 1, 5, 10), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-sample-seed", type=int, default=42,
                        help="Fixed cohort-subsampling seed; keep constant across support seeds")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-length", type=int, choices=FALLBACK_LENGTHS, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    graph_method = args.method in {"full_graspllm", "random_graph_llm"}
    if graph_method and not all((args.model_path, args.projector_provenance, args.contexts, args.graph_embedding)):
        parser.error("graph methods require --model-path --projector-provenance --contexts --graph-embedding")
    if not graph_method and any((args.contexts, args.graph_embedding, args.model_path, args.projector_provenance)):
        parser.error("text-only methods reject all graph/GNN/projector arguments")
    cohort = torch.load(args.text_cohort, map_location="cpu", weights_only=False)
    if cohort.get("metadata", {}).get("graph_free") is not True:
        raise ValueError("--text-cohort must be the graph-free artifact")
    support_nodes = _load_support(args.support_manifest, args.k)
    mask = cohort["val_mask"] if args.split == "validation" else cohort["test_mask"]
    base_nodes = torch.where(mask)[0].tolist()
    nodes = _stratified_subset(
        base_nodes, cohort["years"], cohort["labels"], args.max_rows, args.query_sample_seed
    )
    contexts = _load_contexts(args.contexts) if graph_method else None
    embeddings = None
    if graph_method:
        value = torch.load(args.graph_embedding, map_location="cpu", weights_only=False)
        embeddings = (value["emb"] if isinstance(value, dict) else value).float()
        missing = set([*support_nodes, *nodes]) - set(contexts)
        if missing:
            raise ValueError(f"context file lacks {len(missing)} required nodes")
    tokenizer, model, projector_provenance = _load_model(args, graph_method)
    lengths = [value for value in FALLBACK_LENGTHS if value <= args.max_length]
    attempts = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    actual = None
    for max_length in lengths:
        try:
            _evaluate_once(args, tokenizer, model, cohort, nodes, support_nodes, contexts, embeddings, max_length, temp)
            actual = max_length
            temp.replace(args.output)
            break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"context_length": max_length, "error": type(error).__name__})
            if temp.exists():
                temp.unlink()
            gc.collect()
            torch.cuda.empty_cache()
    if actual is None:
        raise RuntimeError(f"all context lengths exhausted after OOM: {attempts}")
    audit_rows = []
    with args.output.open("r", encoding="utf-8") as handle:
        audit_rows = [json.loads(line) for line in handle if line.strip()]
    document_total = sum(len(row["document_truncated"]) for row in audit_rows)
    truncated_total = sum(sum(map(bool, row["document_truncated"])) for row in audit_rows)
    model_config = {
        key: getattr(model.config, key, None)
        for key in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "max_position_embeddings")
    }
    runtime_device = torch.device(args.device)
    cuda_runtime = None
    if runtime_device.type == "cuda":
        cuda_runtime = {
            "device_name": torch.cuda.get_device_name(runtime_device),
            "compute_capability": list(torch.cuda.get_device_capability(runtime_device)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(runtime_device).total_memory),
        }
    manifest = {
        "status": "COMPLETED", "method": args.method, "split": args.split,
        "k_per_class": args.k, "seed": args.seed, "rows": len(nodes),
        "query_sample_seed": args.query_sample_seed,
        "requested_context_length": args.max_length, "actual_context_length": actual,
        "oom_fallback_attempts": attempts, "enable_thinking": False,
        "canonical_answers": ["Fraud", "Normal"], "score": "length-normalized sequence likelihood",
        "parameters_frozen": True,
        "prompt_audit": {
            "documents": document_total,
            "truncated_documents": truncated_total,
            "truncation_rate": truncated_total / max(1, document_total),
            "maximum_effective_total": max((row["effective_total"] for row in audit_rows), default=0),
            "source_token_hashes_saved": True,
        },
        "text_cohort_sha256": file_sha256(args.text_cohort),
        "tokenizer_hash": tokenizer_hash(tokenizer), "model_base": args.model_base,
        "model_config": model_config,
        "model_dtype": str(next(model.parameters()).dtype),
        "cuda_runtime": cuda_runtime,
        "attention_backend": args.attn_implementation,
        "projector_provenance": projector_provenance,
        "graph_embedding_sha256": file_sha256(args.graph_embedding) if graph_method else None,
        "contexts_sha256": file_sha256(args.contexts) if graph_method else None,
        "graph_assets_loaded": graph_method, "output": str(args.output.resolve()),
    }
    write_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
