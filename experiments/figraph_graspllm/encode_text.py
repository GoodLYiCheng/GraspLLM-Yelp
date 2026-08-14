from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .artifacts import file_sha256
from .text import (
    FALLBACK_LENGTHS,
    model_file_hashes,
    model_inputs_from_view,
    token_view_for_text,
    tokenizer_hash,
)


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden[
        torch.arange(last_hidden.shape[0], device=last_hidden.device), sequence_lengths
    ]


def runtime_policy(device: torch.device) -> tuple[torch.dtype, str]:
    if device.type != "cuda":
        return torch.float32, "sdpa"
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        try:
            import flash_attn  # noqa: F401

            return torch.bfloat16, "flash_attention_2"
        except Exception:
            return torch.bfloat16, "sdpa"
    return torch.float16, "sdpa"


def shard_node_indices(num_nodes: int, *, num_shards: int, shard_id: int) -> list[int]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    return list(range(shard_id, int(num_nodes), num_shards))


def _encode_attempt(
    data,
    tokenizer,
    model_path: Path,
    *,
    max_length: int,
    device: torch.device,
    output_dtype: torch.dtype,
    attention_backend: str,
    show_progress: bool,
    node_indices: list[int],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=output_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
    ).to(device).eval()
    hidden = int(model.config.hidden_size)
    if hidden != 4096:
        raise ValueError(f"strict protocol requires Qwen3-Embedding-8B hidden_size=4096, got {hidden}")
    configured_context = int(getattr(model.config, "max_position_embeddings", max_length))
    if configured_context < max_length:
        raise ValueError(f"encoder config supports {configured_context} positions, requested {max_length}")
    output = torch.zeros((len(node_indices), hidden), dtype=torch.float16)
    records: list[dict[str, object]] = []
    iterator = tqdm(
        enumerate(node_indices), total=len(node_indices),
        desc=f"figraph-embed-{max_length}", disable=not show_progress,
    )
    with torch.inference_mode():
        for output_row, node_id in iterator:
            if not bool(data.mda_present[node_id]):
                records.append({
                    "node_id": node_id,
                    "raw_text_sha256": hashlib.sha256(b"").hexdigest(),
                    "original_tokens": 0,
                    "used_tokens": 0,
                    "truncated": False,
                    "segments": [],
                    "missing_mda": True,
                })
                continue
            view = token_view_for_text(tokenizer, data.raw_texts[node_id], max_length=max_length)
            batch = model_inputs_from_view(tokenizer, view, device=device)
            result = model(**batch)
            pooled = last_token_pool(result.last_hidden_state, batch["attention_mask"])
            output[output_row] = F.normalize(pooled.float(), p=2, dim=-1)[0].half().cpu()
            records.append({
                "node_id": node_id,
                "raw_text_sha256": hashlib.sha256(
                    str(data.raw_texts[node_id]).encode("utf-8")
                ).hexdigest(),
                "original_tokens": view.original_tokens,
                "used_tokens": view.used_tokens,
                "truncated": view.truncated,
                "segments": [list(pair) for pair in view.segments],
                "missing_mda": False,
            })
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, records


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode FiGraph MDA with Qwen3-Embedding-8B")
    parser.add_argument("--processed-data", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=32768, choices=FALLBACK_LENGTHS)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = torch.load(args.processed_data, map_location="cpu", weights_only=False)
    node_indices = shard_node_indices(
        int(data.num_nodes), num_shards=args.num_shards, shard_id=args.shard_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, padding_side="left")
    dtype, attention_backend = runtime_policy(device)
    candidate_lengths = [value for value in FALLBACK_LENGTHS if value <= args.max_length]
    attempts = []
    embedding = records = None
    for length in candidate_lengths:
        try:
            embedding, records = _encode_attempt(
                data,
                tokenizer,
                args.model_path,
                max_length=length,
                device=device,
                output_dtype=dtype,
                attention_backend=attention_backend,
                show_progress=not args.no_progress,
                node_indices=node_indices,
            )
            attempts.append({"max_length": length, "status": "completed"})
            final_length = length
            break
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"max_length": length, "status": "cuda_oom", "error": str(error)})
            gc.collect()
            torch.cuda.empty_cache()
    if embedding is None or records is None:
        raise RuntimeError(f"all embedding context lengths failed: {attempts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "encoder": "Qwen3-Embedding-8B",
        "model_path": str(args.model_path.resolve()),
        "model_file_hashes": model_file_hashes(args.model_path),
        "tokenizer_hash": tokenizer_hash(tokenizer),
        "processed_data": str(args.processed_data.resolve()),
        "processed_data_sha256": file_sha256(args.processed_data),
        "requested_max_length": args.max_length,
        "final_max_length": final_length,
        "oom_attempts": attempts,
        "batch_size": 1,
        "dtype": str(dtype),
        "attention_backend": attention_backend,
        "text_policy": "full_if_fit_else_token_hmt_40_20_40",
        "missing_mda_embedding": "all-zero",
        "truncated_rows": sum(bool(row["truncated"]) for row in records),
        "num_nodes": int(data.num_nodes),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "shard_rows": len(node_indices),
        "node_assignment": "global_node_index_mod_num_shards",
    }
    torch.save({
        "emb": embedding,
        "node_indices": torch.tensor(node_indices, dtype=torch.long),
        "metadata": metadata,
        "token_records": records,
    }, args.output)
    print(json.dumps({"status": "COMPLETED", "output": str(args.output), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
