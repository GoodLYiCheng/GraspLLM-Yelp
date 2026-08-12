from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from preprocess.build_qwen3_embeddings import encode_texts

from .data import load_yelpzip


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode canonical YelpZip review text with frozen Qwen3")
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("GRASP_DUAL_EMBED_MODEL", "/data/Qwen3-Embedding-8B"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    events = load_yelpzip(args.raw_path, max_rows=args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    model = AutoModel.from_pretrained(args.model_path).to(torch.device(args.device))

    rng = np.random.default_rng(42)
    sample_nodes = rng.choice(len(events), size=min(10000, len(events)), replace=False)
    lengths = [
        len(tokenizer(events.texts[int(node)], add_special_tokens=True, truncation=False).input_ids)
        for node in sample_nodes
    ]
    truncation_rate = float(np.mean(np.asarray(lengths) > args.max_length))
    embeddings = encode_texts(
        model,
        tokenizer,
        list(events.texts),
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=torch.device(args.device),
        out_dtype=torch.float16,
        log_prefix="yelpzip",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "emb": embeddings,
        "node_order_hash": events.node_order_hash(),
        "max_length": args.max_length,
        "truncation_sample_size": len(lengths),
        "truncation_sample_rate": truncation_rate,
        "recommended_max_length": 1024 if truncation_rate > 0.05 else args.max_length,
        "model_path": args.model_path,
    }, args.output)
    print(json.dumps({
        "status": "COMPLETED",
        "output": str(args.output),
        "rows": len(events),
        "dim": int(embeddings.shape[1]),
        "max_length": args.max_length,
        "truncation_sample_rate": truncation_rate,
        "recommended_max_length": 1024 if truncation_rate > 0.05 else args.max_length,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
