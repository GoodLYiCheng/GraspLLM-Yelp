from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.builder import load_pretrained_model
from utils.chat_format import build_multi_turn_eval_prompt, install_chat_template
from utils.constants import DEFAULT_GRAPH_PAD_ID
from utils.utils import get_model_name_from_path

from .scoring import binary_answer_probability


def _load_embedding(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return (obj["emb"] if isinstance(obj, dict) else obj).float()


def _graph_inputs(record: dict, embeddings: torch.Tensor, device: torch.device):
    specs = record.get("graphs", [])
    if not specs:
        return {}
    node_lists = [list(map(int, spec["nodes"])) for spec in specs]
    if len({len(nodes) for nodes in node_lists}) != 1:
        raise ValueError("all graph groups must have the same padded width")
    graph = torch.tensor(node_lists, dtype=torch.long)
    mask = graph != DEFAULT_GRAPH_PAD_ID
    graph_emb = torch.zeros((*graph.shape, embeddings.shape[1]), dtype=torch.float32)
    graph_emb[mask] = embeddings[graph[mask]]
    return {
        "graph": graph.to(device),
        "graph_emb": graph_emb.to(device=device, dtype=torch.float16),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Fraudulent vs Legitimate by sequence likelihood")
    parser.add_argument("--model-path")
    parser.add_argument(
        "--model-base",
        default=os.environ.get("GRASP_DUAL_BASE_MODEL", "/data/Qwen/Qwen3-8B"),
    )
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--graph-embedding", type=Path)
    parser.add_argument("--answers-file", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.text_only:
        model_name = get_model_name_from_path(args.model_base)
        tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        if not args.model_path or not args.graph_embedding:
            parser.error("--model-path and --graph-embedding are required unless --text-only is set")
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, _ = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            cache_dir=str(Path(args.model_path) / "_eval_cache"),
        )
    install_chat_template(tokenizer, model_name_or_path=args.model_base)
    model = model.to(device=device, dtype=torch.float16).eval()
    embeddings = _load_embedding(args.graph_embedding) if args.graph_embedding else None
    with args.data_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if args.max_queries is not None:
        records = records[: args.max_queries]
    args.answers_file.parent.mkdir(parents=True, exist_ok=True)
    with args.answers_file.open("w", encoding="utf-8") as output:
        iterator = tqdm(records, desc="llm-score", disable=args.no_progress)
        for record in iterator:
            prompt_ids = build_multi_turn_eval_prompt(
                tokenizer,
                record["conversations"],
                has_graph=bool(record.get("graphs")),
                max_length=args.max_length,
            )
            if args.text_only and record.get("graphs"):
                raise ValueError("--text-only requires records without graph groups")
            probability = binary_answer_probability(
                model,
                tokenizer,
                prompt_ids,
                model_kwargs={} if embeddings is None else _graph_inputs(record, embeddings, device),
            )
            output.write(json.dumps({
                "id": record["id"],
                "node_id": record["node_id"],
                "timestamp": record["timestamp"],
                "ground_truth": int(record["ground_truth"]),
                "fraud_probability": probability,
                "support_node_ids": record.get("support_node_ids", []),
            }) + "\n")
    print(json.dumps({"status": "COMPLETED", "answers": str(args.answers_file), "rows": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
