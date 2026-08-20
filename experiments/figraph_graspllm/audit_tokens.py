from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .artifacts import write_json
from .text import token_length_statistics


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit FiGraph MDA token lengths with Qwen3 tokenizer")
    parser.add_argument("--processed-data", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = torch.load(args.processed_data, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, padding_side="left")
    payload = token_length_statistics(tokenizer, data.raw_texts, data.years, data.mda_present)
    payload["processed_data"] = str(args.processed_data.resolve())
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
