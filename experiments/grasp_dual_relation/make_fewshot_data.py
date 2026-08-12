from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fewshot import load_jsonl, make_fewshot_query, select_support_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict K-shot projector/evaluation JSONL files")
    parser.add_argument("--alignment-contexts", required=True, type=Path)
    parser.add_argument("--query-contexts", required=True, type=Path)
    parser.add_argument("--support-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shots", type=int, choices=[1, 5, 10], required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    alignment = load_jsonl(args.alignment_contexts)
    queries = load_jsonl(args.query_contexts)
    support_payload = json.loads(args.support_ids.read_text(encoding="utf-8"))
    support_nodes = support_payload[str(args.shots)][str(args.seed)]
    support = select_support_records(alignment, support_nodes)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"k{args.shots}_seed{args.seed}_train.jsonl"
    eval_path = args.output_dir / f"k{args.shots}_seed{args.seed}_eval.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for record in support:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(make_fewshot_query(support, query), ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "train_path": str(train_path),
        "train_rows": len(support),
        "eval_path": str(eval_path),
        "eval_rows": len(queries),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

