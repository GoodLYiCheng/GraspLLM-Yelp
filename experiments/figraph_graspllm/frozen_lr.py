from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression

from .artifacts import file_sha256, write_json
from .provenance import validate_gnn_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen MotifGNN embedding + balanced LR baseline")
    parser.add_argument("--text-cohort", required=True, type=Path)
    parser.add_argument("--structure-embedding", required=True, type=Path)
    parser.add_argument("--gnn-checkpoint", required=True, type=Path)
    parser.add_argument("--support-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    provenance = validate_gnn_checkpoint(args.gnn_checkpoint)
    cohort = torch.load(args.text_cohort, map_location="cpu", weights_only=False)
    support = json.loads(args.support_manifest.read_text(encoding="utf-8"))
    support_nodes = [int(row["node_index"]) for row in support["records"]]
    value = torch.load(args.structure_embedding, map_location="cpu", weights_only=False)
    embeddings = (value["emb"] if isinstance(value, dict) else value).float().numpy()
    labels = cohort["labels"].numpy()
    classifier = LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=args.seed, max_iter=2000,
    )
    classifier.fit(embeddings[support_nodes], labels[support_nodes])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, mask in (("validation", cohort["val_mask"]), ("test", cohort["test_mask"])):
        nodes = torch.where(mask)[0].tolist()
        probabilities = classifier.predict_proba(embeddings[nodes])[:, 1]
        path = args.output_dir / f"frozen_motifgnn_lr_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for node, probability in zip(nodes, probabilities):
                handle.write(json.dumps({
                    "node_index": node, "node_key": cohort["node_keys"][node],
                    "year": int(cohort["years"][node]), "ground_truth": int(labels[node]),
                    "fraud_probability": float(probability), "support_node_indices": support_nodes,
                }) + "\n")
        outputs[split] = str(path.resolve())
    manifest = {
        "status": "COMPLETED", "method": "frozen_motifgnn_lr", "seed": args.seed,
        "k_per_class": int(support["k_per_class"]), "parameters_frozen": True,
        "gnn": provenance, "structure_embedding_sha256": file_sha256(args.structure_embedding),
        "outputs": outputs,
    }
    write_json(args.output_dir / "frozen_motifgnn_lr_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
