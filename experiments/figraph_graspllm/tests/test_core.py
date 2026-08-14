from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.figraph_graspllm.contexts import select_context
from experiments.figraph_graspllm.encode_text import shard_node_indices
from experiments.figraph_graspllm.evaluation import annual_report, select_validation_threshold
from experiments.figraph_graspllm.gate import paired_comparison
from experiments.figraph_graspllm.motifs import compute_motifs
from experiments.figraph_graspllm.merge_embeddings import merge_shards
from experiments.figraph_graspllm.prepare import _read_year
from experiments.figraph_graspllm.prompts import matched_text_prompt, pack_prompt
from experiments.figraph_graspllm.provenance import validate_projector_provenance
from experiments.figraph_graspllm.support import sample_support
from experiments.figraph_graspllm.text import head_middle_tail
from utils.constants import DEFAULT_GRAPH_PAD_ID


class _Encoded:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class CharTokenizer:
    def __init__(self):
        self.thinking_values = []

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        values = [ord(char) for char in str(text)]
        if return_tensors == "pt":
            values = torch.tensor([values], dtype=torch.long)
        return _Encoded(values)

    def decode(self, ids, **kwargs):
        return "".join(chr(int(value)) for value in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict=False, **kwargs):
        self.thinking_values.append(enable_thinking)
        text = "".join(f"<{row['role']}>{row['content']}" for row in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        values = [ord(char) for char in text]
        return {"input_ids": values} if return_dict else values


def test_hmt_is_exact_and_deterministic():
    full = head_middle_tail(list(range(20)), 20)
    assert full.token_ids == list(range(20)) and not full.truncated
    first = head_middle_tail(list(range(100)), 10)
    second = head_middle_tail(list(range(100)), 10)
    assert first == second
    assert first.token_ids[:4] == [0, 1, 2, 3]
    assert first.token_ids[-4:] == [96, 97, 98, 99]
    assert len(first.token_ids) == 10


def test_embedding_shards_are_disjoint_and_merge_in_node_order(tmp_path: Path):
    assignments = [shard_node_indices(10, num_shards=4, shard_id=index) for index in range(4)]
    assert sorted(node for values in assignments for node in values) == list(range(10))
    paths = []
    for shard_id, indices in enumerate(assignments):
        path = tmp_path / f"shard{shard_id}.pt"
        metadata = {
            "encoder": "Qwen3-Embedding-8B",
            "tokenizer_hash": "tokenizer",
            "processed_data_sha256": "data",
            "requested_max_length": 16384,
            "final_max_length": 16384,
            "dtype": "torch.float16",
            "attention_backend": "sdpa",
            "num_nodes": 10,
            "num_shards": 4,
            "shard_id": shard_id,
        }
        torch.save({
            "emb": torch.tensor([[node, node + 0.5] for node in indices], dtype=torch.float16),
            "node_indices": torch.tensor(indices),
            "metadata": metadata,
            "token_records": [{"node_id": node} for node in indices],
        }, path)
        paths.append(path)
    output = tmp_path / "merged.pt"
    result = merge_shards(paths, output)
    merged = torch.load(output, map_location="cpu", weights_only=False)
    assert result["context_length"] == 16384
    assert merged["emb"][:, 0].tolist() == list(range(10))
    assert [row["node_id"] for row in merged["token_records"]] == list(range(10))


def test_embedding_merge_rejects_mixed_context_lengths(tmp_path: Path):
    paths = []
    for shard_id, node in enumerate((0, 1)):
        path = tmp_path / f"bad{shard_id}.pt"
        torch.save({
            "emb": torch.ones((1, 2), dtype=torch.float16),
            "node_indices": torch.tensor([node]),
            "metadata": {
                "encoder": "Qwen3-Embedding-8B", "tokenizer_hash": "t",
                "processed_data_sha256": "d", "requested_max_length": 16384,
                "final_max_length": 16384 if shard_id == 0 else 24576,
                "dtype": "torch.float16", "attention_backend": "sdpa",
                "num_nodes": 2, "num_shards": 2, "shard_id": shard_id,
            },
            "token_records": [{"node_id": node}],
        }, path)
        paths.append(path)
    with pytest.raises(ValueError, match="final_max_length"):
        merge_shards(paths, tmp_path / "should_not_exist.pt")


def test_prompt_32k_invariants_and_matched_source_tokens():
    tokenizer = CharTokenizer()
    support = ["a" * 1200, "b" * 900]
    rows = [[DEFAULT_GRAPH_PAD_ID] * 32 for _ in range(3)]
    packed = pack_prompt(tokenizer, support, [0, 1], "q" * 4000, graph_rows=rows, max_length=4096)
    matched = matched_text_prompt(tokenizer, packed, [0, 1])
    assert packed.effective_total <= 4096
    assert packed.conversations[-1]["from"] == "human"
    assert packed.document_token_ids == matched.document_token_ids
    assert packed.expanded_graph_tokens == 96
    assert matched.expanded_graph_tokens == 0
    assert tokenizer.thinking_values and set(tokenizer.thinking_values) == {False}


def test_support_is_balanced_and_reproducible():
    data = SimpleNamespace(
        support_mask=torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool),
        y=torch.tensor([0, 0, 1, 1, 0]),
    )
    first = sample_support(data, k=2, seed=42)
    assert first == sample_support(data, k=2, seed=42)
    assert [int(data.y[index]) for index in first] == [0, 1, 0, 1]


def test_context_keeps_center_and_pads():
    text = torch.eye(4)
    structure = torch.eye(4)
    values = select_context(0, [1, 2], text, structure, method="ocs", seed=42, max_nodes=4)
    assert values[0] == 0
    assert values[-1] == DEFAULT_GRAPH_PAD_ID
    assert len(values) == 4
    assert select_context(0, [1, 2], text, structure, method="random", seed=7, max_nodes=4) == select_context(0, [1, 2], text, structure, method="random", seed=7, max_nodes=4)


def test_exact_motif_counts_on_k4():
    edges = np.asarray([(u, v) for u in range(4) for v in range(u + 1, 4)])
    result = compute_motifs(edges, 4, mode="exact_edge_membership")
    assert result.audit["triangle_instances"] == 4
    assert result.audit["four_cycle_instances"] == 3
    assert result.audit["four_clique_instances"] == 1
    assert result.channels["4-clique"].shape[1] == 12


def test_headerless_edge_first_row_and_alignment(tmp_path: Path):
    year_dir = tmp_path / "2019"
    year_dir.mkdir()
    pd.DataFrame({"nodeID": ["a", "b", "c"], "Year": [2019] * 3, "Label": [0, 1, 0]}).to_csv(year_dir / "ListedCompanyFeatures772_2019.csv", index=False)
    pd.DataFrame({"nodeID": ["a", "b", "c"], "Year": [2019] * 3, "Label": [0, 1, 0], "ManaDiscAnal": ["x", "y", None]}).to_excel(year_dir / "MDA_2019.xlsx", index=False)
    (year_dir / "edges2019.csv").write_text("a,b,r1\nb,c,r2\na,z,background\na,a,self\n", encoding="utf-8")
    _, _, present, pairs, audit = _read_year(tmp_path, 2019)
    assert pairs.values.tolist() == [["a", "b"], ["b", "c"]]
    assert audit["raw_edge_rows"] == 4
    assert present.tolist() == [True, True, False]


def test_projector_requires_hash_and_non_figraph_sources(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    projector = model / "mm_projector.bin"
    projector.write_bytes(b"frozen")
    import hashlib

    digest = hashlib.sha256(b"frozen").hexdigest()
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"source_datasets": ["arxiv"], "training_uses_figraph": False, "mm_projector_sha256": digest}), encoding="utf-8")
    assert validate_projector_provenance(model, provenance)["source_datasets"] == ["arxiv"]
    record = json.loads(provenance.read_text())
    record["source_datasets"] = ["FiGraph"]
    provenance.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate_projector_provenance(model, provenance)


def test_metrics_and_stratified_paired_bootstrap():
    validation = select_validation_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < validation["threshold"] <= 0.8
    report = annual_report([2021] * 4 + [2022] * 4, [0, 0, 1, 1] * 2, [0.1, 0.2, 0.8, 0.9] * 2, validation["threshold"])
    assert report["macro_pr_auc"] == 1.0
    full, base = {}, {}
    for year in (2021, 2022):
        for index, label in enumerate([0, 0, 1, 1]):
            key = f"{year}:{index}"
            common = {"node_key": key, "year": year, "ground_truth": label}
            full[key] = {**common, "fraud_probability": [0.1, 0.2, 0.8, 0.9][index]}
            base[key] = {**common, "fraud_probability": [0.4, 0.6, 0.5, 0.7][index]}
    comparison = paired_comparison(full, base, iterations=100, seed=1)
    assert comparison["delta_macro_pr_auc"] > 0
