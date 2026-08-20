from __future__ import annotations

import json
import argparse

import torch

from preprocess.review_sources import (
    AmazonSource,
    STANDARD_FIELDS,
    iter_concatenated_json,
    sample_amazon_source,
    write_jsonl,
)
from scripts.build_yelp_amazon_upload_bundle import (
    _split_file,
    _write_cloud_readme,
    _write_sha256s,
)
from preprocess.prepare_review_pretrain import _prepare_amazon
from preprocess.review_sources import AMAZON_SOURCES
from utils.paths import DATASETS, STAGE1_SOURCE_DATASETS


def _record(index: int, label: int, category: str = "Demo") -> dict:
    return {
        "_id": {"$oid": f"id-{index}"},
        "reviewerID": f"user-{index % 3}",
        "asin": f"product-{index % 4}",
        "class": float(label),
        "reviewText": f"review café {index}",
        "summary": f"summary {index}",
        "unixReviewTime": 1000 + index,
        "category": category,
    }


def test_concatenated_json_stream_accepts_direct_and_newline_boundaries(tmp_path):
    path = tmp_path / "source.json"
    values = [_record(0, 0), _record(1, 1), _record(2, 0)]
    path.write_text(
        json.dumps(values[0], ensure_ascii=False)
        + json.dumps(values[1], ensure_ascii=False)
        + "\n"
        + json.dumps(values[2], ensure_ascii=False),
        encoding="utf-8",
    )
    iterator, hashing = iter_concatenated_json(path, chunk_size=7)
    actual = list(iterator)
    assert actual == values
    assert len(hashing.digest.hexdigest()) == 64


def test_balanced_bottom_k_is_deterministic_and_writes_stable_schema(tmp_path):
    source = AmazonSource("demo", "demo.json", "Demo", "amazon_demo")
    path = tmp_path / source.filename
    records = [_record(index, index % 2) for index in range(40)]
    path.write_text("".join(json.dumps(item) for item in records), encoding="utf-8")
    first, first_stats = sample_amazon_source(
        source, path, per_class=5, seed=42, chunk_size=11
    )
    second, second_stats = sample_amazon_source(
        source, path, per_class=5, seed=42, chunk_size=103
    )
    assert first == second
    assert first_stats["source_sha256"] == second_stats["source_sha256"]
    assert [item["source_ordinal"] for item in first] == sorted(
        item["source_ordinal"] for item in first
    )
    assert {label: sum(item["label"] == label for item in first) for label in (0, 1)} == {
        0: 5,
        1: 5,
    }
    assert all(tuple(item) == STANDARD_FIELDS for item in first)
    output = tmp_path / "sample.jsonl"
    info = write_jsonl(output, first)
    assert info["records"] == 10
    assert len(info["sha256"]) == 64


def test_bottom_k_rejects_insufficient_class_rows(tmp_path):
    source = AmazonSource("demo", "demo.json", "Demo", "amazon_demo")
    path = tmp_path / source.filename
    path.write_text("".join(json.dumps(_record(i, 0)) for i in range(10)), encoding="utf-8")
    try:
        sample_amazon_source(source, path, per_class=2, seed=42, chunk_size=13)
    except ValueError as error:
        assert "lacks requested balanced sample" in str(error)
    else:
        raise AssertionError("missing class must fail")


def test_split_parts_reassemble_exact_bytes(tmp_path):
    source = tmp_path / "archive.tar.zst"
    source.write_bytes(bytes(range(251)) * 5)
    parts = _split_file(source, tmp_path / "parts", 200)
    assert len(parts) == 7
    assert b"".join(part.read_bytes() for part in parts) == source.read_bytes()


def test_cloud_readme_contains_executable_reassembly_commands(tmp_path):
    output = tmp_path / "CLOUD_UPLOAD_README.md"
    _write_cloud_readme(output)
    text = output.read_text(encoding="utf-8")
    assert '"Reassemble' not in text
    assert "cat parts/yelp_amazon_pretrain_raw_v1.tar.zst.part-* \\" + "\n" in text
    assert "sha256sum -c SHA256SUMS" in text
    assert "zstd -dc yelp_amazon_pretrain_raw_v1.tar.zst | tar -xf -" in text


def test_sha256s_is_lf_only_for_linux_sha256sum(tmp_path):
    output = tmp_path / "SHA256SUMS"
    _write_sha256s(output, ["a" * 64 + "  archive.tar.zst"])
    assert b"\r" not in output.read_bytes()
    assert output.read_bytes().endswith(b"\n")


def test_prepare_amazon_materializes_twelve_relation_graphs(tmp_path):
    raw_root = tmp_path / "raw"
    dataset_root = tmp_path / "dataset"
    for source in AMAZON_SOURCES:
        records = []
        for index in range(4):
            records.append(
                {
                    "review_id": f"{source.slug}-{index}",
                    "reviewer_id": f"user-{index // 2}",
                    "product_id": f"product-{index % 2}",
                    "category": source.category,
                    "label": index % 2,
                    "review_text": f"text {index}",
                    "summary": "",
                    "unix_review_time": index,
                    "source_ordinal": index,
                }
            )
        write_jsonl(raw_root / "amazon" / f"amazon_{source.slug}_100k.jsonl", records)
    args = argparse.Namespace(
        raw_root=raw_root,
        dataset_root=dataset_root,
        amazon_rows_per_category=4,
        max_neighbors=6,
        seed=42,
        overwrite=False,
    )
    manifests = _prepare_amazon(args)
    assert len(manifests) == 12
    for name, metadata in manifests.items():
        data = torch.load(dataset_root / name / "processed_data.pt", weights_only=False)
        assert int(data.train_mask.sum()) == 4
        assert int(data.val_mask.sum()) == 0
        assert int(data.test_mask.sum()) == 0
        assert not torch.any(data.edge_index[0] == data.edge_index[1])
        assert metadata["uses_labels_for_graph"] is False


def test_review_graphs_are_registered_in_official_dataset_paths():
    expected = {"yelpzip_rur", "yelpzip_rbr"}
    for source in AMAZON_SOURCES:
        expected.update(
            {f"{source.dataset_prefix}_rur", f"{source.dataset_prefix}_rpr"}
        )
    assert expected.issubset(DATASETS)
    assert set(STAGE1_SOURCE_DATASETS) == expected
