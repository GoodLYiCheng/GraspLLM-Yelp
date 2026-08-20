from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocess.review_sources import (
    AMAZON_SOURCES,
    SCHEMA_VERSION,
    STANDARD_FIELDS,
    sample_amazon_source,
    sha256_file,
    write_jsonl,
)


BUNDLE_NAME = "yelp_amazon_pretrain_raw_v1"


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _copy_yelp(source_root: Path, raw_root: Path, *, expected_rows: int) -> dict:
    import pandas as pd

    source = source_root / "Yelp-Dataset" / "yelpzip.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    target = raw_root / "Yelp-Dataset" / "yelpzip.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    row_count = 0
    label_counts = {-1: 0, 1: 0}
    required = ["Unnamed: 0", "user_id", "prod_id", "label", "text"]
    for chunk in pd.read_csv(source, usecols=required, chunksize=100000, encoding="utf-8"):
        if chunk[required].isna().any().any():
            raise ValueError(f"{source} contains missing required Yelp values")
        labels = chunk["label"].astype(int)
        unknown = set(labels.unique()) - {-1, 1}
        if unknown:
            raise ValueError(f"{source} contains unexpected Yelp labels: {sorted(unknown)}")
        row_count += len(chunk)
        for label in (-1, 1):
            label_counts[label] += int((labels == label).sum())
    if row_count != expected_rows:
        raise ValueError(f"{source} has {row_count} rows, expected {expected_rows}")
    return {
        "path": "raw_data/Yelp-Dataset/yelpzip.csv",
        "records": row_count,
        "class_counts": {str(key): value for key, value in label_counts.items()},
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "label_mapping": {"-1": "Fraudulent", "1": "Legitimate"},
    }


def _write_cloud_readme(path: Path) -> None:
    path.write_text(
        """# Yelp + Amazon pretraining raw bundle

Reassemble, verify and extract from the directory containing `parts/`:

```bash
cat parts/yelp_amazon_pretrain_raw_v1.tar.zst.part-* \\
  > yelp_amazon_pretrain_raw_v1.tar.zst
sha256sum -c SHA256SUMS
zstd -dc yelp_amazon_pretrain_raw_v1.tar.zst | tar -xf -
```

After extraction set:

```bash
export RAW_REVIEW_ROOT=/absolute/path/yelp_amazon_pretrain_raw_v1/raw_data
```
""",
        encoding="utf-8",
    )


def _split_file(source: Path, parts_dir: Path, part_size: int) -> list[Path]:
    parts_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as handle:
        index = 0
        while True:
            block = handle.read(part_size)
            if not block:
                break
            part = parts_dir / f"{source.name}.part-{index:03d}"
            with part.open("wb") as output:
                output.write(block)
            parts.append(part)
            index += 1
    if not parts:
        raise RuntimeError(f"archive is empty: {source}")
    return parts


def _write_sha256s(path: Path, rows: list[str]) -> None:
    # GNU sha256sum treats a trailing CR as part of the filename, so this
    # machine-readable file must remain LF-only even when built on Windows.
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")


def _package(bundle_dir: Path, output_root: Path, *, part_size: int, zstd_level: int) -> dict:
    tar_path = output_root / f"{BUNDLE_NAME}.tar"
    archive_path = output_root / f"{BUNDLE_NAME}.tar.zst"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(bundle_dir, arcname=BUNDLE_NAME)
    zstd = shutil.which("zstd")
    if not zstd:
        raise FileNotFoundError("zstd executable is required to package the bundle")
    subprocess.run(
        [zstd, f"-{zstd_level}", "-T0", "-f", str(tar_path), "-o", str(archive_path)],
        check=True,
    )
    tar_path.unlink()
    archive_sha = sha256_file(archive_path)
    archive_size = archive_path.stat().st_size
    parts = _split_file(archive_path, output_root / "parts", part_size)
    part_rows = [
        {
            "path": f"parts/{part.name}",
            "bytes": part.stat().st_size,
            "sha256": sha256_file(part),
        }
        for part in parts
    ]
    archive_path.unlink()
    return {
        "archive": {
            "path": archive_path.name,
            "bytes": archive_size,
            "sha256": archive_sha,
        },
        "parts": part_rows,
    }


def build(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    bundle_dir = output_root / BUNDLE_NAME
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to rebuild")
        shutil.rmtree(output_root)
    raw_root = bundle_dir / "raw_data"
    raw_root.mkdir(parents=True)
    files = {
        "yelp": _copy_yelp(
            source_root, raw_root, expected_rows=args.expected_yelp_rows
        ),
        "amazon": {},
    }
    all_selected_ids: dict[str, str] = {}
    cross_category_duplicates = []
    for source in AMAZON_SOURCES:
        source_path = source_root / "amazon" / source.filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        print(f"[sample] {source.filename}", flush=True)
        records, stats = sample_amazon_source(
            source,
            source_path,
            per_class=args.amazon_per_category // 2,
            seed=args.seed,
            chunk_size=args.chunk_size_mib << 20,
        )
        output = raw_root / "amazon" / f"amazon_{source.slug}_100k.jsonl"
        output_stats = write_jsonl(output, records)
        for record in records:
            review_id = str(record["review_id"])
            previous = all_selected_ids.setdefault(review_id, source.slug)
            if previous != source.slug:
                cross_category_duplicates.append(
                    {"review_id": review_id, "first": previous, "second": source.slug}
                )
        files["amazon"][source.slug] = {
            **stats,
            **output_stats,
            "path": f"raw_data/amazon/{output.name}",
            "category": source.category,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle": BUNDLE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo),
        "source_root": str(source_root),
        "sampling": {
            "method": "class_stratified_stable_hash_bottom_k",
            "seed": args.seed,
            "amazon_per_category": args.amazon_per_category,
            "amazon_per_class": args.amazon_per_category // 2,
        },
        "amazon_schema": list(STANDARD_FIELDS),
        "excluded_amazon_files": ["part.json", "separate.json"],
        "cross_category_duplicate_review_ids": len(cross_category_duplicates),
        "cross_category_duplicate_examples": cross_category_duplicates[:100],
        "files": files,
    }
    data_manifest = bundle_dir / "DATA_MANIFEST.json"
    data_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_cloud_readme(bundle_dir / "CLOUD_UPLOAD_README.md")
    package = _package(
        bundle_dir,
        output_root,
        part_size=args.part_size_gib * (1 << 30),
        zstd_level=args.zstd_level,
    )
    shutil.rmtree(bundle_dir)
    final_manifest = {**manifest, "package": package}
    (output_root / "MANIFEST.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_cloud_readme(output_root / "CLOUD_UPLOAD_README.md")
    checksum_rows = [
        f"{package['archive']['sha256']}  {package['archive']['path']}",
        *[f"{part['sha256']}  {part['path']}" for part in package["parts"]],
    ]
    _write_sha256s(output_root / "SHA256SUMS", checksum_rows)
    print(json.dumps({"status": "COMPLETED", "output": str(output_root), **package}, indent=2))
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build uploadable Yelp/Amazon raw bundle")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--amazon-per-category", type=int, default=100000)
    parser.add_argument("--expected-yelp-rows", type=int, default=608458)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size-mib", type=int, default=4)
    parser.add_argument("--part-size-gib", type=int, default=2)
    parser.add_argument("--zstd-level", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.amazon_per_category <= 0 or args.amazon_per_category % 2:
        parser.error("--amazon-per-category must be a positive even integer")
    return args


if __name__ == "__main__":
    build(parse_args())
