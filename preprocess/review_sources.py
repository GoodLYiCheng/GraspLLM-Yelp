from __future__ import annotations

import hashlib
import heapq
import json
import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator


SCHEMA_VERSION = "yelp_amazon_review_v1"
STANDARD_FIELDS = (
    "review_id",
    "reviewer_id",
    "product_id",
    "category",
    "label",
    "review_text",
    "summary",
    "unix_review_time",
    "source_ordinal",
)


@dataclass(frozen=True)
class AmazonSource:
    slug: str
    filename: str
    category: str
    dataset_prefix: str


AMAZON_SOURCES = (
    AmazonSource(
        "cellphones",
        "Cell_Phones_and_Accessories.json",
        "Cell_Phones_and_Accessories",
        "amazon_cellphones",
    ),
    AmazonSource(
        "clothing",
        "Clothing_Shoes_and_Jewelry.json",
        "Clothing_Shoes_and_Jewelry",
        "amazon_clothing",
    ),
    AmazonSource("electronics", "Electronics.json", "Electronics", "amazon_electronics"),
    AmazonSource("home", "Home_and_Kitchen.json", "Home_and_Kitchen", "amazon_home"),
    AmazonSource(
        "sports", "Sports_and_Outdoors.json", "Sports_and_Outdoors", "amazon_sports"
    ),
    AmazonSource("toys", "Toys_and_Games.json", "Toys_and_Games", "amazon_toys"),
)


class HashingReader:
    """Binary reader wrapper that hashes bytes consumed by TextIOWrapper."""

    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def iter_concatenated_json(
    path: Path, *, chunk_size: int = 1 << 20, max_object_bytes: int = 64 << 20
) -> tuple[Iterator[dict], HashingReader]:
    """Stream whitespace-separated or directly concatenated JSON objects.

    The Amazon files are not JSON arrays and are not reliably JSONL.  This
    decoder accepts both ``{...}\n{...}`` and ``{...}{...}`` without loading a
    multi-gigabyte source into memory.
    """

    raw = path.open("rb")
    hashing_reader = HashingReader(raw)

    def _iterator() -> Iterator[dict]:
        decoder = json.JSONDecoder()
        utf8_decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        eof = False
        try:
            while True:
                if not eof:
                    block = hashing_reader.read(chunk_size)
                    if block:
                        buffer += utf8_decoder.decode(block, final=False)
                    else:
                        eof = True
                        buffer += utf8_decoder.decode(b"", final=True)
                position = 0
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position >= len(buffer):
                        buffer = ""
                        break
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError as error:
                        if eof:
                            raise ValueError(
                                f"malformed trailing JSON in {path}: {error}"
                            ) from error
                        buffer = buffer[position:]
                        if len(buffer.encode("utf-8")) > max_object_bytes:
                            raise ValueError(
                                f"JSON object exceeds {max_object_bytes} bytes in {path}"
                            ) from error
                        break
                    if not isinstance(value, dict):
                        raise ValueError(f"expected JSON object in {path}, got {type(value)}")
                    position = end
                    yield value
                if eof:
                    if buffer.strip():
                        raise ValueError(f"unparsed trailing content in {path}")
                    return
        finally:
            raw.close()

    return _iterator(), hashing_reader


def _review_id(record: dict) -> str:
    value = record.get("_id")
    if isinstance(value, dict):
        value = value.get("$oid")
    if value not in (None, ""):
        return str(value)
    fallback = "\0".join(
        str(record.get(key, ""))
        for key in ("reviewerID", "asin", "unixReviewTime", "reviewText")
    )
    return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def normalize_amazon_record(record: dict, *, ordinal: int, expected_category: str) -> dict:
    required = ("reviewerID", "asin", "class", "reviewText")
    missing = [name for name in required if record.get(name) is None]
    if missing:
        raise ValueError(f"Amazon record {ordinal} missing required fields: {missing}")
    label_value = float(record["class"])
    if label_value not in (0.0, 1.0):
        raise ValueError(f"Amazon record {ordinal} has invalid class={record['class']!r}")
    category = str(record.get("category") or expected_category)
    if category != expected_category:
        raise ValueError(
            f"Amazon record {ordinal} category={category!r}, expected={expected_category!r}"
        )
    return {
        "review_id": _review_id(record),
        "reviewer_id": str(record["reviewerID"]),
        "product_id": str(record["asin"]),
        "category": category,
        "label": int(label_value),
        "review_text": str(record.get("reviewText") or record.get("summary") or ""),
        "summary": str(record.get("summary") or ""),
        "unix_review_time": int(record.get("unixReviewTime") or 0),
        "source_ordinal": int(ordinal),
    }


def _selection_score(seed: int, category: str, review_id: str) -> int:
    raw = f"{seed}\0{category}\0{review_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")


def sample_amazon_source(
    source: AmazonSource,
    path: Path,
    *,
    per_class: int,
    seed: int,
    chunk_size: int = 1 << 20,
) -> tuple[list[dict], dict]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    iterator, hashing_reader = iter_concatenated_json(path, chunk_size=chunk_size)
    heaps: dict[int, list[tuple[int, str]]] = {0: [], 1: []}
    selected: dict[int, dict[str, tuple[int, dict]]] = {0: {}, 1: {}}
    source_counts = {0: 0, 1: 0}
    total = 0
    for ordinal, raw_record in enumerate(iterator):
        record = normalize_amazon_record(
            raw_record, ordinal=ordinal, expected_category=source.category
        )
        total += 1
        label = int(record["label"])
        source_counts[label] += 1
        review_id = str(record["review_id"])
        if review_id in selected[label]:
            continue
        score = _selection_score(seed, source.category, review_id)
        heap = heaps[label]
        bucket = selected[label]
        while heap and (
            heap[0][1] not in bucket or bucket[heap[0][1]][0] != -heap[0][0]
        ):
            heapq.heappop(heap)
        if len(bucket) < per_class:
            bucket[review_id] = (score, record)
            heapq.heappush(heap, (-score, review_id))
            continue
        worst_score = -heap[0][0]
        if score >= worst_score:
            continue
        _, worst_id = heapq.heappop(heap)
        del bucket[worst_id]
        bucket[review_id] = (score, record)
        heapq.heappush(heap, (-score, review_id))

    if any(len(selected[label]) != per_class for label in (0, 1)):
        raise ValueError(
            f"{source.filename} lacks requested balanced sample: "
            f"available={source_counts}, selected={{0: {len(selected[0])}, 1: {len(selected[1])}}}"
        )
    records = [entry[1] for label in (0, 1) for entry in selected[label].values()]
    records.sort(key=lambda item: int(item["source_ordinal"]))
    ids = [str(record["review_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate selected review IDs in {source.filename}")
    stats = {
        "source_file": source.filename,
        "source_sha256": hashing_reader.digest.hexdigest(),
        "source_records": total,
        "source_class_counts": {str(key): value for key, value in source_counts.items()},
        "selected_records": len(records),
        "selected_class_counts": {
            str(label): sum(int(record["label"]) == label for record in records)
            for label in (0, 1)
        },
    }
    return records, stats


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if tuple(record.keys()) != STANDARD_FIELDS:
                raise ValueError(f"record schema mismatch: {tuple(record.keys())}")
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return {
        "path": str(path),
        "records": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
