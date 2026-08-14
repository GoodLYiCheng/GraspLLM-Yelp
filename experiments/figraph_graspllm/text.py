from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_MAX_LENGTH = 32768
FALLBACK_LENGTHS = (32768, 24576, 16384)
HMT_RATIOS = (0.40, 0.20, 0.40)


@dataclass(frozen=True)
class TokenView:
    token_ids: list[int]
    original_tokens: int
    used_tokens: int
    truncated: bool
    segments: tuple[tuple[int, int], ...]


def tokenizer_hash(tokenizer) -> str:
    payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "model_max_length": int(getattr(tokenizer, "model_max_length", -1)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _allocation(budget: int, ratios: Sequence[float]) -> tuple[int, ...]:
    exact = np.asarray(ratios, dtype=np.float64) * budget
    values = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - values), kind="stable")[: budget - int(values.sum())]:
        values[index] += 1
    return tuple(map(int, values))


def head_middle_tail(
    token_ids: Sequence[int],
    budget: int,
    *,
    ratios: Sequence[float] = HMT_RATIOS,
) -> TokenView:
    values = list(map(int, token_ids))
    if budget <= 0:
        raise ValueError("budget must be positive")
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("head/middle/tail ratios must contain three non-negative values summing to 1")
    if len(values) <= budget:
        return TokenView(values, len(values), len(values), False, ((0, len(values)),))

    head_n, middle_n, tail_n = _allocation(budget, ratios)
    head_end = head_n
    tail_start = len(values) - tail_n
    gap_start, gap_end = head_end, tail_start
    middle_start = max(gap_start, (len(values) - middle_n) // 2)
    middle_start = min(middle_start, gap_end - middle_n)
    middle_end = middle_start + middle_n
    segments = ((0, head_end), (middle_start, middle_end), (tail_start, len(values)))
    selected: list[int] = []
    for start, end in segments:
        selected.extend(values[start:end])
    if len(selected) != budget:
        raise RuntimeError(f"HMT selected {len(selected)} tokens, expected {budget}")
    return TokenView(selected, len(values), len(selected), True, segments)


def special_token_count(tokenizer) -> int:
    return len(tokenizer.build_inputs_with_special_tokens([]))


def token_view_for_text(tokenizer, text: str, *, max_length: int) -> TokenView:
    raw_ids = tokenizer(str(text), add_special_tokens=False).input_ids
    budget = max_length - special_token_count(tokenizer)
    if budget <= 0:
        raise ValueError("max_length leaves no room after special tokens")
    return head_middle_tail(raw_ids, budget)


def model_inputs_from_view(tokenizer, view: TokenView, *, device=None) -> dict:
    encoded = tokenizer.prepare_for_model(
        view.token_ids,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[-1] != view.used_tokens + special_token_count(tokenizer):
        raise RuntimeError("tokenizer special-token accounting changed unexpectedly")
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded


def token_length_statistics(
    tokenizer,
    texts: Iterable[str],
    years: Iterable[int],
    present: Iterable[bool],
) -> dict[str, object]:
    grouped: dict[int, list[int]] = {}
    missing: dict[int, int] = {}
    for text, year, is_present in zip(texts, years, present):
        year = int(year)
        if not bool(is_present):
            missing[year] = missing.get(year, 0) + 1
            continue
        length = len(tokenizer(str(text), add_special_tokens=True).input_ids)
        grouped.setdefault(year, []).append(length)

    result = {}
    for year, raw_values in sorted(grouped.items()):
        values = np.asarray(raw_values, dtype=np.int64)
        result[str(year)] = {
            "present_rows": int(values.size),
            "missing_rows": int(missing.get(year, 0)),
            "median": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": int(values.max()),
            "over_16k_rate": float((values > 16384).mean()),
            "over_24k_rate": float((values > 24576).mean()),
            "over_32k_rate": float((values > 32768).mean()),
        }
    return {
        "tokenizer_hash": tokenizer_hash(tokenizer),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "")),
        "years": result,
    }


def model_file_hashes(model_path: str | Path) -> dict[str, str | None]:
    root = Path(model_path)
    result = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        path = root / name
        if not path.is_file():
            result[name] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        result[name] = digest.hexdigest()
    return result
