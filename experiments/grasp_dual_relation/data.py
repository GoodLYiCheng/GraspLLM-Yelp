from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReviewEventTable:
    """Canonical YelpZip rows in deterministic (timestamp, review_id) order."""

    review_ids: np.ndarray
    user_ids: np.ndarray
    business_ids: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray
    texts: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def node_ids(self) -> np.ndarray:
        return np.arange(len(self), dtype=np.int64)

    def node_order_hash(self) -> str:
        digest = hashlib.sha256()
        for review_id, timestamp in zip(self.review_ids, self.timestamps):
            digest.update(str(review_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(int(timestamp)).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()


def load_yelpzip(path: str | Path, *, max_rows: int | None = None) -> ReviewEventTable:
    import pandas as pd

    path = Path(path)
    columns = ["Unnamed: 0", "user_id", "prod_id", "date", "label", "text"]
    frame = pd.read_csv(path, usecols=columns, nrows=max_rows, encoding="utf-8")
    if frame[columns].isna().any().any():
        missing = frame[columns].isna().sum()
        raise ValueError(f"YelpZip contains missing required values: {missing[missing > 0].to_dict()}")
    if frame["Unnamed: 0"].duplicated().any():
        raise ValueError("YelpZip review IDs must be unique")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values(["date", "Unnamed: 0"], kind="stable").reset_index(drop=True)
    raw_labels = set(frame["label"].astype(int).unique().tolist())
    if not raw_labels.issubset({-1, 1}):
        raise ValueError(f"unexpected YelpZip labels: {sorted(raw_labels)}")

    timestamps = frame["date"].astype("int64").to_numpy(dtype=np.int64, copy=True)
    return ReviewEventTable(
        review_ids=frame["Unnamed: 0"].astype(str).to_numpy(copy=True),
        user_ids=frame["user_id"].astype(str).to_numpy(copy=True),
        business_ids=frame["prod_id"].astype(str).to_numpy(copy=True),
        timestamps=timestamps,
        labels=(frame["label"].to_numpy(dtype=np.int8, copy=True) == -1).astype(np.int8),
        texts=tuple(frame["text"].astype(str).tolist()),
    )

