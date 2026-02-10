from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CompiledChainData:
    npz_path: str
    meta_path: str
    version: int
    max_height: int
    tip_height: int
    first_block_2019: int
    built_at_utc: str
    heights: np.ndarray
    time_unix_s: np.ndarray
    subsidy_zat: np.ndarray
    fees_zat: np.ndarray
    tx_count: np.ndarray
    issued_supply_end_zat: np.ndarray

    def to_blocks_df(self) -> pd.DataFrame:
        # One-time materialization for existing model functions in viz/app.py.
        t = pd.to_datetime(self.time_unix_s, unit="s", utc=True)
        return pd.DataFrame(
            {
                "height": self.heights.astype(np.int64, copy=False),
                "time_utc": t,
                "subsidy_zat": self.subsidy_zat.astype(np.int64, copy=False),
                "fees_zat": self.fees_zat.astype(np.int64, copy=False),
                "tx_count": self.tx_count.astype(np.int64, copy=False),
                "issued_supply_end_zat": self.issued_supply_end_zat.astype(np.int64, copy=False),
            }
        )


def _read_meta(meta_path: str) -> Dict[str, Any]:
    if not meta_path or not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj
    return {}


def compiled_chain_exists(npz_path: str) -> bool:
    return bool(npz_path and os.path.exists(npz_path))


def load_compiled_chain(npz_path: str, meta_path: Optional[str] = None) -> CompiledChainData:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as z:
        required = (
            "heights",
            "time_unix_s",
            "subsidy_zat",
            "fees_zat",
            "tx_count",
            "issued_supply_end_zat",
        )
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError(f"compiled npz missing keys: {missing}")

        heights = np.asarray(z["heights"], dtype=np.int64)
        time_unix_s = np.asarray(z["time_unix_s"], dtype=np.int64)
        subsidy_zat = np.asarray(z["subsidy_zat"], dtype=np.int64)
        fees_zat = np.asarray(z["fees_zat"], dtype=np.int64)
        tx_count = np.asarray(z["tx_count"], dtype=np.int64)
        issued_supply_end_zat = np.asarray(z["issued_supply_end_zat"], dtype=np.int64)

    n = len(heights)
    if n == 0:
        raise ValueError("compiled chain arrays are empty")
    for name, arr in (
        ("time_unix_s", time_unix_s),
        ("subsidy_zat", subsidy_zat),
        ("fees_zat", fees_zat),
        ("tx_count", tx_count),
        ("issued_supply_end_zat", issued_supply_end_zat),
    ):
        if len(arr) != n:
            raise ValueError(f"compiled array length mismatch for {name}: {len(arr)} != {n}")

    meta = _read_meta(meta_path or "")
    version = int(meta.get("version", 1))
    tip_height = int(meta.get("tip_height", int(heights[-1])))
    first_block_2019 = int(meta.get("first_block_2019", 0))
    built_at_utc = str(meta.get("built_at_utc", ""))
    resolved_meta_path = meta_path or ""

    return CompiledChainData(
        npz_path=npz_path,
        meta_path=resolved_meta_path,
        version=version,
        max_height=int(heights[-1]),
        tip_height=tip_height,
        first_block_2019=first_block_2019,
        built_at_utc=built_at_utc,
        heights=heights,
        time_unix_s=time_unix_s,
        subsidy_zat=subsidy_zat,
        fees_zat=fees_zat,
        tx_count=tx_count,
        issued_supply_end_zat=issued_supply_end_zat,
    )
