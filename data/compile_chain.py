#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "height",
    "time_utc",
    "subsidy_zat",
    "fees_zat",
    "tx_count",
    "issued_supply_end_zat",
)


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def compile_chain(input_csv: str, out_npz: str, out_meta: str) -> None:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(input_csv)

    print(f"[INFO] reading csv: {input_csv}")
    df = pd.read_csv(input_csv, usecols=list(REQUIRED_COLUMNS))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("height").drop_duplicates(subset=["height"], keep="last")

    max_h = int(df["height"].max()) if not df.empty else 0
    n = max_h + 1
    heights = np.arange(0, n, dtype=np.int64)

    # Prepare "known rows" index.
    h_idx = df["height"].to_numpy(dtype=np.int64)
    valid_mask = (h_idx >= 0) & (h_idx <= max_h)
    h_idx = h_idx[valid_mask]
    df = df.iloc[np.flatnonzero(valid_mask)].copy()

    subsidy = np.zeros(n, dtype=np.int64)
    fees = np.zeros(n, dtype=np.int64)
    tx_count = np.zeros(n, dtype=np.int64)
    issued = np.zeros(n, dtype=np.int64)
    time_unix = np.full(n, -1, dtype=np.int64)
    subsidy_known = np.zeros(n, dtype=bool)
    issued_known = np.zeros(n, dtype=bool)

    subsidy_vals = pd.to_numeric(df["subsidy_zat"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    fees_vals = pd.to_numeric(df["fees_zat"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    tx_vals = pd.to_numeric(df["tx_count"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    issued_vals = pd.to_numeric(df["issued_supply_end_zat"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    tvals = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")

    subsidy[h_idx] = subsidy_vals
    fees[h_idx] = fees_vals
    tx_count[h_idx] = tx_vals
    issued[h_idx] = issued_vals
    subsidy_known[h_idx] = True
    issued_known[h_idx] = True

    tvals_s = tvals.to_numpy(dtype="datetime64[s]")
    time_known = ~np.isnat(tvals_s)
    if np.any(time_known):
        hk = h_idx[time_known]
        time_unix[hk] = tvals_s[time_known].astype(np.int64)

    # Fill gaps for factual series similarly to runtime expectations.
    subsidy_series = pd.Series(np.where(subsidy_known, subsidy, np.nan), dtype="float64").ffill().fillna(0)
    issued_series = pd.Series(np.where(issued_known, issued, np.nan), dtype="float64").ffill().fillna(0)
    subsidy = subsidy_series.astype(np.int64).to_numpy()
    issued = issued_series.astype(np.int64).to_numpy()

    tser = pd.Series(time_unix, dtype="float64").replace(-1, np.nan).ffill().bfill()
    if tser.isna().all():
        now = int(datetime.now(timezone.utc).timestamp())
        tser = pd.Series(now + heights * 75, dtype="float64")
    time_unix = tser.astype(np.int64).to_numpy()

    first_2019 = 0
    ts2019 = int(pd.Timestamp("2019-01-01T00:00:00Z").timestamp())
    idx_2019 = np.flatnonzero(time_unix >= ts2019)
    if len(idx_2019):
        first_2019 = int(idx_2019[0])

    _ensure_parent(out_npz)
    _ensure_parent(out_meta)

    np.savez_compressed(
        out_npz,
        heights=heights,
        time_unix_s=time_unix,
        subsidy_zat=subsidy,
        fees_zat=fees,
        tx_count=tx_count,
        issued_supply_end_zat=issued,
    )

    meta = {
        "version": 1,
        "source_csv": input_csv,
        "rows_source": int(len(df)),
        "tip_height": int(max_h),
        "first_block_2019": int(first_2019),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    print(f"[OK] compiled npz: {out_npz}")
    print(f"[OK] compiled meta: {out_meta}")
    print(f"[INFO] tip_height={max_h}, first_block_2019={first_2019}, n={n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile chain CSV into compact NPZ artifacts.")
    parser.add_argument("--input-csv", default="data/blocks_full_with_types.csv")
    parser.add_argument("--out-npz", default="data/compiled/chain_v1.npz")
    parser.add_argument("--out-meta", default="data/compiled/meta_v1.json")
    args = parser.parse_args()
    compile_chain(args.input_csv, args.out_npz, args.out_meta)


if __name__ == "__main__":
    main()
