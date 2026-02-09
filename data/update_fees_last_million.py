#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests


# Zcash consensus constants (Mainnet)
SLOW_START_INTERVAL = 20000
SLOW_START_SHIFT = SLOW_START_INTERVAL // 2
PRE_BLOSSOM_HALVING_INTERVAL = 840000
MAX_BLOCK_SUBSIDY_ZAT = 1_250_000_000
BLOSSOM_ACTIVATION_HEIGHT = 653600
PRE_BLOSSOM_POW_TARGET_SPACING = 150
POST_BLOSSOM_POW_TARGET_SPACING = 75
BLOSSOM_POW_TARGET_SPACING_RATIO = PRE_BLOSSOM_POW_TARGET_SPACING / POST_BLOSSOM_POW_TARGET_SPACING
POST_BLOSSOM_HALVING_INTERVAL = math.floor(PRE_BLOSSOM_HALVING_INTERVAL * BLOSSOM_POW_TARGET_SPACING_RATIO)


def halving(height: int) -> int:
    if height < SLOW_START_SHIFT:
        return 0
    if height < BLOSSOM_ACTIVATION_HEIGHT:
        return (height - SLOW_START_SHIFT) // PRE_BLOSSOM_HALVING_INTERVAL
    return int(
        (BLOSSOM_ACTIVATION_HEIGHT - SLOW_START_SHIFT) / PRE_BLOSSOM_HALVING_INTERVAL
        + (height - BLOSSOM_ACTIVATION_HEIGHT) / POST_BLOSSOM_HALVING_INTERVAL
    )


def block_subsidy_zat(height: int) -> int:
    slow_start_rate = MAX_BLOCK_SUBSIDY_ZAT / SLOW_START_INTERVAL
    if height < SLOW_START_SHIFT:
        return int(slow_start_rate * height)
    if height < SLOW_START_INTERVAL:
        return int(slow_start_rate * (height + 1))
    h = halving(height)
    if height < BLOSSOM_ACTIVATION_HEIGHT:
        return int(MAX_BLOCK_SUBSIDY_ZAT // (2**h))
    return int(MAX_BLOCK_SUBSIDY_ZAT // (BLOSSOM_POW_TARGET_SPACING_RATIO * (2**h)))


def _to_zat(explicit_zat, zec_value) -> int:
    if explicit_zat is not None:
        return int(explicit_zat)
    if zec_value is None:
        return 0
    return int(round(float(zec_value) * 100_000_000))


class RpcClient:
    def __init__(self, url: str, user: str = "", password: str = "", timeout_sec: int = 60):
        self.url = url
        self.timeout_sec = int(timeout_sec)
        self.session = requests.Session()
        self.auth = (user, password) if user or password else None
        self._rpc_id = 0

    def call(self, method: str, params=None):
        self._rpc_id += 1
        payload = {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params or []}
        r = self.session.post(
            self.url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            auth=self.auth,
            timeout=self.timeout_sec,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("error") is not None:
            raise RuntimeError(j["error"])
        return j["result"]


class FeeResolver:
    def __init__(self, rpc: RpcClient):
        self.rpc = rpc
        self.getblockstats_supported: Optional[bool] = None

    def fee_for_height(self, height: int, block_hash: Optional[str]) -> int:
        if self.getblockstats_supported is not False:
            try:
                stats = self.rpc.call("getblockstats", [height])
                self.getblockstats_supported = True
                if isinstance(stats, dict) and stats.get("totalfee") is not None:
                    return int(stats["totalfee"])
            except Exception as e:
                msg = str(e).lower()
                if "method not found" in msg or "-32601" in msg:
                    self.getblockstats_supported = False
                else:
                    raise
        return self._fee_from_coinbase_fallback(height, block_hash)

    def _fee_from_coinbase_fallback(self, height: int, block_hash: Optional[str]) -> int:
        if not block_hash:
            block_hash = self.rpc.call("getblockhash", [height])

        block = self.rpc.call("getblock", [block_hash, 2])
        txs = block.get("tx", [])
        if not txs:
            return 0

        coinbase = txs[0]
        coinbase_total_zat = 0
        for out in coinbase.get("vout", []):
            coinbase_total_zat += _to_zat(out.get("valueZat"), out.get("value"))

        expected_no_fee_zat = None
        try:
            sub = self.rpc.call("getblocksubsidy", [height])
            miner_zat = _to_zat(sub.get("minerZat"), sub.get("miner"))
            fs_total_zat = _to_zat(sub.get("fundingstreamstotalZat"), sub.get("fundingstreamstotal"))
            # lockboxtotal is not paid through coinbase vouts, so do not subtract it here.
            expected_no_fee_zat = miner_zat + fs_total_zat
        except Exception:
            expected_no_fee_zat = block_subsidy_zat(height)

        return max(0, int(coinbase_total_zat - expected_no_fee_zat))


def detect_max_height(csv_path: Path) -> int:
    max_height = -1
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                h = int(row["height"])
                if h > max_height:
                    max_height = h
            except Exception:
                continue
    return max_height


def update_fees(
    csv_path: Path,
    rpc: RpcClient,
    start_height: int,
    end_height: int,
    backup: bool = True,
):
    resolver = FeeResolver(rpc)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")

    changed = 0
    scanned = 0
    errors = 0
    t0 = time.time()
    updated_rows_total = 0

    last_log_time = t0
    last_log_scanned = 0
    last_log_updated = 0

    with csv_path.open("r", newline="", encoding="utf-8") as src, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        r = csv.DictReader(src)
        if r.fieldnames is None:
            raise RuntimeError("CSV has no header.")
        fieldnames = r.fieldnames
        if "height" not in fieldnames or "fees_zat" not in fieldnames:
            raise RuntimeError("CSV must contain height and fees_zat columns.")
        w = csv.DictWriter(dst, fieldnames=fieldnames)
        w.writeheader()

        for row in r:
            scanned += 1
            try:
                h = int(row["height"])
            except Exception:
                w.writerow(row)
                continue

            if start_height <= h <= end_height:
                updated_rows_total += 1
                old_fee = row.get("fees_zat", "")
                block_hash = row.get("block_hash")
                try:
                    new_fee = resolver.fee_for_height(h, block_hash)
                    if str(new_fee) != str(old_fee):
                        changed += 1
                    row["fees_zat"] = str(new_fee)
                except Exception as e:
                    errors += 1
                    if errors <= 20:
                        print(f"[WARN] h={h}: fee update failed, keeping old value ({e})", file=sys.stderr)
            w.writerow(row)

            if scanned % 10000 == 0:
                now = time.time()
                dt_total = max(0.001, now - t0)
                dt_segment = max(0.001, now - last_log_time)
                scanned_segment = scanned - last_log_scanned
                updated_segment = updated_rows_total - last_log_updated
                print(
                    "[INFO] "
                    f"scanned={scanned} changed={changed} errors={errors} "
                    f"rate_total~{scanned / dt_total:.1f} rows/s "
                    f"rate_segment~{scanned_segment / dt_segment:.1f} rows/s "
                    f"in_range_total={updated_rows_total} "
                    f"in_range_segment={updated_segment} "
                    f"in_range_rate_segment~{updated_segment / dt_segment:.1f} rows/s",
                    file=sys.stderr,
                )
                last_log_time = now
                last_log_scanned = scanned
                last_log_updated = updated_rows_total

    if backup:
        if backup_path.exists():
            backup_path = csv_path.with_suffix(csv_path.suffix + f".bak.{int(time.time())}")
        os.replace(csv_path, backup_path)
        os.replace(tmp_path, csv_path)
        print(f"[OK] backup saved: {backup_path}")
    else:
        os.replace(tmp_path, csv_path)

    print(f"[OK] fees_zat updated in range [{start_height}, {end_height}]")
    print(f"[OK] scanned={scanned}, changed={changed}, errors={errors}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir / "blocks_full_with_types.csv"

    p = argparse.ArgumentParser(
        description="Update only fees_zat in blocks_full_with_types.csv for recent heights."
    )
    p.add_argument("--csv", default=str(default_csv), help="Path to blocks_full_with_types.csv")
    p.add_argument("--rpc-url", default=os.getenv("ZEBRA_RPC_URL", "http://127.0.0.1:8232"))
    p.add_argument("--rpc-user", default=os.getenv("ZEBRA_RPC_USER", ""))
    p.add_argument("--rpc-pass", default=os.getenv("ZEBRA_RPC_PASS", ""))
    p.add_argument("--rpc-timeout-sec", type=int, default=60)
    p.add_argument("--last-blocks", type=int, default=1_000_000)
    p.add_argument("--start-height", type=int, default=None)
    p.add_argument("--end-height", type=int, default=None)
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    max_height = detect_max_height(csv_path)
    if max_height < 0:
        raise RuntimeError("Could not detect max height from CSV.")

    start_height = (
        int(args.start_height)
        if args.start_height is not None
        else max(0, max_height - int(args.last_blocks) + 1)
    )
    end_height = int(args.end_height) if args.end_height is not None else max_height

    if start_height > end_height:
        raise ValueError(f"start-height {start_height} > end-height {end_height}")

    rpc = RpcClient(
        url=args.rpc_url,
        user=args.rpc_user,
        password=args.rpc_pass,
        timeout_sec=args.rpc_timeout_sec,
    )

    info = rpc.call("getblockchaininfo")
    chain = str(info.get("chain", "unknown"))
    if chain != "main":
        print(f"[WARN] RPC chain={chain}. Script assumes mainnet constants.", file=sys.stderr)

    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] max_height_csv={max_height}")
    print(f"[INFO] update_range=[{start_height}, {end_height}]")
    print(f"[INFO] rpc={args.rpc_url}")

    update_fees(
        csv_path=csv_path,
        rpc=rpc,
        start_height=start_height,
        end_height=end_height,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
