# viz/model.py
#
# Core modeling utilities for the NSM / Burns Visualizer.
# This module is intentionally "pure": it should not depend on Dash.
#
# It supports:
# - height -> time axis (factual from CSV + extrapolated after tip)
# - status quo subsidy extension after tip (using halving schedule)
# - ZIP234-like smoothing after activation (fixed activation height)
# - issued supply (fact from CSV + extension after tip)
# - burned supply (fee burn + voluntary events + one-time Sprout burn)
#
# Note: we deliberately treat "fact <= tip" as immutable (pulled from CSV).
# Future projections are scenario-based and should be clearly labeled in the UI.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Future fee model: calibration window near tip.
FUTURE_FEE_ANCHOR_BLOCKS = 200_000

# -----------------------------
# Protocol-ish constants
# -----------------------------
BLOSSOM_HEIGHT = 653600
PRE_BLOSSOM_BLOCK_SEC = 150
POST_BLOSSOM_BLOCK_SEC = 75

# Default heights / intervals that are useful for display markers
SECOND_HALVING_HEIGHT_DEFAULT = 2726400
POST_BLOSSOM_HALVING_INTERVAL = 1680000


# -----------------------------
# Time axis: factual + extrapolated
# -----------------------------
def extend_time_axis(blocks_df: pd.DataFrame, end_height: int) -> pd.Series:
    """
    Returns a pd.Series of UTC timestamps indexed by height 0..end_height.
    Uses factual time_utc where available; extrapolates after tip using 75s/block.
    Also forward-fills holes if CSV is partial/incomplete.
    """
    if end_height < 0:
        raise ValueError("end_height must be >= 0")

    # Keep only the columns we need, coerce safely
    df = blocks_df[["height", "time_utc"]].copy()
    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).sort_values("height")

    idx = pd.RangeIndex(0, end_height + 1, 1, name="height")

    # If CSV is empty or still building, create synthetic axis
    if df.empty:
        now = pd.Timestamp.now(tz="UTC")
        return pd.Series(now + pd.to_timedelta(idx * POST_BLOSSOM_BLOCK_SEC, unit="s"), index=idx)

    tip_h = int(df["height"].iloc[-1])
    tip_t = df["time_utc"].iloc[-1]

    out = pd.Series(index=idx, dtype="datetime64[ns, UTC]")
    factual = df.set_index("height")["time_utc"]
    factual = factual[(factual.index >= 0) & (factual.index <= end_height)]
    out.loc[factual.index] = factual

    # Extrapolate after tip (we're post-Blossom at current chain tip)
    if end_height > tip_h:
        future_heights = pd.RangeIndex(tip_h + 1, end_height + 1, 1)
        secs = (future_heights - tip_h) * POST_BLOSSOM_BLOCK_SEC
        out.loc[future_heights] = tip_t + pd.to_timedelta(secs, unit="s")

    # Fill holes if CSV is partial
    out = out.ffill().bfill()
    return out


# -----------------------------
# Fee / activity projection
# -----------------------------
@dataclass(frozen=True)
class FeeProjection:
    fee_per_tx_zat: int
    tx_per_block: float


def estimate_fee_projection(blocks_df: pd.DataFrame, anchor_blocks: int = FUTURE_FEE_ANCHOR_BLOCKS) -> FeeProjection:
    """
    Estimate fee_per_tx and tx_per_block from a recent window.
    Robust to partial CSV and zeros.
    """
    if blocks_df.empty:
        return FeeProjection(fee_per_tx_zat=0, tx_per_block=1.0)

    df = blocks_df[["height", "fees_zat", "tx_count"]].copy()
    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("height").tail(max(1, int(anchor_blocks)))

    tx = pd.to_numeric(df["tx_count"], errors="coerce").fillna(0).astype(float)
    fees = pd.to_numeric(df["fees_zat"], errors="coerce").fillna(0).astype(float)

    fee_per_tx = fees / tx.replace(0, np.nan)
    fee_per_tx_med = float(np.nanmedian(fee_per_tx.values)) if np.isfinite(np.nanmedian(fee_per_tx.values)) else 0.0
    tx_per_block_med = float(np.nanmedian(tx.values)) if np.isfinite(np.nanmedian(tx.values)) else 1.0

    return FeeProjection(
        fee_per_tx_zat=int(max(0.0, fee_per_tx_med)),
        tx_per_block=max(0.0, tx_per_block_med),
    )


def tx_growth_multiplier(profile: str, t_years: np.ndarray, k: float) -> np.ndarray:
    """
    Compute a multiplier applied to tx_per_block for future modeling.

    profile:
      - flat: 1
      - linear: 1 + k*t
      - exponential: exp(k*t)
      - logistic: saturating curve, L=5, t0=4 years, slope k
    """
    profile = (profile or "flat").lower()
    k = float(k)

    t = np.asarray(t_years, dtype=float)
    t = np.maximum(t, 0.0)

    if profile == "flat":
        return np.ones_like(t)
    if profile == "linear":
        return 1.0 + k * t
    if profile == "exponential":
        return np.exp(k * t)
    if profile == "logistic":
        L = 5.0
        t0 = 4.0
        return 1.0 + (L - 1.0) / (1.0 + np.exp(-k * (t - t0)))

    return np.ones_like(t)


# -----------------------------
# Issuance models
# -----------------------------
def build_status_quo_subsidy(
    heights: np.ndarray,
    blocks_df: pd.DataFrame,
    *,
    second_halving_height: int = SECOND_HALVING_HEIGHT_DEFAULT,
    halving_interval: int = POST_BLOSSOM_HALVING_INTERVAL,
) -> np.ndarray:
    """
    Status quo subsidy curve:

    - For heights <= tip: uses subsidy_zat from CSV (ffill for partial gaps).
    - For heights > tip: continues a halving schedule starting from the subsidy
      value at second_halving_height (or last non-zero fallback).
    """
    heights = np.asarray(heights, dtype=int)
    if heights.size == 0:
        return np.array([], dtype=np.int64)

    df = blocks_df[["height", "subsidy_zat"]].copy()
    if df.empty:
        return np.zeros_like(heights, dtype=np.int64)

    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
    df["subsidy_zat"] = pd.to_numeric(df["subsidy_zat"], errors="coerce").fillna(0).astype(np.int64)
    df = df.sort_values("height")

    tip_h = int(df["height"].max())

    factual = df.set_index("height")["subsidy_zat"]
    factual_full = (
        factual.reindex(pd.RangeIndex(0, tip_h + 1, 1))
        .ffill()
        .fillna(0)
        .astype(np.int64)
    )

    out = np.zeros_like(heights, dtype=np.int64)

    # factual part
    mask_f = heights <= tip_h
    if np.any(mask_f):
        out[mask_f] = factual_full.iloc[heights[mask_f]].values

    # base at second halving or fallback last non-zero
    if 0 <= second_halving_height <= tip_h:
        base = int(factual_full.iloc[second_halving_height])
    else:
        nonzero = factual_full[factual_full > 0]
        base = int(nonzero.iloc[-1]) if len(nonzero) else 0

    # future continuation
    mask_u = heights > tip_h
    if np.any(mask_u) and base > 0:
        h = heights[mask_u]
        shifts = np.maximum(0, (h - second_halving_height) // int(halving_interval)).astype(int)
        out_future = np.array([base >> int(min(s, 63)) for s in shifts], dtype=np.int64)
        out[mask_u] = out_future

    return out


def build_zip234_subsidy_after_activation(
    base_subsidy: np.ndarray,
    issued_supply_end_zat: np.ndarray,
    *,
    max_money_zat: int,
    activation_height: int,
    numerator: int,
    denominator: int,
) -> np.ndarray:
    """
    ZIP234-like smoothing after activation (fixed activation height):

    - Before activation: base_subsidy
    - From activation: subsidy = ceil(reserve * numerator / denominator)
      where reserve starts at (max_money - issued_supply_end_zat at activation-1)
      and decreases by subsidy each block.

    This matches the "fixed deployment" interpretation used in many simulators.
    """
    base_subsidy = np.asarray(base_subsidy, dtype=np.int64)
    issued_supply_end_zat = np.asarray(issued_supply_end_zat, dtype=np.int64)
    out = base_subsidy.copy()

    n = len(out)
    if n == 0:
        return out

    activation_height = int(max(0, min(activation_height, n - 1)))
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("denominator must be > 0")

    # reserve at activation-1
    if activation_height <= 0:
        reserve = int(max_money_zat)
    else:
        reserve = int(max_money_zat) - int(issued_supply_end_zat[activation_height - 1])
    reserve = max(0, reserve)

    for h in range(activation_height, n):
        if reserve <= 0:
            out[h] = 0
            continue
        s = (reserve * numerator + (denominator - 1)) // denominator  # ceil
        s = max(0, int(s))
        out[h] = s
        reserve -= s

    return out


def compute_issued_supply(
    heights: np.ndarray,
    blocks_df: pd.DataFrame,
    subsidy_zat: np.ndarray,
    *,
    max_money_zat: int,
) -> np.ndarray:
    """
    Issued supply (end-of-block cumulative):

    - For heights <= tip: use issued_supply_end_zat from CSV (fact; ffill if partial).
    - For heights > tip: extend by adding subsidy_zat each block (capped by max_money_zat).
    """
    heights = np.asarray(heights, dtype=int)
    subsidy_zat = np.asarray(subsidy_zat, dtype=np.int64)

    if heights.size == 0:
        return np.array([], dtype=np.int64)

    df = blocks_df[["height", "issued_supply_end_zat"]].copy()
    if df.empty:
        # no factual data; purely cumulative from subsidy
        out = np.cumsum(subsidy_zat).astype(np.int64)
        return np.minimum(out, int(max_money_zat))

    df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
    df["issued_supply_end_zat"] = pd.to_numeric(df["issued_supply_end_zat"], errors="coerce").fillna(0).astype(np.int64)
    df = df.sort_values("height")

    tip_h = int(df["height"].max())
    factual = df.set_index("height")["issued_supply_end_zat"]
    factual_full = (
        factual.reindex(pd.RangeIndex(0, tip_h + 1, 1))
        .ffill()
        .fillna(0)
        .astype(np.int64)
    )

    out = np.zeros_like(heights, dtype=np.int64)

    # factual part
    mask_f = heights <= tip_h
    if np.any(mask_f):
        out[mask_f] = factual_full.iloc[heights[mask_f]].values

    # extend to future
    if np.any(heights > tip_h):
        supply = int(factual_full.iloc[-1]) if len(factual_full) else 0
        max_money = int(max_money_zat)
        # we assume heights are 0..end in order; our app passes that
        for h in range(tip_h + 1, len(out)):
            supply = min(max_money, supply + int(subsidy_zat[h]))
            out[h] = supply

    return np.minimum(out, int(max_money_zat))


# -----------------------------
# Burns model
# -----------------------------
def compute_burned_supply(
    *,
    blocks_df: pd.DataFrame,
    time_axis: pd.Series,
    end_height: int,
    enable_fee_burn: bool,
    fee_burn_ratio: float,
    enable_voluntary_burns: bool,
    burn_events_df: Optional[pd.DataFrame],
    enable_sprout_burn: bool,
    sprout_burn_height: int,
    sprout_burn_amount_zat: int,
    enable_future_activity: bool,
    future_profile: str,
    future_k: float,
    cap_tx_per_block: Optional[float],
) -> np.ndarray:
    """
    Burned supply cumulative by height 0..end_height.

    Components:
    - fee burn: ratio * fees_zat on each block, historical from CSV;
      optional future projection if enable_future_activity.
    - voluntary burns: from burn_events_df (expects columns height,burn_zat).
    - one-time Sprout burn at sprout_burn_height.

    Returned: cumulative burned array length end_height+1.
    """
    end_height = int(end_height)
    if end_height < 0:
        raise ValueError("end_height must be >= 0")

    burned_per_block = np.zeros(end_height + 1, dtype=np.int64)

    # --- historical fees/tx
    df = blocks_df[["height", "fees_zat", "tx_count"]].copy()
    if not df.empty:
        df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
        df["fees_zat"] = pd.to_numeric(df["fees_zat"], errors="coerce").fillna(0).astype(np.int64)
        df["tx_count"] = pd.to_numeric(df["tx_count"], errors="coerce").fillna(0).astype(np.int64)
        df = df.sort_values("height")
        tip_h = int(df["height"].max())
    else:
        tip_h = 0

    # fee burn (historical)
    if enable_fee_burn and not df.empty:
        ratio = float(max(0.0, min(1.0, fee_burn_ratio)))
        fees_full = (
            df.set_index("height")["fees_zat"]
            .reindex(pd.RangeIndex(0, min(tip_h, end_height) + 1, 1))
            .fillna(0)
            .astype(np.int64)
        )
        burned_per_block[: len(fees_full)] += (fees_full.values.astype(np.float64) * ratio).astype(np.int64)

    # fee burn (future projection)
    if enable_fee_burn and enable_future_activity and not df.empty and end_height > tip_h:
        proj = estimate_fee_projection(blocks_df, anchor_blocks=FUTURE_FEE_ANCHOR_BLOCKS)
        fee_per_tx = max(0, int(proj.fee_per_tx_zat))
        tx0 = float(proj.tx_per_block)

        t_tip = time_axis.loc[tip_h]
        t_future = time_axis.loc[tip_h + 1 : end_height]
        years = (t_future - t_tip).dt.total_seconds().values / (365.25 * 24 * 3600)
        mult = tx_growth_multiplier(future_profile, years.astype(float), float(future_k))
        tx = tx0 * mult

        if cap_tx_per_block is not None:
            try:
                cap = float(cap_tx_per_block)
                if cap > 0:
                    tx = np.minimum(tx, cap)
            except Exception:
                pass

        fees_future = (tx * fee_per_tx).astype(np.int64)
        ratio = float(max(0.0, min(1.0, fee_burn_ratio)))
        burned_future = (fees_future.astype(np.float64) * ratio).astype(np.int64)
        burned_per_block[tip_h + 1 : end_height + 1] += burned_future

    # voluntary burns
    if enable_voluntary_burns and burn_events_df is not None and not burn_events_df.empty:
        ev = burn_events_df.copy()
        if "height" in ev.columns and "burn_zat" in ev.columns:
            ev["height"] = pd.to_numeric(ev["height"], errors="coerce").fillna(-1).astype(int)
            ev["burn_zat"] = pd.to_numeric(ev["burn_zat"], errors="coerce").fillna(0).astype(np.int64)
            for _, row in ev.iterrows():
                h = int(row["height"])
                if 0 <= h <= end_height:
                    burned_per_block[h] += int(row["burn_zat"])

    # one-time Sprout burn
    if enable_sprout_burn:
        h = int(sprout_burn_height)
        amt = int(max(0, sprout_burn_amount_zat))
        if 0 <= h <= end_height:
            burned_per_block[h] += amt

    # cumulative
    return np.cumsum(burned_per_block).astype(np.int64)


# -----------------------------
# Convenience utilities
# -----------------------------
def circulating_from_issued_and_burned(issued_zat: np.ndarray, burned_zat: np.ndarray) -> np.ndarray:
    """
    Circulating supply in zatoshis = max(issued - burned, 0)
    """
    issued_zat = np.asarray(issued_zat, dtype=np.int64)
    burned_zat = np.asarray(burned_zat, dtype=np.int64)
    return np.maximum(0, issued_zat - burned_zat).astype(np.int64)
