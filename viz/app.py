# viz/app.py
#
# Local Dash app for Zcash NSM + Burns visualization.
# Run from project root:
#   python -m viz.app
#
# Requirements (in .venv):
#   pip install dash plotly pandas numpy pyyaml

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, ctx

from viz.io import load_blocks_csv, load_burn_events_csv, load_config, get_sprout_snapshot

# -----------------------------
# Protocol-ish constants
# -----------------------------
BLOSSOM_HEIGHT = 653600
PRE_BLOSSOM_BLOCK_SEC = 150
POST_BLOSSOM_BLOCK_SEC = 75

# Halving markers (heights) for annotation.
# We can compute them from second_halving_height + n * 1,680,000 (post-Blossom interval).
# We'll still keep these for labels (and clip to horizon at runtime).
ZATOSHIS_PER_ZEC = 100_000_000
MAX_MONEY_ZAT = 2_100_000_000_000_000
SECOND_HALVING_HEIGHT_DEFAULT = 2726400
POST_BLOSSOM_HALVING_INTERVAL = 1680000
FIRST_HALVING_HEIGHT_DEFAULT = SECOND_HALVING_HEIGHT_DEFAULT - POST_BLOSSOM_HALVING_INTERVAL
THIRD_HALVING_HEIGHT_DEFAULT = SECOND_HALVING_HEIGHT_DEFAULT + POST_BLOSSOM_HALVING_INTERVAL
FUTURE_FEE_ANCHOR_BLOCKS = 200_000
ZIP233_DEPENDENT_TOGGLES = {"fee_burn", "sprout_burn", "voluntary_burns"}
REISSUE_BURNED_TOGGLES = {"reissue_burned"}


# -----------------------------
# Helpers: safe parsing
# -----------------------------
def _to_int(x, default: int) -> int:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(x)
    except Exception:
        return default


def _parse_ratio_input(x, default: float) -> Tuple[float, Optional[str]]:
    """
    Parse ratio input with locale-safe decimal separator support.
    Accepts both "0.6" and "0,6".
    """
    if x is None:
        return float(default), None

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return float(default), None
        if "," in s and "." not in s:
            normalized = s.replace(",", ".")
            try:
                return float(normalized), f"fee ratio parsed from {s} -> {normalized}"
            except Exception:
                return float(default), f"fee ratio parse failed for '{s}', using default {default}"
        try:
            return float(s), None
        except Exception:
            return float(default), f"fee ratio parse failed for '{s}', using default {default}"

    try:
        return float(x), None
    except Exception:
        return float(default), f"fee ratio parse failed, using default {default}"


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _plot_indices(n: int, max_points: int = 20_000) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if n <= max_points:
        return np.arange(n, dtype=int)
    step = int(math.ceil(n / float(max_points)))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _scenario_toggle_options(
    zip233_enabled: bool,
    zip234_enabled: bool,
    fee_burn_enabled: bool,
    burn_source_enabled: bool,
    sprout_burn_enabled: bool,
) -> list[dict]:
    return [
        {"label": "Enable Funds Removal From Circulation (ZIP 233)", "value": "zip233"},
        {"label": "↳ Remove 60% of Transaction Fees (ZIP 235)", "value": "fee_burn", "disabled": not zip233_enabled},
        {"label": "↳ One-time Sprout burn", "value": "sprout_burn", "disabled": not zip233_enabled},
        {"label": "↳ Apply voluntary burns (events_burn.csv)", "value": "voluntary_burns", "disabled": not zip233_enabled},
        {"label": "Apply Issuance Smoothing (ZIP 234)", "value": "zip234"},
        {
            "label": "↳ Reissue Burned Amount in Future Subsidies (ZIP 234)",
            "value": "reissue_burned",
            "disabled": not (zip233_enabled and zip234_enabled and burn_source_enabled),
        },
        {
            "label": "↳ Reissue Sprout Burn in Future Subsidies (ZIP 234)",
            "value": "sprout_reissue",
            "disabled": not (zip233_enabled and zip234_enabled and sprout_burn_enabled),
        },
    ]


def _chart_title(enable_zip233: bool, enable_zip234: bool, enable_zip235: bool) -> str:
    applied = []
    if enable_zip233:
        applied.append("ZIP 233")
    if enable_zip234:
        applied.append("ZIP 234")
    if enable_zip235:
        applied.append("ZIP 235")
    if applied:
        if len(applied) == 1:
            return f"Zcash Inflation Chart with {applied[0]} Applied"
        if len(applied) == 2:
            return f"Zcash Inflation Chart with {applied[0]} and {applied[1]} Applied"
        return f"Zcash Inflation Chart with {', '.join(applied[:-1])}, and {applied[-1]} Applied"
    return "Zcash Inflation Chart - Status Quo"


def _instant_inflation_percent(
    heights: np.ndarray,
    subsidy_zat: np.ndarray,
    circulating_zat: np.ndarray,
) -> np.ndarray:
    """
    Inflation (%) as annualized issuance rate from current per-block subsidy:
      inflation = (subsidy_per_block * blocks_per_year) / circulating * 100
    """
    h = np.asarray(heights, dtype=np.int64)
    sub = np.asarray(subsidy_zat, dtype=np.float64)
    circ = np.asarray(circulating_zat, dtype=np.float64)
    blocks_per_year = np.where(
        h < BLOSSOM_HEIGHT,
        (365.25 * 24 * 3600) / PRE_BLOSSOM_BLOCK_SEC,
        (365.25 * 24 * 3600) / POST_BLOSSOM_BLOCK_SEC,
    ).astype(np.float64)
    out = np.full(len(sub), np.nan, dtype=float)
    mask = circ > 0
    out[mask] = (sub[mask] * blocks_per_year[mask] / circ[mask]) * 100.0
    return out


# -----------------------------
# Time axis: factual + extrapolated
# -----------------------------
def extend_time_axis(blocks_df: pd.DataFrame, end_height: int) -> pd.Series:
    """
    Returns a pd.Series of UTC timestamps indexed by height 0..end_height.
    Uses factual time_utc where available; extrapolates after tip using 75s/block.
    Also forward-fills holes if CSV is partial.
    """
    df = blocks_df[["height", "time_utc"]].copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_utc"]).sort_values("height")

    if df.empty:
        # fallback: synthetic time axis starting now (not ideal, but prevents crash)
        idx = pd.RangeIndex(0, end_height + 1, 1, name="height")
        now = pd.Timestamp.now(tz="UTC")
        return pd.Series(now + pd.to_timedelta(idx * POST_BLOSSOM_BLOCK_SEC, unit="s"), index=idx)

    tip_h = int(df["height"].iloc[-1])
    tip_t = df["time_utc"].iloc[-1]

    idx = pd.RangeIndex(0, end_height + 1, 1, name="height")
    out = pd.Series(index=idx, dtype="datetime64[ns, UTC]")

    factual = df.set_index("height")["time_utc"]
    factual = factual[(factual.index >= 0) & (factual.index <= end_height)]
    out.loc[factual.index] = factual

    if end_height > tip_h:
        future_heights = pd.RangeIndex(tip_h + 1, end_height + 1, 1)
        secs = (future_heights - tip_h) * POST_BLOSSOM_BLOCK_SEC
        out.loc[future_heights] = tip_t + pd.to_timedelta(secs, unit="s")

    # Fill holes in historical part if CSV is incomplete / still downloading
    out = out.ffill()

    # If genesis hole exists, backfill from first known
    out = out.bfill()

    return out


# -----------------------------
# Fee projection model
# -----------------------------
@dataclass
class FeeProjection:
    fee_per_tx_zat: int
    tx_per_block: float


def estimate_fee_projection(blocks_df: pd.DataFrame, anchor_blocks: int = FUTURE_FEE_ANCHOR_BLOCKS) -> FeeProjection:
    """
    Estimate typical fee_per_tx and tx_per_block from the recent window.
    Robust to partial CSV.
    """
    df = blocks_df[["height", "fees_zat", "tx_count"]].copy()
    df = df.sort_values("height").tail(anchor_blocks)

    # Avoid division by zero
    tx = pd.to_numeric(df["tx_count"], errors="coerce").fillna(0).astype(float)
    fees = pd.to_numeric(df["fees_zat"], errors="coerce").fillna(0).astype(float)

    fee_per_tx = fees / tx.replace(0, np.nan)
    fee_per_tx_med = float(np.nanmedian(fee_per_tx.values)) if np.isfinite(np.nanmedian(fee_per_tx.values)) else 0.0
    tx_per_block_med = float(np.nanmedian(tx.values)) if np.isfinite(np.nanmedian(tx.values)) else 1.0

    return FeeProjection(
        fee_per_tx_zat=int(max(0.0, fee_per_tx_med)),
        tx_per_block=max(0.0, tx_per_block_med),
    )


def tx_growth_multiplier(profile: str, t: np.ndarray, k: float) -> np.ndarray:
    """
    t: time in years since tip (>=0)
    profile:
      - flat: 1
      - linear: 1 + k*t
      - exponential: exp(k*t)
      - logistic: 1 + (L-1)/(1+exp(-k*(t - t0))) with L=5, t0=4
    """
    profile = (profile or "flat").lower()
    k = float(k)

    if profile == "flat":
        return np.ones_like(t, dtype=float)
    if profile == "linear":
        return 1.0 + k * t
    if profile == "exponential":
        return np.exp(k * t)
    if profile == "logistic":
        L = 5.0
        t0 = 4.0
        return 1.0 + (L - 1.0) / (1.0 + np.exp(-k * (t - t0)))
    # fallback
    return np.ones_like(t, dtype=float)


# -----------------------------
# Issuance models (status quo + ZIP234-like smoothing)
# -----------------------------
def build_status_quo_subsidy(
    heights: np.ndarray,
    blocks_df: pd.DataFrame,
    second_halving_height: int,
    halving_interval: int,
) -> np.ndarray:
    """
    Status quo:
      - For heights <= tip and present in CSV: use subsidy_zat from CSV.
      - For heights > tip: derive subsidy schedule from the subsidy at (or near) second_halving_height in CSV.
    """
    df = blocks_df[["height", "subsidy_zat"]].copy().sort_values("height")
    df["subsidy_zat"] = pd.to_numeric(df["subsidy_zat"], errors="coerce").fillna(0).astype(np.int64)

    tip_h = int(df["height"].max()) if not df.empty else 0

    # Map factual subsidies
    factual = df.set_index("height")["subsidy_zat"]

    out = np.zeros_like(heights, dtype=np.int64)
    mask_factual = heights <= tip_h
    # fill factual where possible
    h_f = heights[mask_factual]
    # Use reindex for possible missing heights; ffill to avoid gaps in partial CSV
    factual_full = factual.reindex(pd.RangeIndex(0, tip_h + 1, 1)).ffill().fillna(0).astype(np.int64)
    out[mask_factual] = factual_full.iloc[h_f].values

    # Derive baseline subsidy at second halving (fallback: last known non-zero)
    if second_halving_height <= tip_h and second_halving_height in factual_full.index:
        base = int(factual_full.loc[second_halving_height])
    else:
        nonzero = factual_full[factual_full > 0]
        base = int(nonzero.iloc[-1]) if len(nonzero) else 0

    # For future heights, apply halving schedule from second_halving_height
    mask_future = heights > tip_h
    if np.any(mask_future) and base > 0:
        h2 = heights[mask_future]
        # If h is before second halving, keep base (rare; only if tip < second halving)
        shifts = np.maximum(0, (h2 - second_halving_height) // halving_interval).astype(np.int64)
        # right shift: base // (2**shifts), but safe
        out_future = np.array([base >> int(min(s, 63)) for s in shifts], dtype=np.int64)
        out[mask_future] = out_future

    return out


def build_zip234_subsidy_after_activation(
    heights: np.ndarray,
    base_subsidy: np.ndarray,
    issued_supply_end_zat: np.ndarray,
    max_money_zat: int,
    activation_height: int,
    numerator: int,
    denominator: int,
) -> np.ndarray:
    """
    Simple "fixed activation" ZIP234 smoothing:
      - Before activation: base_subsidy.
      - From activation onward: subsidy = ceil(reserve * numerator / denominator), where reserve decreases by subsidy.
      - Reserve initialized as (max_money_zat - issued_supply_end_zat at (activation_height-1)).
    """
    out = base_subsidy.copy().astype(np.int64)

    # Need reserve at activation-1. If activation_height is 0, reserve is max.
    if activation_height <= 0:
        reserve0 = int(max_money_zat)
    else:
        h_prev = activation_height - 1
        if 0 <= h_prev < len(issued_supply_end_zat):
            reserve0 = int(max_money_zat - int(issued_supply_end_zat[h_prev]))
        else:
            reserve0 = int(max_money_zat - int(issued_supply_end_zat[min(len(issued_supply_end_zat) - 1, 0)]))

    reserve = max(0, reserve0)

    # Iterate from activation height to end (vectorization is hard due to recurrence)
    for h in range(activation_height, len(out)):
        if reserve <= 0:
            out[h] = 0
            continue
        # ceil(reserve * num / den)
        s = (reserve * int(numerator) + (int(denominator) - 1)) // int(denominator)
        s = max(0, int(s))
        out[h] = s
        reserve -= s

    return out


# -----------------------------
# Supply model with burns
# -----------------------------
def compute_issued_supply(
    heights: np.ndarray,
    blocks_df: pd.DataFrame,
    subsidy_zat: np.ndarray,
    max_money_zat: int,
) -> np.ndarray:
    """
    Issued supply:
      - For heights <= tip: use issued_supply_end_zat from CSV (fact).
      - For heights > tip: extend by cumulative subsidy.
    """
    df = blocks_df[["height", "issued_supply_end_zat"]].copy().sort_values("height")
    df["issued_supply_end_zat"] = pd.to_numeric(df["issued_supply_end_zat"], errors="coerce").fillna(0).astype(np.int64)

    tip_h = int(df["height"].max()) if not df.empty else 0

    factual = df.set_index("height")["issued_supply_end_zat"]
    factual_full = factual.reindex(pd.RangeIndex(0, tip_h + 1, 1)).ffill().fillna(0).astype(np.int64)

    out = np.zeros_like(heights, dtype=np.int64)
    mask_factual = heights <= tip_h
    out[mask_factual] = factual_full.iloc[heights[mask_factual]].values

    # Extend into future
    if np.any(heights > tip_h):
        supply = int(factual_full.iloc[-1]) if len(factual_full) else 0
        for h in range(tip_h + 1, len(out)):
            supply = min(int(max_money_zat), supply + int(subsidy_zat[h]))
            out[h] = supply

        # For completeness, fill any earlier gaps (if CSV has holes)
        out[: tip_h + 1] = np.minimum(out[: tip_h + 1], max_money_zat)

    return out


def compute_burned_supply(
    heights: np.ndarray,
    blocks_df: pd.DataFrame,
    time_axis: pd.Series,
    end_height: int,
    activation_height: int,
    enable_fee_burn: bool,
    fee_burn_ratio: float,
    enable_voluntary_burns: bool,
    burn_events_df: pd.DataFrame,
    enable_sprout_burn: bool,
    sprout_burn_height: int,
    sprout_burn_amount_zat: int,
    enable_future_activity: bool,
    future_profile: str,
    future_k: float,
    future_anchor_blocks: int,
    cap_tx_per_block: Optional[float],
) -> np.ndarray:
    """
    Burned supply (cumulative), combining:
      - fee burn: ratio * fees_zat from activation height, and projected fees (future) if enabled
      - voluntary burns: from events_burn.csv (applied from activation height)
      - one-time sprout burn at specified height
    """
    burned = np.zeros(end_height + 1, dtype=np.int64)
    activation_height = _clamp_int(int(activation_height), 0, int(end_height))

    # Historical fees and tx counts
    df = blocks_df[["height", "fees_zat", "tx_count"]].copy().sort_values("height")
    df["fees_zat"] = pd.to_numeric(df["fees_zat"], errors="coerce").fillna(0).astype(np.int64)
    df["tx_count"] = pd.to_numeric(df["tx_count"], errors="coerce").fillna(0).astype(np.int64)

    tip_h = int(df["height"].max()) if not df.empty else 0

    # Fee burn (historical)
    if enable_fee_burn and not df.empty:
        hist_end = min(tip_h, end_height)
        if activation_height <= hist_end:
            fees_full = (
                df.set_index("height")["fees_zat"]
                .reindex(pd.RangeIndex(activation_height, hist_end + 1, 1))
                .fillna(0)
                .astype(np.int64)
            )
            fee_burn_hist = (fees_full.values * float(fee_burn_ratio)).astype(np.int64)
            burned[activation_height : hist_end + 1] += fee_burn_hist

    # Project future fees (optional)
    if enable_fee_burn and enable_future_activity and end_height > tip_h:
        proj = estimate_fee_projection(blocks_df, anchor_blocks=max(1, int(future_anchor_blocks)))
        fee_per_tx = max(0, int(proj.fee_per_tx_zat))
        tx0 = float(proj.tx_per_block)

        # years since tip for each future height
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
        fee_burn_future = (fees_future * float(fee_burn_ratio)).astype(np.int64)
        future_start = max(tip_h + 1, activation_height)
        if future_start <= end_height:
            offset = future_start - (tip_h + 1)
            burned[future_start : end_height + 1] += fee_burn_future[offset:]

    # Voluntary burns
    if enable_voluntary_burns and burn_events_df is not None and not burn_events_df.empty:
        ev = burn_events_df.copy()
        if "height" in ev.columns and "burn_zat" in ev.columns:
            ev["height"] = pd.to_numeric(ev["height"], errors="coerce").fillna(-1).astype(int)
            ev["burn_zat"] = pd.to_numeric(ev["burn_zat"], errors="coerce").fillna(0).astype(np.int64)
            for _, row in ev.iterrows():
                h = int(row["height"])
                if activation_height <= h <= end_height:
                    burned[h] += int(row["burn_zat"])

    # One-time Sprout burn
    if enable_sprout_burn:
        h = max(int(sprout_burn_height), int(activation_height))
        if 0 <= h <= end_height:
            burned[h] += int(max(0, sprout_burn_amount_zat))

    # Convert to cumulative burned
    burned = np.cumsum(burned).astype(np.int64)
    return burned


def apply_burn_reissuance_to_subsidy(
    subsidy_zat: np.ndarray,
    burned_total_zat: np.ndarray,
    activation_height: int,
    numerator: int,
    denominator: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reissue mechanism (ZIP 234 style, for burned pool):
      - Burns reduce circulating supply immediately.
      - Burned amount is accumulated into a reissuance pool.
      - From activation onward, each block reissues ceil(pool * num / den).
      - Burn added at block h starts affecting subsidy from h+1 (future subsidies).
    Returns:
      - updated subsidy array
      - per-block reissued amount
    """
    out = subsidy_zat.copy().astype(np.int64)
    burn_total = np.asarray(burned_total_zat, dtype=np.int64)
    burn_delta = np.diff(np.concatenate(([0], burn_total))).astype(np.int64)
    reissued = np.zeros_like(out, dtype=np.int64)

    pool = 0
    start_h = _clamp_int(int(activation_height), 0, len(out) - 1)
    num = int(numerator)
    den = max(1, int(denominator))

    for h in range(start_h, len(out)):
        if pool > 0:
            add = (pool * num + (den - 1)) // den
            add = max(0, int(add))
            reissued[h] = add
            out[h] = int(out[h]) + add
            pool -= add

        d = int(burn_delta[h])
        if d > 0:
            pool += d

    return out, reissued


# -----------------------------
# UI app
# -----------------------------
cfg = load_config("config/config.yaml")
blocks_path = cfg.paths.blocks_csv
events_path = cfg.paths.events_burn_csv

ZIP234_NUM = int(cfg.issuance.zip234.numerator)
ZIP234_DEN = int(cfg.issuance.zip234.denominator)

# NSM activation default is taken directly from config.
DEFAULT_NSM_HEIGHT = int(cfg.defaults.nsm_activation_height)
DEFAULT_HORIZON_HEIGHT = int(cfg.defaults.horizon_end_height)
DISPLAY_START_DATE_UTC = pd.Timestamp(str(cfg.defaults.display_start_date_utc), tz="UTC")
DEFAULT_FEE_BURN_RATIO = float(cfg.burns.fee_burn.ratio)
FUTURE_FEE_ANCHOR_BLOCKS = int(cfg.future_activity.anchor_blocks)
DEFAULT_FUTURE_PROFILE = str(cfg.future_activity.default_profile).lower()
DEFAULT_FUTURE_PRESET = str(cfg.future_activity.default_preset).lower()
DEFAULT_FUTURE_LOGISTIC_K = float(cfg.future_activity.default_logistic_k)
PLOT_MAX_POINTS = max(1000, int(cfg.ui.plot.max_points))
PLOT_HEIGHT_VH = max(30, int(cfg.ui.plot.height_vh))
Y_AXIS_LEGACY_FULL_TOP_MIN = float(cfg.ui.y_axis.legacy_full_top_min)
Y_AXIS_ADAPTIVE_PADDING_RATIO = float(cfg.ui.y_axis.adaptive_padding_ratio)
Y_AXIS_ADAPTIVE_PADDING_MIN = float(cfg.ui.y_axis.adaptive_padding_min)
MARKER_LINE_WIDTH = int(cfg.ui.markers.line_width)
MARKER_LINE_DASH = str(cfg.ui.markers.line_dash)
MARKER_LINE_COLOR = str(cfg.ui.markers.line_color)
MARKER_OPACITY = float(cfg.ui.markers.opacity)
MARKER_LABEL_FONT_SIZE = int(cfg.ui.markers.label_font_size)
MARKER_LABEL_XSHIFT = int(cfg.ui.markers.label_xshift)
MARKER_LABEL_YSHIFT = int(cfg.ui.markers.label_yshift)
ENFORCE_FROM_NSM = bool(cfg.model.activation_rules.enforce_from_nsm)
FUTURE_ACTIVITY_PRESETS = {
    k: {"linear_k": float(v.linear_k), "exp_k": float(v.exponential_k)}
    for k, v in cfg.future_activity.presets.items()
}
_preset_priority = ["conservative", "base", "aggressive"]
_preset_tail = [k for k in FUTURE_ACTIVITY_PRESETS.keys() if k not in _preset_priority]
FUTURE_PRESET_KEYS = [k for k in _preset_priority if k in FUTURE_ACTIVITY_PRESETS] + _preset_tail
FUTURE_PRESET_OPTIONS = [
    {"label": k.replace("_", " ").title(), "value": k}
    for k in FUTURE_PRESET_KEYS
]


def _preset_help_text() -> str:
    parts = []
    for key in FUTURE_PRESET_KEYS:
        p = FUTURE_ACTIVITY_PRESETS[key]
        lin_pct_5y = p["linear_k"] * 5.0 * 100.0
        exp_pct_5y = (math.exp(p["exp_k"] * 5.0) - 1.0) * 100.0
        parts.append(
            f"{key.replace('_', ' ').title()} (~+{lin_pct_5y:.0f}% linear / ~+{exp_pct_5y:.0f}% exp over 5y)"
        )
    return "Used for ZIP 235 beyond tip: " + ", ".join(parts) + "."


FUTURE_PRESET_HELP_TEXT = _preset_help_text()

sprout_snap = get_sprout_snapshot(cfg)
DEFAULT_SPROUT_ZAT = int(sprout_snap["pool_sprout_zat"])
DEFAULT_SPROUT_HEIGHT_HINT = int(sprout_snap.get("measured_height", DEFAULT_NSM_HEIGHT))
DEFAULT_SPROUT_TIME_HINT = str(sprout_snap.get("measured_time_utc", ""))

app = Dash(__name__)
app.title = "Zcash NSM / Burns Visualizer"

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "padding": "14px"},
    children=[
        html.H2("Zcash NSM / Burns Visualizer (local)"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
            children=[
                html.Div(
                    children=[
                        html.Div("X axis"),
                        dcc.Dropdown(
                            id="x_axis_mode",
                            options=[
                                {"label": "Date (UTC)", "value": "date"},
                                {"label": "Height", "value": "height"},
                                {"label": "Years from genesis", "value": "years"},
                            ],
                            value="date",
                            clearable=False,
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Start height (blocks from genesis)"),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "42px 1fr 42px", "gap": "6px"},
                            children=[
                                html.Button("-1M", id="start_minus_1m", n_clicks=0),
                                dcc.Input(
                                    id="start_height",
                                    type="number",
                                    value=None,
                                    min=0,
                                    step="any",
                                    placeholder=f"empty = first block on/after {DISPLAY_START_DATE_UTC.date()}",
                                    style={"width": "100%"},
                                ),
                                html.Button("+1M", id="start_plus_1m", n_clicks=0),
                            ],
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Horizon (blocks from genesis)"),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "42px 1fr 42px", "gap": "6px"},
                            children=[
                                html.Button("-1M", id="horizon_minus_1m", n_clicks=0),
                                dcc.Input(
                                    id="horizon_end_height",
                                    type="number",
                                    value=None,
                                    min=0,
                                    step="any",
                                    placeholder=f"default = {DEFAULT_HORIZON_HEIGHT}",
                                    style={"width": "100%"},
                                ),
                                html.Button("+1M", id="horizon_plus_1m", n_clicks=0),
                            ],
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("NSM activation height"),
                        dcc.Input(
                            id="nsm_height",
                            type="number",
                            value=DEFAULT_NSM_HEIGHT,
                            style={"width": "100%"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div("Scenario toggles"),
                        dcc.Checklist(
                            id="toggles",
                            options=_scenario_toggle_options(False, False, False, False, False),
                            value=[],
                            style={"display": "grid", "gap": "6px"},
                        ),
                        html.Div(
                            "Options marked with ↳ depend on ZIP 233 and/or ZIP 235.",
                            style={"fontSize": "12px", "opacity": 0.75, "marginTop": "4px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Div("Fee burn ratio (0..1)"),
                        dcc.Input(
                            id="fee_burn_ratio",
                            type="number",
                            value=DEFAULT_FEE_BURN_RATIO,
                            min=0,
                            max=1,
                            step=0.01,
                            style={"width": "100%"},
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Sprout burn amount (zatoshis)"),
                        dcc.Input(
                            id="sprout_burn_amount",
                            type="number",
                            value=DEFAULT_SPROUT_ZAT,
                            style={"width": "100%"},
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Sprout burn height"),
                        dcc.Input(
                            id="sprout_burn_height",
                            type="number",
                            value=DEFAULT_NSM_HEIGHT,
                            style={"width": "100%"},
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Future activity profile (for ZIP 235 beyond tip)"),
                        dcc.Dropdown(
                            id="future_profile",
                            options=[
                                {"label": "Flat", "value": "flat"},
                                {"label": "Linear", "value": "linear"},
                                {"label": "Exponential", "value": "exponential"},
                                {"label": "Logistic", "value": "logistic"},
                            ],
                            value=DEFAULT_FUTURE_PROFILE,
                            clearable=False,
                        ),
                        html.Div(style={"height": "8px"}),
                        html.Div("Future activity preset"),
                        dcc.Dropdown(
                            id="future_preset",
                            options=FUTURE_PRESET_OPTIONS,
                            value=DEFAULT_FUTURE_PRESET,
                            clearable=False,
                        ),
                        html.Div(
                            FUTURE_PRESET_HELP_TEXT,
                            style={"fontSize": "12px", "opacity": 0.75, "marginTop": "4px"},
                        ),
                        html.Div(style={"height": "10px"}),
                        html.Div(
                            style={"fontSize": "12px", "opacity": 0.75},
                            children=[
                                "Sprout snapshot hint: ",
                                html.Code(f"{DEFAULT_SPROUT_ZAT}"),
                                " at height ",
                                html.Code(str(DEFAULT_SPROUT_HEIGHT_HINT)),
                                " (",
                                html.Code(DEFAULT_SPROUT_TIME_HINT),
                                ")",
                            ],
                        ),
                    ]
                ),
            ],
        ),
        html.Hr(),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px"},
            children=[
                html.Div(
                    children=[
                        html.H4(id="chart_title", style={"margin": "0 0 6px 0"}),
                        html.Div(
                            id="chart_wrap",
                            children=[
                                dcc.Graph(id="supply_chart", style={"height": f"{PLOT_HEIGHT_VH}vh"}),
                            ],
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.H4("Diagnostics"),
                        html.Pre(id="diag", style={"whiteSpace": "pre-wrap", "fontSize": "12px"}),
                    ]
                ),
            ],
        ),
        dcc.Interval(id="interval_refresh", interval=60_000, n_intervals=0),
    ],
)

@app.callback(
    Output("horizon_end_height", "style"),
    Output("start_height", "style"),
    Input("horizon_end_height", "value"),
    Input("start_height", "value"),
)
def validate_range_styles(horizon_end_height, start_height):
    normal = {"width": "100%"}
    error = {"width": "100%", "border": "2px solid #d93025", "backgroundColor": "#fff5f5"}
    if horizon_end_height is not None and start_height is not None:
        h = _to_int(horizon_end_height, DEFAULT_HORIZON_HEIGHT)
        s = _to_int(start_height, 0)
        if s > h:
            return error, error
    return normal, normal


def add_vline(fig: go.Figure, x_value, label: str):
    # Single source of truth for all vertical event markers (Blossom/halvings/NSM).
    fig.add_vline(
        x=x_value,
        line_width=MARKER_LINE_WIDTH,
        line_dash=MARKER_LINE_DASH,
        line_color=MARKER_LINE_COLOR,
        opacity=MARKER_OPACITY,
    )
    fig.add_annotation(
        x=x_value,
        y=1,
        yref="paper",
        text=label,
        showarrow=False,
        textangle=-90,
        xanchor="left",
        yanchor="top",
        xshift=MARKER_LABEL_XSHIFT,
        yshift=MARKER_LABEL_YSHIFT,
        font={"size": MARKER_LABEL_FONT_SIZE, "color": MARKER_LINE_COLOR},
    )


@app.callback(
    Output("toggles", "options"),
    Input("toggles", "value"),
)
def sync_toggle_dependencies(toggles):
    vals = list(toggles or [])
    enable_zip233 = "zip233" in vals
    enable_zip234 = "zip234" in vals
    enable_fee_burn = enable_zip233 and ("fee_burn" in vals)
    enable_sprout_burn = enable_zip233 and ("sprout_burn" in vals)
    burn_source_enabled = enable_zip233 and any(
        v in vals for v in ("fee_burn", "sprout_burn", "voluntary_burns")
    )
    return _scenario_toggle_options(
        enable_zip233,
        enable_zip234,
        enable_fee_burn,
        burn_source_enabled,
        enable_sprout_burn,
    )


@app.callback(
    Output("horizon_end_height", "value"),
    Input("horizon_minus_1m", "n_clicks"),
    Input("horizon_plus_1m", "n_clicks"),
    State("horizon_end_height", "value"),
    prevent_initial_call=True,
)
def nudge_horizon_by_million(_minus, _plus, horizon_value):
    base = DEFAULT_HORIZON_HEIGHT if horizon_value is None else _to_int(horizon_value, DEFAULT_HORIZON_HEIGHT)
    trigger = ctx.triggered_id
    if trigger == "horizon_minus_1m":
        return max(0, base - 1_000_000)
    if trigger == "horizon_plus_1m":
        return max(0, base + 1_000_000)
    return base


@app.callback(
    Output("start_height", "value"),
    Input("start_minus_1m", "n_clicks"),
    Input("start_plus_1m", "n_clicks"),
    State("start_height", "value"),
    prevent_initial_call=True,
)
def nudge_start_by_million(_minus, _plus, start_value):
    base = 0 if start_value is None else _to_int(start_value, 0)
    trigger = ctx.triggered_id
    if trigger == "start_minus_1m":
        return max(0, base - 1_000_000)
    if trigger == "start_plus_1m":
        return max(0, base + 1_000_000)
    return base


@app.callback(
    Output("chart_title", "children"),
    Output("supply_chart", "figure"),
    Output("diag", "children"),
    Input("interval_refresh", "n_intervals"),
    Input("x_axis_mode", "value"),
    Input("horizon_end_height", "value"),
    Input("start_height", "value"),
    Input("nsm_height", "value"),
    Input("toggles", "value"),
    Input("future_profile", "value"),
    Input("future_preset", "value"),
    Input("fee_burn_ratio", "value"),
    Input("sprout_burn_amount", "value"),
    Input("sprout_burn_height", "value"),
)
def update_chart(
    _tick,
    x_axis_mode,
    horizon_end_height,
    start_height,
    nsm_height,
    toggles,
    future_profile,
    future_preset,
    fee_burn_ratio,
    sprout_burn_amount,
    sprout_burn_height,
):
    try:
        parse_notes = []
        toggles = list(toggles or [])

        # Fast fail: obviously invalid numeric range should not trigger heavy model work.
        raw_end_h = None if horizon_end_height is None else _to_int(horizon_end_height, DEFAULT_HORIZON_HEIGHT)
        raw_start_h = None if start_height is None else _to_int(start_height, 0)
        if raw_end_h is not None and raw_start_h is not None and raw_start_h > raw_end_h:
            fig = go.Figure()
            fig.update_layout(
                title=None,
                xaxis_title="Date (UTC)",
                yaxis_title="Inflation (%/year)",
            )
            diag = (
                "input validation error:\n"
                f"- start_height ({raw_start_h}) cannot be greater than horizon_end_height ({raw_end_h}).\n"
                "- Fix inputs to continue."
            )
            return (
                "Zcash Inflation Chart - Input Validation Error",
                fig,
                diag,
            )

        # Load data (CSV can be mid-download; keep robust)
        blocks_df = load_blocks_csv(blocks_path)
        burn_events_df = load_burn_events_csv(events_path)

        blocks_df = blocks_df.copy()
        blocks_df["height"] = pd.to_numeric(blocks_df["height"], errors="coerce").fillna(0).astype(int)
        blocks_df = blocks_df.sort_values("height")

        max_height_csv = int(blocks_df["height"].max()) if not blocks_df.empty else 0

        # default display start: first block on/after configured start date
        if "time_utc" in blocks_df.columns and not blocks_df.empty:
            t = pd.to_datetime(blocks_df["time_utc"], utc=True, errors="coerce")
            m2019 = t >= DISPLAY_START_DATE_UTC
            if bool(m2019.any()):
                default_start_h = int(blocks_df.loc[m2019, "height"].iloc[0])
            else:
                default_start_h = 0
        else:
            default_start_h = 0

        # horizon: default anchored at 3rd halving height
        user_end_h = DEFAULT_HORIZON_HEIGHT if horizon_end_height is None else _to_int(horizon_end_height, DEFAULT_HORIZON_HEIGHT)
        user_end_h = max(0, user_end_h)

        end_h = user_end_h

        # activation height
        nsm_h = _to_int(nsm_height, DEFAULT_NSM_HEIGHT)
        nsm_h = _clamp_int(nsm_h, 0, end_h)

        # start of displayed range
        start_default_for_window = default_start_h if end_h >= default_start_h else 0
        user_start_h = start_default_for_window if start_height is None else _to_int(start_height, start_default_for_window)
        user_start_h = max(0, user_start_h)
        start_h = _clamp_int(user_start_h, 0, end_h)
        if start_h != user_start_h:
            parse_notes.append(f"start height clamped {user_start_h} -> {start_h}")
        if start_height is None and start_default_for_window != default_start_h:
            parse_notes.append(
                f"default configured start ({default_start_h}) is above horizon ({end_h}); using genesis start (0)"
            )

        heights = np.arange(0, end_h + 1, dtype=int)
        time_axis = extend_time_axis(blocks_df, end_h)

        # Status quo subsidy and issued supply
        base_subsidy = build_status_quo_subsidy(
            heights=heights,
            blocks_df=blocks_df,
            second_halving_height=SECOND_HALVING_HEIGHT_DEFAULT,
            halving_interval=POST_BLOSSOM_HALVING_INTERVAL,
        )
        base_issued = compute_issued_supply(
            heights=heights,
            blocks_df=blocks_df,
            subsidy_zat=base_subsidy,
            max_money_zat=MAX_MONEY_ZAT,
        )

        # Scenario subsidy (ZIP234 or baseline)
        enable_zip234 = "zip234" in toggles
        if enable_zip234:
            scenario_subsidy = build_zip234_subsidy_after_activation(
                heights=heights,
                base_subsidy=base_subsidy,
                issued_supply_end_zat=base_issued,
                max_money_zat=MAX_MONEY_ZAT,
                activation_height=nsm_h,
                numerator=ZIP234_NUM,
                denominator=ZIP234_DEN,
            )
        else:
            scenario_subsidy = base_subsidy.copy()

        # Burns
        enable_zip233 = "zip233" in toggles
        if not enable_zip233 and any(v in ZIP233_DEPENDENT_TOGGLES for v in toggles):
            parse_notes.append("ZIP 233 is required for burn options; dependent options were ignored.")
        enable_fee_burn = enable_zip233 and ("fee_burn" in toggles)
        enable_voluntary_burns = enable_zip233 and ("voluntary_burns" in toggles)
        enable_sprout_burn = enable_zip233 and ("sprout_burn" in toggles)
        # No separate checkbox: future activity is always used for ZIP 235 beyond tip.
        enable_future_activity = enable_fee_burn
        scenario_effect_start_h = nsm_h if ENFORCE_FROM_NSM else 0
        enable_reissue_burned = (
            enable_zip233
            and enable_zip234
            and ("reissue_burned" in toggles)
            and (enable_fee_burn or enable_voluntary_burns or enable_sprout_burn)
        )
        enable_sprout_reissue = (
            enable_reissue_burned
            and enable_sprout_burn
            and ("sprout_reissue" in toggles)
        )
        enable_zip235 = enable_fee_burn
        zip233_effective = enable_fee_burn or enable_voluntary_burns or enable_sprout_burn
        chart_title = _chart_title(
            enable_zip233=zip233_effective,
            enable_zip234=enable_zip234,
            enable_zip235=enable_zip235,
        )

        fee_ratio, ratio_note = _parse_ratio_input(fee_burn_ratio, DEFAULT_FEE_BURN_RATIO)
        if ratio_note:
            parse_notes.append(ratio_note)
        fee_ratio_raw = fee_ratio
        fee_ratio = max(0.0, min(1.0, fee_ratio))
        if fee_ratio != fee_ratio_raw:
            parse_notes.append(f"fee ratio clamped {fee_ratio_raw} -> {fee_ratio}")

        sprout_amt = int(max(0, _to_int(sprout_burn_amount, DEFAULT_SPROUT_ZAT)))
        sprout_h = _to_int(sprout_burn_height, nsm_h)
        sprout_h = _clamp_int(sprout_h, 0, end_h)
        if ENFORCE_FROM_NSM and enable_sprout_burn and sprout_h < nsm_h:
            parse_notes.append(f"sprout burn height shifted {sprout_h} -> {nsm_h} (scenarios start at NSM activation)")
            sprout_h = nsm_h

        # Future activity scenario is configurable, but only applied for ZIP 235 beyond tip.
        future_profile = str(future_profile or DEFAULT_FUTURE_PROFILE).lower()
        if future_profile not in {"flat", "linear", "exponential", "logistic"}:
            parse_notes.append(f"future profile normalized '{future_profile}' -> '{DEFAULT_FUTURE_PROFILE}'")
            future_profile = DEFAULT_FUTURE_PROFILE

        preset_name = str(future_preset or DEFAULT_FUTURE_PRESET).lower()
        if preset_name not in FUTURE_ACTIVITY_PRESETS:
            parse_notes.append(f"future preset normalized '{preset_name}' -> '{DEFAULT_FUTURE_PRESET}'")
            preset_name = DEFAULT_FUTURE_PRESET
        preset = FUTURE_ACTIVITY_PRESETS[preset_name]
        linear_k = max(0.0, min(2.0, float(preset["linear_k"])))
        exp_k = max(0.0, min(2.0, float(preset["exp_k"])))

        if future_profile == "linear":
            future_k = linear_k
        elif future_profile == "exponential":
            future_k = exp_k
        elif future_profile == "logistic":
            future_k = max(0.0, min(2.0, DEFAULT_FUTURE_LOGISTIC_K))
        else:
            future_k = 0.0
        cap_tx = None  # could be added to UI later

        scenario_burned = compute_burned_supply(
            heights=heights,
            blocks_df=blocks_df,
            time_axis=time_axis,
            end_height=end_h,
            activation_height=scenario_effect_start_h,
            enable_fee_burn=enable_fee_burn,
            fee_burn_ratio=fee_ratio,
            enable_voluntary_burns=enable_voluntary_burns,
            burn_events_df=burn_events_df,
            enable_sprout_burn=enable_sprout_burn,
            sprout_burn_height=sprout_h,
            sprout_burn_amount_zat=sprout_amt,
            enable_future_activity=enable_future_activity,
            future_profile=future_profile,
            future_k=future_k,
            future_anchor_blocks=FUTURE_FEE_ANCHOR_BLOCKS,
            cap_tx_per_block=cap_tx,
        )

        sprout_burn_cumulative = np.zeros_like(scenario_burned, dtype=np.int64)
        if enable_sprout_burn and 0 <= sprout_h <= end_h:
            sprout_burn_cumulative[sprout_h:] = sprout_amt

        reissued_per_block = np.zeros_like(scenario_subsidy, dtype=np.int64)
        if enable_reissue_burned:
            reissue_pool_burned = scenario_burned.copy()
            if enable_sprout_burn and not enable_sprout_reissue:
                reissue_pool_burned = np.maximum(
                    0,
                    reissue_pool_burned - sprout_burn_cumulative,
                ).astype(np.int64)
            scenario_subsidy, reissued_per_block = apply_burn_reissuance_to_subsidy(
                subsidy_zat=scenario_subsidy,
                burned_total_zat=reissue_pool_burned,
                activation_height=scenario_effect_start_h,
                numerator=ZIP234_NUM,
                denominator=ZIP234_DEN,
            )

        scenario_issued = compute_issued_supply(
            heights=heights,
            blocks_df=blocks_df,
            subsidy_zat=scenario_subsidy,
            max_money_zat=MAX_MONEY_ZAT,
        )

        # Baseline is status-quo circulating without scenario burns.
        base_burned = np.zeros_like(base_issued, dtype=np.int64)

        # Circulating = issued - burned (never below 0)
        base_circ = np.maximum(0, base_issued - base_burned).astype(np.int64)
        scen_circ = np.maximum(0, scenario_issued - scenario_burned).astype(np.int64)

        # Inflation series (old visualizer-compatible definition).
        base_infl_full = _instant_inflation_percent(heights, base_subsidy, base_circ)
        scen_infl_full = _instant_inflation_percent(heights, scenario_subsidy, scen_circ)
        reduction_pp_full = base_infl_full - scen_infl_full

        # Display window: user-controlled start height (default: first block on/after configured date).
        display_idx = np.arange(start_h, end_h + 1, dtype=int)
        if len(display_idx) == 0:
            display_idx = np.arange(len(heights), dtype=int)

        # X axis
        x_mode = (x_axis_mode or "date").lower()
        t0 = time_axis.iloc[0]
        if x_mode == "height":
            x = heights
            x_title = "Height"
        elif x_mode == "years":
            x = (time_axis - t0).dt.total_seconds().values / (365.25 * 24 * 3600)
            x_title = "Years from genesis (modeled)"
        else:
            x = time_axis.values
            x_title = "Date (UTC)"

        # Plot downsampling: keeps UI responsive on large horizons.
        local_idx = _plot_indices(len(display_idx), max_points=PLOT_MAX_POINTS)
        plot_idx = display_idx[local_idx]
        x_plot = np.asarray(x)[plot_idx]
        base_infl_plot = base_infl_full[plot_idx]
        scen_infl_plot = scen_infl_full[plot_idx]

        # Build figure
        fig = go.Figure()

        scenario_enabled = any(
            [
                enable_zip234,
                enable_fee_burn,
                enable_voluntary_burns,
                enable_sprout_burn,
                enable_reissue_burned,
            ]
        )
        fig.add_trace(
            go.Scattergl(
                x=x_plot,
                y=base_infl_plot,
                mode="lines",
                name="Status quo inflation (%/year)",
                line=dict(color="green", width=2),
            )
        )
        # Keep legend stable: always keep scenario trace in legend.
        scen_plot_for_render = scen_infl_plot.copy()
        # Scenario is shown from configured scenario effect start.
        scen_plot_for_render[plot_idx < scenario_effect_start_h] = np.nan
        scenario_visible = True
        if not scenario_enabled:
            scenario_visible = "legendonly"
            scen_plot_for_render = np.full_like(scen_plot_for_render, np.nan, dtype=float)
        # Draw scenario last so it stays visually on top of status quo.
        fig.add_trace(
            go.Scattergl(
                x=x_plot,
                y=scen_plot_for_render,
                mode="lines",
                name="Scenario inflation (%/year)",
                line=dict(color="orange", width=2.6, dash="solid"),
                visible=scenario_visible,
            )
        )

        # Vertical lines: Blossom, halving markers, NSM activation
        def _x_at_height(h: int):
            if x_mode == "height":
                return h
            if x_mode == "years":
                return (time_axis.iloc[h] - t0).total_seconds() / (365.25 * 24 * 3600)
            return time_axis.iloc[h]

        # Blossom line if in visible range
        if start_h <= BLOSSOM_HEIGHT <= end_h:
            add_vline(fig, _x_at_height(BLOSSOM_HEIGHT), "Blossom")

        # Halving lines: 1st/2nd/3rd/4rd and all subsequent as "next halving".
        halving_heights = []
        if start_h <= FIRST_HALVING_HEIGHT_DEFAULT <= end_h:
            halving_heights.append(FIRST_HALVING_HEIGHT_DEFAULT)
        for i in range(0, 8):
            hh = SECOND_HALVING_HEIGHT_DEFAULT + i * POST_BLOSSOM_HALVING_INTERVAL
            if start_h <= hh <= end_h:
                halving_heights.append(hh)
        for hh in halving_heights:
            if hh == FIRST_HALVING_HEIGHT_DEFAULT:
                lbl = "1st halving"
            elif hh == SECOND_HALVING_HEIGHT_DEFAULT:
                lbl = "2nd halving"
            elif hh == THIRD_HALVING_HEIGHT_DEFAULT:
                lbl = "3rd halving"
            elif hh == THIRD_HALVING_HEIGHT_DEFAULT + POST_BLOSSOM_HALVING_INTERVAL:
                lbl = "4rd halving"
            else:
                lbl = "next halving"
            add_vline(fig, _x_at_height(hh), lbl)

        # NSM activation
        if start_h <= nsm_h <= end_h:
            add_vline(fig, _x_at_height(nsm_h), "NSM activation")

        fig.update_layout(
            title=None,
            xaxis_title=x_title,
            yaxis_title="Inflation (%/year)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=40, r=20, t=24, b=40),
        )
        finite_vals = np.concatenate(
            [base_infl_plot[np.isfinite(base_infl_plot)], scen_infl_plot[np.isfinite(scen_infl_plot)]]
        )
        y_axis_mode = "adaptive"
        if finite_vals.size > 0:
            # Keep legacy wide scale on default full-view window, but use adaptive scale for detailed windows.
            if start_h == default_start_h:
                y_axis_mode = "legacy_full"
                y_top = max(Y_AXIS_LEGACY_FULL_TOP_MIN, float(np.nanmax(finite_vals) * 1.08))
                fig.update_yaxes(range=[0.0, y_top])
            else:
                vmin = float(np.nanmin(finite_vals))
                vmax = float(np.nanmax(finite_vals))
                span = max(vmax - vmin, 1e-9)
                pad = max(span * Y_AXIS_ADAPTIVE_PADDING_RATIO, Y_AXIS_ADAPTIVE_PADDING_MIN)
                y_lo = max(0.0, vmin - pad)
                y_hi = vmax + pad
                if y_hi <= y_lo:
                    y_hi = y_lo + 0.1
                fig.update_yaxes(range=[y_lo, y_hi])

        burned_total_h = int(scenario_burned[end_h]) if len(scenario_burned) else 0
        reissued_total_h = int(np.sum(reissued_per_block)) if len(reissued_per_block) else 0
        net_removed_h = burned_total_h - reissued_total_h
        reissued_cum = np.cumsum(reissued_per_block).astype(np.int64)

        # Diagnostics
        diag_lines = [
            f"blocks_csv: {blocks_path}",
            f"max_height_csv: {max_height_csv}",
            f"end_height requested: {user_end_h}",
            f"end_height effective: {end_h}",
            f"start_height default ({DISPLAY_START_DATE_UTC.date()}): {default_start_h}",
            f"start_height requested: {user_start_h}",
            f"start_height effective: {start_h}",
            f"NSM activation height: {nsm_h}",
            f"toggles: {toggles}",
            f"zip233 enabled: {'yes' if enable_zip233 else 'no'}",
            f"zip233 effective: {'yes' if zip233_effective else 'no'}",
            f"zip235 enabled: {'yes' if enable_zip235 else 'no'}",
            f"zip234 reissue burned enabled: {'yes' if enable_reissue_burned else 'no'}",
            f"zip234 sprout reissue enabled: {'yes' if enable_sprout_reissue else 'no'}",
            f"scenario curve enabled: {'yes' if scenario_enabled else 'no'}",
            f"fee_burn_ratio: {fee_ratio}",
            f"sprout_burn: {'on' if enable_sprout_burn else 'off'} at h={sprout_h}, amount={sprout_amt} zat ({sprout_amt / ZATOSHIS_PER_ZEC:.8f} ZEC)",
            f"voluntary burns loaded: {0 if burn_events_df is None else len(burn_events_df)} rows",
            (
                "future activity scenario: "
                f"{'active with ZIP 235 beyond tip' if enable_fee_burn else 'inactive (ZIP 235 is off)'} "
                f"(profile={future_profile}, preset={preset_name}, "
                f"k_active={future_k}, linear_k={linear_k}, exp_k={exp_k}, "
                f"anchor_blocks={FUTURE_FEE_ANCHOR_BLOCKS})"
            ),
            (
                "burns activation rule: "
                + (
                    f"fee/voluntary from h>={scenario_effect_start_h}; sprout at explicit height {sprout_h}"
                    if ENFORCE_FROM_NSM
                    else f"fee/voluntary from h>=0; sprout at explicit height {sprout_h}"
                )
            ),
            f"plot points shown: {len(plot_idx)} of {len(heights)}",
            f"inflation at horizon: baseline={base_infl_full[end_h]:.6f}%/yr, scenario={scen_infl_full[end_h]:.6f}%/yr, reduction={reduction_pp_full[end_h]:.6f} pp/yr",
            f"y-axis mode: {y_axis_mode}",
            "",
            "nsm burn/reissue balance at horizon:",
            f"- burned total: {burned_total_h} zat ({burned_total_h / ZATOSHIS_PER_ZEC:.8f} ZEC)",
            f"- reissued total: {reissued_total_h} zat ({reissued_total_h / ZATOSHIS_PER_ZEC:.8f} ZEC)",
            f"- net removed from circulation: {net_removed_h} zat ({net_removed_h / ZATOSHIS_PER_ZEC:.8f} ZEC)",
        ]

        # Scenario audit table at key checkpoints.
        checkpoint_heights = sorted(
            {
                int(start_h),
                int(nsm_h),
                int(min(max_height_csv, end_h)),
                int(end_h),
            }
        )
        diag_lines += [
            "",
            "scenario audit (selected heights):",
            "height | baseline_% | scenario_% | delta_pp | burned_zec | reissued_zec | net_removed_zec",
        ]
        for hh in checkpoint_heights:
            base_v = float(base_infl_full[hh])
            scen_v = float(scen_infl_full[hh])
            delta_v = float(reduction_pp_full[hh])
            burned_zec = float(scenario_burned[hh]) / ZATOSHIS_PER_ZEC
            reissued_zec = float(reissued_cum[hh]) / ZATOSHIS_PER_ZEC
            net_removed_zec = burned_zec - reissued_zec
            marker = "*" if hh == int(nsm_h) else " "
            diag_lines.append(
                f"{marker}{hh} | {base_v:.6f} | {scen_v:.6f} | {delta_v:.6f} | "
                f"{burned_zec:.8f} | {reissued_zec:.8f} | {net_removed_zec:.8f}"
            )

        # Help explain "no visible change" cases.
        diff_window = np.abs((scen_infl_full - base_infl_full)[start_h : end_h + 1])
        max_diff_pp = float(np.nanmax(diff_window)) if diff_window.size else 0.0
        diag_lines.append(f"max observed delta in window: {max_diff_pp:.12f} pp/yr")
        if max_diff_pp < 1e-12 and len(toggles) > 0:
            diag_lines.append("no visible scenario change: active options do not alter this selected window.")
            if "zip233" in toggles and ("zip234" not in toggles) and ("fee_burn" not in toggles) and ("sprout_burn" not in toggles) and ("voluntary_burns" not in toggles):
                diag_lines.append("- ZIP 233 alone enables burn mechanics but does not burn by itself.")
            if "fee_burn" in toggles and (nsm_h > max_height_csv):
                diag_lines.append("- ZIP 235 fee burn starts at NSM activation, and selected activation is above current tip.")
            if "zip234" in toggles and (end_h <= nsm_h):
                diag_lines.append("- ZIP 234 applies from NSM activation; selected window ends at/before activation.")
            if "reissue_burned" in toggles and not enable_reissue_burned:
                diag_lines.append("- Burn reissue requires ZIP 233 + ZIP 234 and at least one active burn source.")
        if parse_notes:
            diag_lines.append("parsing/normalization:")
            for note in parse_notes:
                diag_lines.append(f"- {note}")
        diag_lines += [
            "",
            "Notes:",
            "- Before tip: time axis uses factual time_utc from CSV.",
            "- After tip: time axis extrapolates with 75s/block (post-Blossom).",
            "- Status quo supply <= tip uses issued_supply_end_zat from CSV (fact).",
            "- Inflation = subsidy_per_block * blocks_per_year / circulating_supply.",
            f"- Default display start is first block on/after {DISPLAY_START_DATE_UTC.date()}.",
            (
                f"- Scenario effects are applied and displayed from h>={nsm_h} (NSM activation)."
                if ENFORCE_FROM_NSM
                else "- Scenario effects are not forced to start at NSM activation."
            ),
        ]
        return chart_title, fig, "\n".join(diag_lines)
    except Exception:
        fig = go.Figure()
        fig.update_layout(
            title="Zcash Supply (error; see diagnostics)",
            xaxis_title="Height",
            yaxis_title="ZEC",
        )
        return (
            "Zcash Inflation Chart - Error (see Diagnostics)",
            fig,
            traceback.format_exc(),
        )


if __name__ == "__main__":
    # Dash v3+ uses app.run (run_server is obsolete)
    app.run(debug=True, port=8050)
