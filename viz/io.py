# viz/io.py
#
# I/O helpers for the NSM visualizer:
# - load_config: reads YAML and returns an AppConfig dataclass
# - load_blocks_csv: reads blocks_full_with_types.csv
# - load_burn_events_csv: reads events_burn.csv
# - get_sprout_snapshot: returns sprout snapshot dict (for defaults/hints)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import yaml


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass(frozen=True)
class PathsConfig:
    blocks_csv: str
    events_burn_csv: str
    compiled_chain_npz: str
    compiled_meta_json: str


@dataclass(frozen=True)
class DefaultsConfig:
    nsm_activation_height: int
    horizon_end_height: int
    display_start_date_utc: str


@dataclass(frozen=True)
class IssuanceZip234Config:
    numerator: int
    denominator: int


@dataclass(frozen=True)
class IssuanceConfig:
    zip234: IssuanceZip234Config


@dataclass(frozen=True)
class BurnsFeeBurnConfig:
    ratio: float


@dataclass(frozen=True)
class BurnsConfig:
    fee_burn: BurnsFeeBurnConfig


@dataclass(frozen=True)
class SproutSnapshotConfig:
    pool_sprout_zat: int
    measured_height: int
    measured_time_utc: str


@dataclass(frozen=True)
class SproutConfig:
    snapshot: SproutSnapshotConfig


@dataclass(frozen=True)
class FutureActivityConfig:
    @dataclass(frozen=True)
    class Preset:
        linear_k: float
        exponential_k: float

    anchor_blocks: int
    default_profile: str
    default_preset: str
    default_logistic_k: float
    presets: Dict[str, "FutureActivityConfig.Preset"]


@dataclass(frozen=True)
class UiPlotConfig:
    max_points: int
    height_vh: int


@dataclass(frozen=True)
class UiYAxisConfig:
    legacy_full_top_min: float
    adaptive_padding_ratio: float
    adaptive_padding_min: float


@dataclass(frozen=True)
class UiMarkersConfig:
    line_width: int
    line_dash: str
    line_color: str
    opacity: float
    label_font_size: int
    label_xshift: int
    label_yshift: int


@dataclass(frozen=True)
class UiConfig:
    plot: UiPlotConfig
    y_axis: UiYAxisConfig
    markers: UiMarkersConfig


@dataclass(frozen=True)
class ModelActivationRulesConfig:
    enforce_from_nsm: bool


@dataclass(frozen=True)
class ModelConfig:
    activation_rules: ModelActivationRulesConfig


@dataclass(frozen=True)
class RuntimeConfig:
    require_compiled_chain: bool


@dataclass(frozen=True)
class MetaConfig:
    name: str
    version: int


@dataclass(frozen=True)
class AppConfig:
    meta: MetaConfig
    paths: PathsConfig
    defaults: DefaultsConfig
    issuance: IssuanceConfig
    burns: BurnsConfig
    sprout: SproutConfig
    future_activity: FutureActivityConfig
    ui: UiConfig
    model: ModelConfig
    runtime: RuntimeConfig


# -----------------------------
# YAML helpers
# -----------------------------
def _dget(d: Dict[str, Any], dotted: str, default=None):
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def load_config(path: str) -> AppConfig:
    """
    Load YAML config from `path` and return AppConfig.
    YAML is expected to be the minimal schema we agreed on.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # meta
    meta_name = str(_dget(raw, "meta.name", "Zcash NSM / Burns Visualizer"))
    meta_version = int(_dget(raw, "meta.version", 1))
    meta = MetaConfig(name=meta_name, version=meta_version)

    # paths
    blocks_csv = str(_dget(raw, "paths.blocks_csv", "data/blocks_full_with_types.csv"))
    events_burn_csv = str(_dget(raw, "paths.events_burn_csv", "data/events_burn.csv"))
    compiled_chain_npz = str(_dget(raw, "paths.compiled_chain_npz", "data/compiled/chain_v1.npz"))
    compiled_meta_json = str(_dget(raw, "paths.compiled_meta_json", "data/compiled/meta_v1.json"))
    paths_cfg = PathsConfig(
        blocks_csv=blocks_csv,
        events_burn_csv=events_burn_csv,
        compiled_chain_npz=compiled_chain_npz,
        compiled_meta_json=compiled_meta_json,
    )

    # defaults
    nsm_activation_height = int(_dget(raw, "defaults.nsm_activation_height", 3_566_400))
    horizon_end_height = int(_dget(raw, "defaults.horizon_end_height", 6_400_000))
    display_start_date_utc = str(_dget(raw, "defaults.display_start_date_utc", "2019-01-01T00:00:00Z"))
    defaults_cfg = DefaultsConfig(
        nsm_activation_height=nsm_activation_height,
        horizon_end_height=horizon_end_height,
        display_start_date_utc=display_start_date_utc,
    )

    # issuance
    zip234_num = int(_dget(raw, "issuance.zip234.numerator", 4126))
    zip234_den = int(_dget(raw, "issuance.zip234.denominator", 10_000_000_000))
    zip234_cfg = IssuanceZip234Config(numerator=zip234_num, denominator=zip234_den)

    issuance_cfg = IssuanceConfig(zip234=zip234_cfg)

    # burns
    fee_ratio = float(_dget(raw, "burns.fee_burn.ratio", 0.60))
    burns_cfg = BurnsConfig(fee_burn=BurnsFeeBurnConfig(ratio=fee_ratio))

    # sprout snapshot (optional but useful)
    pool_sprout_zat = int(_dget(raw, "sprout.snapshot.pool_sprout_zat", 0))
    measured_height = int(_dget(raw, "sprout.snapshot.measured_height", 0))
    measured_time_utc = str(_dget(raw, "sprout.snapshot.measured_time_utc", ""))
    sprout_cfg = SproutConfig(
        snapshot=SproutSnapshotConfig(
            pool_sprout_zat=pool_sprout_zat,
            measured_height=measured_height,
            measured_time_utc=measured_time_utc,
        )
    )

    # future activity defaults
    future_anchor_blocks = int(_dget(raw, "future_activity.anchor_blocks", 200_000))
    future_default_profile = str(_dget(raw, "future_activity.default_profile", "linear")).lower()
    future_default_preset = str(_dget(raw, "future_activity.default_preset", "base")).lower()
    future_default_logistic_k = float(_dget(raw, "future_activity.default_logistic_k", 0.80))
    presets_raw = _dget(raw, "future_activity.presets", {}) or {}
    presets: Dict[str, FutureActivityConfig.Preset] = {}
    for name, values in presets_raw.items():
        if not isinstance(values, dict):
            continue
        presets[str(name).lower()] = FutureActivityConfig.Preset(
            linear_k=float(values.get("linear_k", 0.10)),
            exponential_k=float(values.get("exp_k", values.get("exponential_k", 0.08))),
        )
    if "conservative" not in presets:
        presets["conservative"] = FutureActivityConfig.Preset(linear_k=0.05, exponential_k=0.04)
    if "base" not in presets:
        presets["base"] = FutureActivityConfig.Preset(linear_k=0.10, exponential_k=0.08)
    if "aggressive" not in presets:
        presets["aggressive"] = FutureActivityConfig.Preset(linear_k=0.20, exponential_k=0.14)
    if future_default_preset not in presets:
        future_default_preset = "base"

    future_cfg = FutureActivityConfig(
        anchor_blocks=future_anchor_blocks,
        default_profile=future_default_profile,
        default_preset=future_default_preset,
        default_logistic_k=future_default_logistic_k,
        presets=presets,
    )

    # UI defaults
    ui_plot_cfg = UiPlotConfig(
        max_points=int(_dget(raw, "ui.plot.max_points", 20_000)),
        height_vh=int(_dget(raw, "ui.plot.height_vh", 72)),
    )
    ui_y_axis_cfg = UiYAxisConfig(
        legacy_full_top_min=float(_dget(raw, "ui.y_axis.legacy_full_top_min", 52.0)),
        adaptive_padding_ratio=float(_dget(raw, "ui.y_axis.adaptive_padding_ratio", 0.15)),
        adaptive_padding_min=float(_dget(raw, "ui.y_axis.adaptive_padding_min", 0.02)),
    )
    ui_markers_cfg = UiMarkersConfig(
        line_width=int(_dget(raw, "ui.markers.line_width", 1)),
        line_dash=str(_dget(raw, "ui.markers.line_dash", "dash")),
        line_color=str(_dget(raw, "ui.markers.line_color", "#2f4f6f")),
        opacity=float(_dget(raw, "ui.markers.opacity", 0.75)),
        label_font_size=int(_dget(raw, "ui.markers.label_font_size", 11)),
        label_xshift=int(_dget(raw, "ui.markers.label_xshift", 2)),
        label_yshift=int(_dget(raw, "ui.markers.label_yshift", -2)),
    )
    ui_cfg = UiConfig(plot=ui_plot_cfg, y_axis=ui_y_axis_cfg, markers=ui_markers_cfg)

    # Model behavior flags
    model_cfg = ModelConfig(
        activation_rules=ModelActivationRulesConfig(
            enforce_from_nsm=bool(_dget(raw, "model.activation_rules.enforce_from_nsm", True))
        )
    )
    runtime_cfg = RuntimeConfig(
        require_compiled_chain=bool(_dget(raw, "runtime.require_compiled_chain", True))
    )

    return AppConfig(
        meta=meta,
        paths=paths_cfg,
        defaults=defaults_cfg,
        issuance=issuance_cfg,
        burns=burns_cfg,
        sprout=sprout_cfg,
        future_activity=future_cfg,
        ui=ui_cfg,
        model=model_cfg,
        runtime=runtime_cfg,
    )


def get_sprout_snapshot(cfg: AppConfig) -> Dict[str, Any]:
    """
    Return a dict-like snapshot used for UI defaults/hints.
    """
    snap = cfg.sprout.snapshot
    return {
        "pool_sprout_zat": int(snap.pool_sprout_zat),
        "measured_height": int(snap.measured_height),
        "measured_time_utc": str(snap.measured_time_utc),
    }


# -----------------------------
# CSV loaders
# -----------------------------
def load_blocks_csv(path: str) -> pd.DataFrame:
    """
    Load blocks_full_with_types.csv.
    Required columns (minimum):
      height, time_utc, subsidy_zat, fees_zat, tx_count, issued_supply_end_zat
    Extra columns are allowed.
    """
    df = pd.read_csv(path)
    # Minimal coercions; app/model will be defensive too
    if "height" in df.columns:
        df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
    return df


def load_burn_events_csv(path: str) -> pd.DataFrame:
    """
    Load events_burn.csv (optional).
    Expected header:
      height,burn_zat,label,notes
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["height", "burn_zat", "label", "notes"])
    except Exception:
        # If user created only header or malformed, still return empty-safe frame
        return pd.DataFrame(columns=["height", "burn_zat", "label", "notes"])

    # Normalize columns if missing
    for c in ["height", "burn_zat", "label", "notes"]:
        if c not in df.columns:
            df[c] = None
    return df
