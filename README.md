# NSM-Visualizer

A local interactive visualizer (Dash + Plotly, Python) for Zcash monetary policy scenarios:
- status quo based on factual chain data;
- ZIP 233 / ZIP 234 / ZIP 235 scenario modeling;
- burn/reissue scenarios with configurable future activity scenarios for ZIP 235 beyond tip.

## 1. What the app shows

Main chart:
- `Status quo inflation (%/year)` (green): baseline trajectory without scenario changes.
- `Scenario inflation (%/year)` (orange): trajectory based on selected options.

Diagnostics panel (right side):
- factual `max_height_csv`;
- requested and effective range (`start_height`, `end_height`);
- which ZIP options are actually active;
- burn/reissue balance;
- scenario checkpoints (scenario audit).

Run flow:
- update inputs/options first;
- press `Run calculation` (or `Ctrl+Enter`) to apply changes;
- the button highlights pending unapplied changes and shows in-progress state during recalculation.

## 2. Model invariants

- Up to current tip (`max_height_csv`), facts from `data/blocks_full_with_types.csv` are used.
- Historical facts are never overwritten (subsidy, fees, tx_count, issued supply, timestamps).
- Scenario effects are applied only from `NSM activation height` onward.
- If block range input is invalid (`start_height > end_height`), heavy model computation is skipped.
- Partial CSV datasets are handled via safe reindex/ffill logic to avoid crashes on missing heights.
- Exact-mode performance guard: extremely large horizons are rejected to keep UI responsive on local machines.

## 3. ZIP logic in UI

`Enable Funds Removal From Circulation (ZIP 233)`:
- enables the burn mechanics class (by itself it does not burn anything).

`Remove 60% of Transaction Fees (ZIP 235)`:
- depends on ZIP 233;
- burns `floor(fees_zat * ratio)` from NSM activation onward;
- uses factual fee data from CSV up to tip;
- beyond tip, future activity scenario is applied automatically (no separate checkbox).

`Apply Issuance Smoothing (ZIP 234)`:
- enables smoothed subsidy trajectory after NSM activation;
- before activation it follows status quo.

`Reissue Burned Amount in Future Subsidies (ZIP 234)`:
- available only with `ZIP 233 + ZIP 234` and at least one active burn source;
- burned amount is accumulated in a reissue pool and returned via future subsidies.

`One-time Sprout burn` and `Apply voluntary burns`:
- depend on ZIP 233;
- reduce circulating supply (level effect).

## 4. Inflation formula used in chart

Per-block inflation is calculated as:

`inflation = subsidy_per_block * blocks_per_year / circulating_supply * 100`

where:
- pre-Blossom uses 150 sec/block, post-Blossom uses 75 sec/block;
- `circulating = issued - burned`.

## 5. Project structure

- `viz/app.py` - UI, callbacks, validation, chart generation, diagnostics.
- `viz/io.py` - YAML and CSV loading.
- `viz/model.py` - pure modeling utilities.
- `config/config.yaml` - chain/scenario configuration.
- `data/blocks_full_with_types.csv` - factual base dataset.
- `data/events_burn.csv` - voluntary burn events.
- `data/update_fees_last_million.py` - `fees_zat` updater via RPC.
- `backend/app.py` - optional FastAPI helper backend (snapshot/RPC health endpoints).

## 6. `config/config.yaml` reference

Minimal working schema:

```yaml
meta:
  name: "Zcash NSM / Burns Visualizer"
  version: 1

paths:
  blocks_csv: "data/blocks_full_with_types.csv"
  events_burn_csv: "data/events_burn.csv"

defaults:
  nsm_activation_height: 3566401
  horizon_end_height: 6400000
  display_start_date_utc: "2019-01-01T00:00:00Z"

issuance:
  zip234:
    numerator: 4126
    denominator: 10000000000

burns:
  fee_burn:
    ratio: 0.60

sprout:
  snapshot:
    pool_sprout_zat: 2548104009822
    measured_height: 3233397
    measured_time_utc: "2026-02-08T00:00:00Z"

future_activity:
  anchor_blocks: 200000
  default_profile: "linear"
  default_preset: "base"
  default_logistic_k: 0.80
  presets:
    conservative:
      linear_k: 0.05
      exp_k: 0.04
    base:
      linear_k: 0.10
      exp_k: 0.08
    aggressive:
      linear_k: 0.20
      exp_k: 0.14

ui:
  plot:
    max_points: 20000
    height_vh: 72
  y_axis:
    legacy_full_top_min: 52.0
    adaptive_padding_ratio: 0.15
    adaptive_padding_min: 0.02
  markers:
    line_width: 1
    line_dash: "dash"
    line_color: "#2f4f6f"
    opacity: 0.75
    label_font_size: 11
    label_xshift: 2
    label_yshift: -2

model:
  activation_rules:
    enforce_from_nsm: true
```

Key meanings:
- `paths.*` - CSV file paths.
- `defaults.nsm_activation_height` - base NSM activation height used directly by UI default.
- `defaults.horizon_end_height` - default horizon in UI.
- `defaults.display_start_date_utc` - default start date anchor for the visible window.
- `issuance.zip234.*` - smoothing and reissue distribution parameters.
- `burns.fee_burn.ratio` - default fee burn ratio in UI.
- `sprout.snapshot.*` - default values/hints for Sprout burn.
- `future_activity.anchor_blocks` - calibration window near tip.
- `future_activity.presets.*` - user-facing activity growth presets.
- `ui.plot.*` - chart height and downsampling limit.
- `ui.y_axis.*` - y-axis scaling behavior.
- `ui.markers.*` - vertical marker style and label offsets.
- `model.activation_rules.enforce_from_nsm` - forces scenario effects to start from NSM activation.

Mainnet constants are hardcoded in code (not configurable via YAML):
- `zatoshis_per_zec = 100_000_000`
- `max_money_zat = 2_100_000_000_000_000`
- `second_halving_height = 2_726_400`
- `halving_interval_blocks = 1_680_000` (post-Blossom)

## 7. Quick start (local)

Requirements:
- Python 3.11+ (3.12/3.13 recommended).

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run Dash UI:

```bash
python -m viz.app
```

Open:
- `http://127.0.0.1:8050`

Usage:
- after changing any input, click `Run calculation` to update the chart;
- if range is invalid (`start > horizon`), the run button is disabled until fixed.

Optional backend (if API health/snapshot endpoints are needed):

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## 8. Deployment notes

Non-container setup:
- process 1: Dash (`python -m viz.app`) on an internal port;
- process 2: FastAPI (`uvicorn backend.app:app ...`) if needed;
- external reverse proxy (Nginx/Caddy) on 80/443.

Recommendations:
- keep `config/config.yaml` and `data/*.csv` on persistent storage;
- run processes under `systemd` or a supervisor;
- disable debug mode in production.
