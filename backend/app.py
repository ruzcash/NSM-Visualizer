from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.normpath(os.path.join(APP_ROOT, "..", "config", "config.yaml"))

app = FastAPI(title="NSM Visualizer Local Backend", version="0.1.0")

# Local dev convenience: allow your local site to fetch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def load_yaml_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_nested(d: Dict[str, Any], keys: list[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

async def zebra_rpc_call(rpc_url: str, method: str, params: list[Any] | None = None, timeout_ms: int = 1500) -> Dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": "nsm", "method": method, "params": params or []}
    timeout = httpx.Timeout(timeout_ms / 1000.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(rpc_url, json=payload)
        r.raise_for_status()
        data = r.json()
        if "error" in data and data["error"]:
            raise RuntimeError(f"Zebra RPC error: {data['error']}")
        return data["result"]

def extract_tip_height(chain_info: Dict[str, Any]) -> Optional[int]:
    # zcashd-style: blocks
    if isinstance(chain_info.get("blocks"), int):
        return chain_info["blocks"]
    return None

def extract_value_pools(chain_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    zcashd getblockchaininfo has `valuePools` (array of objects with id/chainValue).
    Zebra output aims to be compatible; fields may evolve.
    We'll handle both:
      - valuePools: [ { "id": "sprout", "chainValue": <float or str> }, ... ]
      - value_pools: { ... } (if any alternative exists)
    """
    vp = chain_info.get("valuePools")
    if isinstance(vp, list):
        out: Dict[str, Any] = {}
        for item in vp:
            if not isinstance(item, dict):
                continue
            pid = item.get("id")
            if not pid:
                continue
            out[str(pid)] = item
        return out

    # fallback: some implementations might use snake_case
    vp2 = chain_info.get("value_pools")
    if isinstance(vp2, dict):
        return vp2

    return {}

def zec_to_zat(value: Any) -> Optional[int]:
    """
    Accept float / int / string; try to parse ZEC amount and convert to zatoshi.
    """
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            zec = float(value)
        elif isinstance(value, str):
            zec = float(value)
        else:
            return None
        return int(round(zec * 100_000_000))
    except Exception:
        return None

def pick_sprout_pool_zat_from_valuepools(value_pools: Dict[str, Any]) -> Optional[int]:
    """
    zcashd uses valuePools items like:
      { "id": "sprout", "chainValue": 12345.6789, ... }
    We convert chainValue (ZEC) -> zatoshi.
    """
    spr = value_pools.get("sprout")
    if isinstance(spr, dict):
        # common field name in zcashd docs
        if "chainValue" in spr:
            return zec_to_zat(spr.get("chainValue"))
        # fallback
        if "chain_value" in spr:
            return zec_to_zat(spr.get("chain_value"))
    # if dict-form already has sprout value numeric
    if isinstance(spr, (int, float, str)):
        return zec_to_zat(spr)
    return None

def snapshot_from_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sprout_zat = get_nested(cfg, ["sprout", "snapshot", "pool_sprout_zat"], None)
    measured_height = get_nested(cfg, ["sprout", "snapshot", "measured_height"], None)
    measured_time_utc = get_nested(cfg, ["sprout", "snapshot", "measured_time_utc"], None)
    return {
        "measured_height": measured_height,
        "measured_time_utc": measured_time_utc,
        "sprout_pool_zat": sprout_zat,
        "source": "snapshot",
    }

def zebra_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled_default = bool(get_nested(cfg, ["backend", "enabled_default"], False))
    base_url = get_nested(cfg, ["backend", "rpc_url"], None) or get_nested(cfg, ["zebra", "rpc_url"], None)
    timeout_ms = int(get_nested(cfg, ["backend", "timeout_ms"], 1500))
    return {"enabled_default": enabled_default, "rpc_url": base_url, "timeout_ms": timeout_ms}

@app.get("/api/config")
def api_config():
    cfg = load_yaml_config()
    return {
        "loaded_from": DEFAULT_CONFIG_PATH,
        "defaults": cfg.get("defaults", {}),
        "issuance": cfg.get("issuance", {}),
        "burns": cfg.get("burns", {}),
        "future_activity_model": cfg.get("future_activity_model", {}),
        "sprout": cfg.get("sprout", {}),
        "backend": cfg.get("backend", cfg.get("zebra", {})),
    }

@app.get("/api/status")
async def api_status():
    cfg = load_yaml_config()
    zs = zebra_settings(cfg)

    # fallback: last known from snapshot
    snap = snapshot_from_yaml(cfg)

    if zs["enabled_default"] and zs["rpc_url"]:
        try:
            info = await zebra_rpc_call(zs["rpc_url"], "getblockchaininfo", timeout_ms=zs["timeout_ms"])
            tip = extract_tip_height(info)
            if tip is not None:
                return {"tip_height": tip, "tip_time_utc": utc_now_iso(), "source": "zebra"}
        except Exception:
            pass

    return {
        "tip_height": snap.get("measured_height"),
        "tip_time_utc": snap.get("measured_time_utc") or utc_now_iso(),
        "source": "snapshot",
    }

@app.get("/api/pools")
async def api_pools():
    cfg = load_yaml_config()
    zs = zebra_settings(cfg)
    snap = snapshot_from_yaml(cfg)

    if zs["enabled_default"] and zs["rpc_url"]:
        try:
            info = await zebra_rpc_call(zs["rpc_url"], "getblockchaininfo", timeout_ms=zs["timeout_ms"])
            tip = extract_tip_height(info)
            value_pools = extract_value_pools(info)
            sprout_zat = pick_sprout_pool_zat_from_valuepools(value_pools)

            if sprout_zat is not None:
                return {
                    "measured_height": tip,
                    "measured_time_utc": utc_now_iso(),
                    "sprout_pool_zat": sprout_zat,
                    "source": "zebra",
                }
        except Exception:
            pass

    return snap

@app.get("/api/health")
async def api_health():
    cfg = load_yaml_config()
    zs = zebra_settings(cfg)

    result = {
        "ok": True,
        "backend": "local",
        "rpc_url": zs.get("rpc_url"),
        "zebra_enabled_default": bool(zs.get("enabled_default", False)),
        "zebra_reachable": False,
        "tip_height": None,
        "error": None,
        "time_utc": utc_now_iso(),
    }

    # If zebra is disabled in config or rpc_url is missing, we're still "ok" but not live.
    if not result["zebra_enabled_default"] or not result["rpc_url"]:
        result["ok"] = True
        result["error"] = "Zebra backend disabled in config or rpc_url not set; using snapshot mode."
        return result

    try:
        info = await zebra_rpc_call(
            zs["rpc_url"],
            "getblockchaininfo",
            timeout_ms=int(zs.get("timeout_ms", 1500)),
        )
        tip = extract_tip_height(info)
        result["zebra_reachable"] = True
        result["tip_height"] = tip
        return result
    except Exception as e:
        result["ok"] = False
        result["zebra_reachable"] = False
        result["error"] = f"{type(e).__name__}: {e}"
        return result
