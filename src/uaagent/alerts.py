"""Build the alert feed from the pipeline artifacts.

`marine_alerts` / `energy_alerts` emit the uncertainty-aware alert dicts
consumed by `uaagent.agent.explain`; `marine_naive_alerts` emits the
conventional point-estimate alarm feed shown for comparison.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .conformal import conformal_quantile
from .marine import count_raises, episodes

_MARINE_KIND = {"kMc": "compressor_decay", "kMt": "turbine_decay"}

# Marine sensor columns carried into alert context.
_MARINE_CTX_COLS = {
    "speed_knots": "v",
    "fuel_flow_kg_s": "mf",
    "T48_C": "T48",
    "T2_C": "T2",
    "P2_bar": "P2",
    "fuel_excess_pct": "fuel_excess_pct",
}


def marine_alerts(voy: pd.DataFrame, metrics: dict) -> list[dict]:
    """UA alerts at every debounced watch/confirmed raise on the voyage."""
    out: list[dict] = []
    alpha = metrics["alpha"]
    for target, kind in _MARINE_KIND.items():
        thr = metrics["thresholds"][target]
        typical = metrics["targets"][target]["voyage"]["median_width_adaptive"]
        watch_state = voy[f"{target}_watch_state"].to_numpy(dtype=bool)
        conf_state = voy[f"{target}_confirmed_state"].to_numpy(dtype=bool)
        raises = [
            (s, "watch") for s in count_raises(watch_state)
            if not conf_state[s]
        ] + [(s, "confirmed") for s in count_raises(conf_state)]
        for step, state in raises:
            row = voy.iloc[step]
            lo, hi = float(row[f"{target}_lo"]), float(row[f"{target}_hi"])
            ctx = {k: float(row[col]) for k, col in _MARINE_CTX_COLS.items()}
            ctx["interval_width"] = hi - lo
            ctx["typical_width"] = float(typical)
            out.append({
                "id": f"M-{target}-{step}-{state}",
                "domain": "marine",
                "kind": kind,
                "state": state,
                "t": f"hour {step}",
                "estimate": float(row[f"{target}_mu"]),
                "lo": lo,
                "hi": hi,
                "threshold": float(thr),
                "coverage": 1.0 - alpha,
                "unit": "",
                "context": ctx,
                "step": int(step),
                "source": "ua",
            })
    return sorted(out, key=lambda a: a["step"])


def marine_naive_alerts(voy: pd.DataFrame, metrics: dict) -> list[dict]:
    """Conventional point-estimate alarm raises for comparison."""
    out: list[dict] = []
    for target, kind in _MARINE_KIND.items():
        thr = metrics["thresholds"][target]
        for step in count_raises(voy[f"{target}_naive_state"].to_numpy(dtype=bool)):
            row = voy.iloc[step]
            out.append({
                "id": f"M-{target}-{step}-naive",
                "step": int(step),
                "kind": kind,
                "estimate": float(row[f"{target}_mu"]),
                "threshold": float(thr),
                "is_false": bool(row[target] >= thr),
                "source": "naive",
            })
    return sorted(out, key=lambda a: a["step"])


def energy_alerts(df: pd.DataFrame, metrics: dict) -> list[dict]:
    """UA alerts from debounced reserve-risk and high-uncertainty flags."""
    out: list[dict] = []
    unit_mw = float(metrics["unit_mw"])
    horizon = metrics["horizon_h"]
    typical_width = float(df["width"].median())

    def ctx(row) -> dict:
        committed = float(row["mid"] + row["reserve_ref"])
        margin = committed - float(row["upper95"])
        extra = math.ceil(max(float(row["upper95"]) - committed, 0.0) / unit_mw)
        return {
            "forecast_mw": float(row["mid"]),
            "lo_mw": float(row["lo"]),
            "hi_mw": float(row["hi"]),
            "upper_mw": float(row["upper95"]),
            "upper85_mw": float(row["upper85"]),
            "committed_mw": committed,
            "standing_reserve_mw": float(row["reserve_ref"]),
            "recommended_reserve_mw": float(row["reserve_dynamic"]),
            "margin_mw": margin,
            "extra_units": int(extra),
            "unit_mw": unit_mw,
            "wind_mw": float(row["wind_now"]),
            "horizon_h": float(horizon),
            "interval_width": float(row["width"]),
            "typical_width": typical_width,
        }

    def mk(step: int, kind: str, state: str) -> dict:
        row = df.iloc[step]
        committed = float(row["mid"] + row["reserve_ref"])
        return {
            "id": f"E-{step}-{kind}-{state}",
            "domain": "energy",
            "kind": kind,
            "state": state,
            "t": pd.Timestamp(row["target_time"]).strftime("%Y-%m-%d %H:%M UTC"),
            "estimate": float(row["mid"]),
            "lo": float(row["lo"]),
            "hi": float(row["hi"]),
            "threshold": committed,
            "coverage": 1.0 - metrics["alpha"],
            "unit": "MW",
            "context": ctx(row),
            "step": int(step),
            "source": "ua",
        }

    watch_state = episodes(df["risk_watch"].to_numpy(dtype=bool), persist=2)
    for step in count_raises(watch_state):
        state = "confirmed" if bool(df.iloc[step]["risk_confirmed"]) else "watch"
        out.append(mk(step, "reserve_shortfall_risk", state))

    unc_state = episodes(df["uncertainty_high"].to_numpy(dtype=bool), persist=2)
    for step in count_raises(unc_state):
        out.append(mk(step, "forecast_uncertainty_high", "watch"))

    return sorted(out, key=lambda a: a["step"])


def interval_stats_at_level(
    test: pd.DataFrame,
    target: str,
    scores_norm: np.ndarray,
    threshold: float,
    level: float,
) -> dict:
    """Recompute adaptive-normalised interval stats at a nominal coverage level.

    Pure function used by the Trust page's interactive slider and by tests.
    Definitions match the pipeline metrics: coverage = mean(lo <= y <= hi);
    false-alarm rate = mean(hi < thr | healthy); confirmed detection =
    mean(hi < thr | degraded).
    """
    q = conformal_quantile(scores_norm, 1.0 - level)
    mu = test[f"{target}_mu"].to_numpy(dtype=float)
    s = test[f"{target}_sigma"].to_numpy(dtype=float)
    lo, hi = mu - q * s, mu + q * s
    y = test[target].to_numpy(dtype=float)
    healthy = y >= threshold
    alarm = hi < threshold
    return {
        "nominal": level,
        "q": float(q),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(hi - lo)),
        "false_alarm_rate": (float(np.mean(alarm[healthy]))
                             if healthy.any() else float("nan")),
        "confirmed_detection": (float(np.mean(alarm[~healthy]))
                                if (~healthy).any() else float("nan")),
    }
