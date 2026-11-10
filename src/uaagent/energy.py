"""Energy track: uncertainty-aware residual-load forecast and reserve sizing.

Target: Finnish residual load (load - onshore wind) HORIZON hours ahead.
Model:  LightGBM quantile regression (5 % / 50 % / 95 %), conformalized (CQR).
Online: adaptive conformal inference (ACI) to hold coverage under drift.
Decision: how much fast-start reserve to hold above the scheduled (median)
          forecast - a fixed MW rule vs. the conformal upper bound.

Run with `python -m uaagent.energy`. Strictly chronological split:
train 2015-2017, calibration 2018, test 2019-01 .. 2020-09.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import conformal as cp
from .data import ARTIFACTS, load_energy

SEED = 7
ALPHA = 0.10
HORIZON = 3
UNIT_MW = 20.0  # illustrative fast-start unit size used to express reserve in "units"
ACI_GAMMA = 0.005
ACI_WINDOW = 24 * 60
ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
CONFIRM_ALPHA = 0.30  # an alert is "confirmed" if even the 70 % interval breaches the committed capacity


def build_features(df: pd.DataFrame, h: int = HORIZON) -> pd.DataFrame:
    """Row at time t (decision time) -> features known at t, target at t + h."""
    res = df["load"] - df["wind"]
    X = pd.DataFrame(index=df.index)
    for lag in [0, 1, 2, 3, 6]:
        X[f"res_lag{lag}"] = res.shift(lag)
        X[f"wind_lag{lag}"] = df["wind"].shift(lag)
    X["load_lag0"] = df["load"]
    X["res_d1"] = res - res.shift(1)
    X["res_d3"] = res - res.shift(3)
    X["wind_d1"] = df["wind"] - df["wind"].shift(1)
    X["wind_d3"] = df["wind"] - df["wind"].shift(3)
    X["wind_std6"] = df["wind"].rolling(6).std()
    # same target hour yesterday / last week (already observed at t because h < 24)
    X["res_same_hour_d1"] = res.shift(24 - h)
    X["res_same_hour_d7"] = res.shift(168 - h)
    # published day-ahead load forecast for the target hour and for now
    X["da_fc_target"] = df["load_da_forecast"].shift(-h)
    X["da_fc_err_now"] = df["load"] - df["load_da_forecast"]
    tt = df.index + pd.Timedelta(hours=h)
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * tt.hour / 24), np.cos(2 * np.pi * tt.hour / 24)
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * tt.dayofyear / 365.25), np.cos(2 * np.pi * tt.dayofyear / 365.25)
    X["dow"] = tt.dayofweek
    X["y"] = res.shift(-h)
    X["wind_now"] = df["wind"]
    X["target_time"] = tt
    return X.dropna()


# Hyper-parameters were selected by pinball loss on 2017 (models fitted on 2015-2016), i.e. without
# touching the calibration or test periods; the final models are refitted on 2015-2017.
MEDIAN_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=40)
TAIL_PARAMS = dict(n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=200, reg_lambda=5)


def fit_quantile(X: pd.DataFrame, y: pd.Series, q: float) -> lgb.LGBMRegressor:
    params = MEDIAN_PARAMS if q == 0.5 else TAIL_PARAMS
    m = lgb.LGBMRegressor(objective="quantile", alpha=q, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=SEED, verbose=-1, **params)
    return m.fit(X, y)


def reserve_stats(y: np.ndarray, scheduled: np.ndarray, reserve: np.ndarray) -> dict:
    short = y - (scheduled + reserve)
    return {
        "mean_reserve_mw": float(np.mean(reserve)),
        "shortfall_frequency": float(np.mean(short > 0)),
        "mean_shortfall_mw_when_short": float(short[short > 0].mean()) if (short > 0).any() else 0.0,
        "p99_shortfall_mw": float(np.quantile(np.maximum(short, 0), 0.99)),
        "max_shortfall_mw": float(max(short.max(), 0)),
    }


def _alert_stats(test: pd.DataFrame, reserve_ref: float, width_ref: float) -> dict:
    """Is the uncertainty signal informative? Compare what actually happened in flagged vs. un-flagged hours."""
    short = test["shortfall_static"] > 0
    under = np.maximum(test["y"] - test["mid"], 0)  # realised under-forecast [MW]
    out = {"reserve_ref_mw": reserve_ref, "width_ref_mw": width_ref, "base_rate_fixed_rule_shortfall": float(short.mean()),
           "mean_underforecast_mw_all_hours": float(under.mean())}
    for flag in ("risk_watch", "risk_confirmed", "uncertainty_high"):
        f = test[flag].to_numpy()
        out[flag] = {
            "hours": int(f.sum()), "share_of_hours": float(f.mean()),
            "fixed_rule_shortfall_rate_when_flagged": float(short[f].mean()) if f.any() else None,
            "fixed_rule_shortfall_rate_when_not_flagged": float(short[~f].mean()),
            "share_of_fixed_rule_shortfalls_flagged": float(f[short].mean()),
            "share_of_fixed_rule_shortfall_mwh_flagged": float(test.loc[f, "shortfall_static"].sum() / test["shortfall_static"].sum()),
            "mean_abs_error_mw_when_flagged": float(np.abs(test["y"] - test["mid"])[f].mean()) if f.any() else None,
            "mean_abs_error_mw_when_not_flagged": float(np.abs(test["y"] - test["mid"])[~f].mean()),
        }
    return out


def run() -> dict:
    data = build_features(load_energy())
    feats = [c for c in data.columns if c not in ("y", "wind_now", "target_time")]
    tt = pd.DatetimeIndex(data["target_time"])
    train, cal, test = data[tt.year <= 2017], data[tt.year == 2018], data[tt.year >= 2019].copy()

    models = {q: fit_quantile(train[feats], train["y"], q) for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2)}

    def predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo, mid, hi = (models[q].predict(df[feats]) for q in (ALPHA / 2, 0.5, 1 - ALPHA / 2))
        return np.minimum(lo, mid), mid, np.maximum(hi, mid)

    qlo_c, mid_c, qhi_c = predict(cal)
    qlo, mid, qhi = predict(test)
    y_c, y = cal["y"].to_numpy(), test["y"].to_numpy()

    scores = cp.cqr_scores(y_c, qlo_c, qhi_c)
    q_static = cp.conformal_quantile(scores, ALPHA)
    lo_s, hi_s = cp.interval_cqr(qlo, qhi, q_static)
    lo_a, hi_a, alpha_t = cp.aci_cqr(y, qlo, qhi, scores, ALPHA, gamma=ACI_GAMMA, window=ACI_WINDOW)
    # raw (un-conformalized) quantile-regression band and a constant-width split band, for comparison
    q_split = cp.conformal_quantile(cp.split_scores(y_c, mid_c), ALPHA)

    # one-sided upper bounds for the reserve decision: conformalize the upper quantile only
    up_scores = y_c - qhi_c
    q_up = cp.conformal_quantile(up_scores, ALPHA / 2)          # 95 % one-sided
    q_up_confirm = cp.conformal_quantile(up_scores, CONFIRM_ALPHA / 2)  # 85 % one-sided
    upper95 = qhi + q_up
    upper85 = qhi + q_up_confirm

    # --- reserve policies (scheduled generation = median forecast) -----------
    err_c = y_c - mid_c
    static_reserve = float(np.quantile(err_c, 1 - ALPHA / 2))  # fixed MW covering 95 % of calibration-year errors
    dyn_reserve = np.maximum(upper95 - mid, 0)
    # persistence baseline for context
    pers = test["res_lag0"].to_numpy()

    test["mid"], test["q_lo_raw"], test["q_hi_raw"] = mid, qlo, qhi
    test["lo_static"], test["hi_static"], test["lo"], test["hi"], test["alpha_t"] = lo_s, hi_s, lo_a, hi_a, alpha_t
    test["upper95"], test["upper85"] = upper95, upper85
    test["reserve_static"], test["reserve_dynamic"] = static_reserve, dyn_reserve
    test["committed_static"] = mid + static_reserve
    test["shortfall_static"] = np.maximum(y - (mid + static_reserve), 0)
    test["shortfall_dynamic"] = np.maximum(y - (mid + dyn_reserve), 0)
    width = hi_a - lo_a
    width_ref = float(np.quantile((qhi_c - qlo_c) + 2 * q_static, 0.95))
    test["width"] = width
    # Alert flags. Reference = standing reserve that the model itself considered sufficient in 90 % of
    # calibration-year hours. "watch": the 95 % upper bound needs more than that; "confirmed": so does the 85 % bound.
    reserve_ref = float(np.quantile(np.maximum(qhi_c + q_up - mid_c, 0), 0.90))
    test["reserve_ref"] = reserve_ref
    test["risk_watch"] = (upper95 - mid) > reserve_ref
    test["risk_confirmed"] = (upper85 - mid) > reserve_ref
    test["uncertainty_high"] = width > width_ref

    def roll_cov(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return pd.Series(((y >= lo) & (y <= hi)).astype(float)).rolling(24 * 30).mean().to_numpy()

    test["rollcov_static"], test["rollcov_aci"] = roll_cov(lo_s, hi_s), roll_cov(lo_a, hi_a)
    test["rollcov_raw"] = roll_cov(qlo, qhi)

    month = pd.DatetimeIndex(test["target_time"]).to_period("M").astype(str)
    by_month = [
        {"month": m,
         "coverage_raw": cp.coverage(y[idx], qlo[idx], qhi[idx]),
         "coverage_static": cp.coverage(y[idx], lo_s[idx], hi_s[idx]),
         "coverage_aci": cp.coverage(y[idx], lo_a[idx], hi_a[idx]),
         "width_aci": float(np.mean(width[idx])),
         "mean_reserve_dynamic": float(np.mean(dyn_reserve[idx]))}
        for m in sorted(set(month)) for idx in [np.flatnonzero(month == m)]
    ]

    # reserve comparison, incl. a fixed rule tuned (in hindsight, on test) to the same shortfall frequency
    dyn = reserve_stats(y, mid, dyn_reserve)
    stat = reserve_stats(y, mid, np.full_like(y, static_reserve))
    matched = float(np.quantile(y - mid, 1 - dyn["shortfall_frequency"]))
    stat_matched = reserve_stats(y, mid, np.full_like(y, matched))

    # demo replay window: 14 days containing the largest fixed-rule shortfall
    peak = int(np.argmax(test["shortfall_static"].to_numpy()))
    start = max(0, peak - 24 * 9)
    window = {"start": str(test["target_time"].iloc[start]), "end": str(test["target_time"].iloc[min(start + 24 * 14, len(test) - 1)])}

    metrics = {
        "alpha": ALPHA, "horizon_h": HORIZON, "unit_mw": UNIT_MW, "aci_gamma": ACI_GAMMA, "aci_window_h": ACI_WINDOW,
        "confirm_level_one_sided": 1 - CONFIRM_ALPHA / 2,
        "n": {"train": len(train), "calibration": len(cal), "test": len(test)},
        "periods": {"train": "2015-2017", "calibration": "2018", "test": f"{test['target_time'].iloc[0]} .. {test['target_time'].iloc[-1]}"},
        "accuracy": {
            "mae_model": float(np.mean(np.abs(y - mid))), "mae_persistence": float(np.mean(np.abs(y - pers))),
            "rmse_model": float(np.sqrt(np.mean((y - mid) ** 2))), "mean_residual_load_mw": float(np.mean(y)),
        },
        "coverage": {
            "raw_quantile_regression": cp.coverage(y, qlo, qhi),
            "split_constant_width": cp.coverage(y, mid - q_split, mid + q_split),
            "cqr_static": cp.coverage(y, lo_s, hi_s),
            "cqr_aci": cp.coverage(y, lo_a, hi_a),
            "upper95_one_sided": float(np.mean(y <= upper95)),
        },
        "mean_width_mw": {
            "raw_quantile_regression": float(np.mean(qhi - qlo)), "split_constant_width": float(2 * q_split),
            "cqr_static": float(np.mean(hi_s - lo_s)), "cqr_aci": float(np.mean(width)),
        },
        "worst_month_coverage": {k: float(min(b[k] for b in by_month)) for k in ("coverage_raw", "coverage_static", "coverage_aci")},
        "rolling30d_coverage_min": {
            "raw": float(np.nanmin(test["rollcov_raw"])), "static": float(np.nanmin(test["rollcov_static"])), "aci": float(np.nanmin(test["rollcov_aci"])),
        },
        "calibration_curve": [
            {"nominal": 1 - a, "cqr_static": cp.coverage(y, *cp.interval_cqr(qlo, qhi, cp.conformal_quantile(scores, a)))}
            for a in ALPHA_GRID
        ],
        "reserve": {
            "static_reserve_mw": static_reserve, "fixed_rule": stat, "conformal_dynamic": dyn,
            "fixed_rule_matched_to_dynamic_shortfall": {"reserve_mw": matched, **stat_matched},
            "dynamic_reserve_p05_p50_p95_mw": [float(v) for v in np.quantile(dyn_reserve, [0.05, 0.5, 0.95])],
        },
        "alerts": _alert_stats(test, reserve_ref, width_ref),
        "by_month": by_month,
        "demo_window": window,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    keep = ["target_time", "y", "mid", "q_lo_raw", "q_hi_raw", "lo_static", "hi_static", "lo", "hi", "alpha_t", "upper95", "upper85",
            "reserve_static", "reserve_dynamic", "reserve_ref", "committed_static", "shortfall_static", "shortfall_dynamic", "width",
            "risk_watch", "risk_confirmed", "uncertainty_high", "rollcov_raw", "rollcov_static", "rollcov_aci", "wind_now", "res_lag0"]
    out = test[keep].copy()
    out.insert(0, "decision_time", test.index)
    out.to_csv(ARTIFACTS / "energy_test.csv", index=False)
    (ARTIFACTS / "energy_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "by_month"}, indent=1))
    print(pd.DataFrame(m["by_month"]).round(3).to_string())
