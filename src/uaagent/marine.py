"""Marine track: uncertainty-aware health estimation of a propulsion plant.

Pipeline (run with `python -m uaagent.marine`):
1. Load the public UCI naval propulsion dataset (noise-free simulator output).
2. Inject instrument-class sensor noise (documented assumption, see NOISE_REL).
3. Hold out every (kMc, kMt) grid cell visited by the replay voyage, so the
   voyage is never seen in training or calibration.
4. Fit a three-model ensemble for the point estimate and a difficulty model
   sigma(x); calibrate split and locally-adaptive conformal intervals.
5. Evaluate on an i.i.d. test split and on the voyage; write artifacts.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import conformal as cp
from .data import ARTIFACTS, MARINE_FEATURES, MARINE_SENSORS, load_marine

SEED = 7
ALPHA = 0.10
# Relative 1-sigma measurement noise per sensor (assumption, typical instrument classes).
NOISE_REL = {
    "GTT": 0.005, "Ts": 0.005, "Tp": 0.005,      # torque meters 0.5 %
    "GTn": 0.0005, "GGn": 0.0005,                # speed pickups 0.05 %
    "T48": 0.002, "T2": 0.002,                   # thermocouples 0.2 %
    "P48": 0.002, "P2": 0.002, "Pexh": 0.002,    # pressure transmitters 0.2 %
    "TIC": 0.005, "mf": 0.005,                   # fuel metering 0.5 %
}
THRESHOLDS = {"kMc": 0.975, "kMt": 0.985}  # act when the coefficient is below this
PERSIST = 3  # consecutive steps required to raise / clear an alert
N_STEPS = 720
ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def add_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for col, rel in NOISE_REL.items():
        out[col] = df[col] * (1 + rel * rng.standard_normal(len(df)))
    return out


def make_voyage(rng: np.random.Generator) -> pd.DataFrame:
    """Ground-truth degradation and speed profile for the replay (1 step = 1 h)."""
    t = np.arange(N_STEPS)
    wash_at = 560
    kmc = np.where(t < wash_at, 0.998 - 0.036 * (t / wash_at) ** 1.3, 0.992 - 0.004 * (t - wash_at) / (N_STEPS - wash_at))
    kmt = 0.999 - 0.011 * t / N_STEPS
    speeds = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27], dtype=float)
    v = np.empty(N_STEPS)
    i = 0
    while i < N_STEPS:
        # alternating port / manoeuvring / transit legs
        leg = rng.choice(["port", "coastal", "transit"], p=[0.2, 0.3, 0.5])
        dwell = int(rng.integers(6, 30))
        pool = {"port": speeds[:2], "coastal": speeds[2:5], "transit": speeds[4:]}[leg]
        v[i:i + dwell] = rng.choice(pool)
        i += dwell
    return pd.DataFrame({"step": t, "v": v, "kMc": np.round(kmc, 3), "kMt": np.round(kmt, 3), "wash": t == wash_at})


def fit_ensemble(X: pd.DataFrame, y: pd.Series) -> list:
    models = [
        lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=31, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=SEED, verbose=-1),
        ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-4, early_stopping=True,
                                                     max_iter=600, random_state=SEED)),
    ]
    y_mean, y_std = y.mean(), y.std()
    for m in models:
        m.fit(X, (y - y_mean) / y_std)
    return [models, y_mean, y_std]


def predict_ensemble(ens: list, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    models, y_mean, y_std = ens
    preds = np.stack([m.predict(X) * y_std + y_mean for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def episodes(flag: np.ndarray, persist: int = PERSIST) -> np.ndarray:
    """Debounced alert state: raise after `persist` consecutive True, clear after `persist` consecutive False."""
    state = np.zeros(len(flag), dtype=bool)
    on, run_t, run_f = False, 0, 0
    for i, f in enumerate(flag):
        run_t, run_f = (run_t + 1, 0) if f else (0, run_f + 1)
        if not on and run_t >= persist:
            on = True
        elif on and run_f >= persist:
            on = False
        state[i] = on
    return state


def count_raises(state: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(state & ~np.r_[False, state[:-1]])]


def run() -> dict:
    rng = np.random.default_rng(SEED)
    clean = load_marine()
    voyage = make_voyage(rng)

    # --- split: hold out all voyage grid cells -------------------------------
    cells = set(zip(voyage.kMc, voyage.kMt))
    in_voyage = np.array([(a, b) in cells for a, b in zip(clean.kMc, clean.kMt)])
    pool = add_noise(clean[~in_voyage], rng).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(pool)
    i1, i2, i3 = int(0.50 * n), int(0.65 * n), int(0.80 * n)
    train, sig, cal, test = pool[:i1], pool[i1:i2], pool[i2:i3], pool[i3:].copy()

    # voyage rows: exact grid lookup of the clean simulator point + fresh noise
    voy = voyage.merge(clean, on=["v", "kMc", "kMt"], how="left", validate="many_to_one")
    assert voy[MARINE_SENSORS].notna().all().all(), "voyage left the simulator grid"
    voy = add_noise(voy, rng)

    # healthy-reference fuel flow per speed -> fuel penalty of the true state
    healthy = clean[(clean.kMc == 1.0) & (clean.kMt == 1.0)].set_index("v")["mf"]
    clean_mf = voyage.merge(clean[["v", "kMc", "kMt", "mf"]], on=["v", "kMc", "kMt"], how="left")["mf"].to_numpy()
    voy["mf_healthy"] = voy["v"].map(healthy).to_numpy()
    voy["fuel_excess_pct"] = 100 * (clean_mf / voy["mf_healthy"] - 1)
    # below 9 kn the simulator is in a different control regime and the fuel comparison is not meaningful
    voy.loc[voy["v"] < 9, "fuel_excess_pct"] = np.nan

    metrics: dict = {
        "alpha": ALPHA, "thresholds": THRESHOLDS, "persist": PERSIST, "noise_rel": NOISE_REL,
        "n": {"train": len(train), "sigma_fit": len(sig), "calibration": len(cal), "test": len(test), "voyage": len(voy),
              "held_out_cells": len(cells)},
        "targets": {},
    }
    cal_scores = {}

    for target in ["kMc", "kMt"]:
        thr = THRESHOLDS[target]
        ens = fit_ensemble(train[MARINE_FEATURES], train[target])
        # difficulty model: expected |residual| learned on a disjoint split
        mu_s, sd_s = predict_ensemble(ens, sig[MARINE_FEATURES])
        sig_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                                      random_state=SEED, verbose=-1)
        feat_s = sig[MARINE_FEATURES].assign(ens_sd=sd_s)
        sig_model.fit(feat_s, np.abs(sig[target] - mu_s))
        floor = float(np.quantile(np.abs(sig[target] - mu_s), 0.05))

        def predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            mu, sd = predict_ensemble(ens, df[MARINE_FEATURES])
            sigma = np.maximum(sig_model.predict(df[MARINE_FEATURES].assign(ens_sd=sd)), floor)
            return mu, sigma

        mu_c, s_c = predict(cal)
        sc_split = cp.split_scores(cal[target], mu_c)
        sc_norm = cp.normalized_scores(cal[target], mu_c, s_c)
        cal_scores[f"{target}_split"] = sc_split
        cal_scores[f"{target}_norm"] = sc_norm
        q_split, q_norm = cp.conformal_quantile(sc_split, ALPHA), cp.conformal_quantile(sc_norm, ALPHA)

        tm: dict = {"q_split": q_split, "q_norm": q_norm, "sigma_floor": floor}
        for name, df in [("test", test), ("voyage", voy)]:
            mu, s = predict(df)
            df[f"{target}_mu"], df[f"{target}_sigma"] = mu, s
            df[f"{target}_lo_split"], df[f"{target}_hi_split"] = cp.interval_split(mu, q_split)
            df[f"{target}_lo"], df[f"{target}_hi"] = cp.interval_normalized(mu, s, q_norm)
            y = df[target].to_numpy()
            truth_bad = y < thr
            naive = mu < thr
            confirmed = df[f"{target}_hi"].to_numpy() < thr
            watch = (df[f"{target}_lo"].to_numpy() < thr) & ~confirmed
            df[f"{target}_naive"], df[f"{target}_confirmed"], df[f"{target}_watch"] = naive, confirmed, watch
            healthy_n, bad_n = int((~truth_bad).sum()), int(truth_bad.sum())
            res = {
                "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
                "coverage_split": cp.coverage(y, df[f"{target}_lo_split"], df[f"{target}_hi_split"]),
                "coverage_adaptive": cp.coverage(y, df[f"{target}_lo"], df[f"{target}_hi"]),
                "mean_width_split": float(2 * q_split),
                "mean_width_adaptive": float(np.mean(df[f"{target}_hi"] - df[f"{target}_lo"])),
                "median_width_adaptive": float(np.median(df[f"{target}_hi"] - df[f"{target}_lo"])),
                "n_healthy": healthy_n, "n_degraded": bad_n,
                # per-sample alert quality (no debouncing)
                "naive_false_alarm_rate": float(naive[~truth_bad].mean()) if healthy_n else None,
                "ua_false_alarm_rate": float(confirmed[~truth_bad].mean()) if healthy_n else None,
                "naive_detection_rate": float(naive[truth_bad].mean()) if bad_n else None,
                "ua_confirmed_rate_when_degraded": float(confirmed[truth_bad].mean()) if bad_n else None,
                "ua_watch_or_confirmed_when_degraded": float((confirmed | watch)[truth_bad].mean()) if bad_n else None,
                "ua_watch_rate_when_healthy": float(watch[~truth_bad].mean()) if healthy_n else None,
                "naive_precision": float(truth_bad[naive].mean()) if naive.any() else None,
                "ua_precision": float(truth_bad[confirmed].mean()) if confirmed.any() else None,
            }
            # conditional coverage / width by ship speed
            by_v = []
            for v, g in df.groupby("v"):
                yy = g[target].to_numpy()
                by_v.append({
                    "v": float(v), "n": len(g),
                    "coverage_split": cp.coverage(yy, g[f"{target}_lo_split"], g[f"{target}_hi_split"]),
                    "coverage_adaptive": cp.coverage(yy, g[f"{target}_lo"], g[f"{target}_hi"]),
                    "width_split": float(2 * q_split),
                    "width_adaptive": float(np.mean(g[f"{target}_hi"] - g[f"{target}_lo"])),
                    "mae": float(np.mean(np.abs(yy - g[f"{target}_mu"]))),
                })
            res["by_speed"] = by_v
            if name == "test":
                res["calibration_curve"] = [
                    {"nominal": 1 - a,
                     "split": cp.coverage(y, *cp.interval_split(mu, cp.conformal_quantile(sc_split, a))),
                     "adaptive": cp.coverage(y, *cp.interval_normalized(mu, s, cp.conformal_quantile(sc_norm, a)))}
                    for a in ALPHA_GRID
                ]
            else:
                # debounced alert episodes over the voyage timeline
                st_naive, st_conf, st_watch = episodes(naive), episodes(confirmed), episodes(confirmed | watch)
                df[f"{target}_naive_state"], df[f"{target}_confirmed_state"], df[f"{target}_watch_state"] = st_naive, st_conf, st_watch
                first_bad = int(np.flatnonzero(truth_bad)[0]) if bad_n else None

                def summarise(state: np.ndarray) -> dict:
                    raises = count_raises(state)
                    false = [r for r in raises if not truth_bad[r]]
                    true = [r for r in raises if truth_bad[r]]
                    return {
                        "raises": raises, "n_raises": len(raises), "n_false_raises": len(false),
                        "first_true_raise": true[0] if true else None,
                        "detection_delay_steps": (true[0] - first_bad) if (true and first_bad is not None) else None,
                        "steps_in_false_alarm": int((state & ~truth_bad).sum()),
                    }

                res["first_degraded_step"] = first_bad
                res["episodes"] = {"naive": summarise(st_naive), "ua_confirmed": summarise(st_conf), "ua_watch": summarise(st_watch)}
            tm[name] = res
        metrics["targets"][target] = tm

    ARTIFACTS.mkdir(exist_ok=True)
    test.to_csv(ARTIFACTS / "marine_test.csv", index=False)
    voy.to_csv(ARTIFACTS / "marine_voyage.csv", index=False)
    np.savez(ARTIFACTS / "marine_cal_scores.npz", **cal_scores)
    (ARTIFACTS / "marine_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = run()
    for tgt, tm in m["targets"].items():
        for split in ["test", "voyage"]:
            r = {k: v for k, v in tm[split].items() if k not in ("by_speed", "calibration_curve")}
            print(tgt, split, json.dumps(r, indent=1))
        print(pd.DataFrame(tm["test"]["by_speed"]).round(4).to_string())
        print(pd.DataFrame(tm["test"]["calibration_curve"]).round(3).to_string())
