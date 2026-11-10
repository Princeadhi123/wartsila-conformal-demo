"""Alerts built from the real artifacts conform to the alert schema."""
import json
from pathlib import Path

import pandas as pd
import pytest

import numpy as np

from uaagent.agent import explain
from uaagent.alerts import (
    energy_alerts,
    interval_stats_at_level,
    marine_alerts,
    marine_naive_alerts,
)
from uaagent.rag import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

REQUIRED_KEYS = {"id", "domain", "kind", "state", "t", "estimate", "lo", "hi",
                 "threshold", "coverage", "unit", "context", "step", "source"}


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(ROOT / "kb")


@pytest.fixture(scope="module")
def marine():
    voy = pd.read_csv(ART / "marine_voyage.csv")
    metrics = json.loads((ART / "marine_metrics.json").read_text())
    return voy, metrics


@pytest.fixture(scope="module")
def energy():
    df = pd.read_csv(ART / "energy_test.csv")
    metrics = json.loads((ART / "energy_metrics.json").read_text())
    return df, metrics


def test_marine_alert_schema_and_counts(marine):
    voy, metrics = marine
    alerts = marine_alerts(voy, metrics)
    assert alerts
    for a in alerts:
        assert REQUIRED_KEYS <= set(a), a["id"]
        assert a["domain"] == "marine"
        assert a["state"] in ("watch", "confirmed")
        assert a["lo"] <= a["estimate"] <= a["hi"]
    expected = {
        "kMc": metrics["targets"]["kMc"]["voyage"]["episodes"]["ua_confirmed"]["n_raises"],
        "kMt": metrics["targets"]["kMt"]["voyage"]["episodes"]["ua_confirmed"]["n_raises"],
    }
    for target, n in expected.items():
        kind = {"kMc": "compressor_decay", "kMt": "turbine_decay"}[target]
        got = sum(1 for a in alerts if a["kind"] == kind and a["state"] == "confirmed")
        assert got == n, f"{kind}: expected {n} confirmed raises, got {got}"


def test_marine_naive_alerts(marine):
    voy, metrics = marine
    naive = marine_naive_alerts(voy, metrics)
    assert naive
    expected = sum(
        metrics["targets"][t]["voyage"]["episodes"]["naive"]["n_raises"]
        for t in ("kMc", "kMt")
    )
    assert len(naive) == expected
    assert all("is_false" in a for a in naive)


def test_energy_alert_schema(energy):
    df, metrics = energy
    alerts = energy_alerts(df, metrics)
    assert alerts
    kinds = {a["kind"] for a in alerts}
    assert kinds == {"reserve_shortfall_risk", "forecast_uncertainty_high"}
    for a in alerts[:50]:
        assert REQUIRED_KEYS <= set(a), a["id"]
        assert a["domain"] == "energy" and a["unit"] == "MW"
        ctx = a["context"]
        for k in ("forecast_mw", "upper_mw", "upper85_mw", "committed_mw",
                  "standing_reserve_mw", "recommended_reserve_mw",
                  "margin_mw", "extra_units", "unit_mw"):
            assert k in ctx, f"{a['id']} missing context.{k}"


def test_interval_stats_matches_metrics(marine):
    """Acceptance: at nominal 0.90 the recomputed stats equal metrics JSON."""
    _, metrics = marine
    test = pd.read_csv(ART / "marine_test.csv")
    scores = np.load(ART / "marine_cal_scores.npz")
    for t in ("kMc", "kMt"):
        m = metrics["targets"][t]["test"]
        s = interval_stats_at_level(
            test, t, scores[f"{t}_norm"], metrics["thresholds"][t], 0.90)
        assert s["coverage"] == pytest.approx(m["coverage_adaptive"], abs=1e-9)
        assert s["false_alarm_rate"] == pytest.approx(
            m["ua_false_alarm_rate"], abs=1e-9)
        assert s["confirmed_detection"] == pytest.approx(
            m["ua_confirmed_rate_when_degraded"], abs=1e-9)


def test_every_alert_explains(kb, marine, energy):
    voy, mm = marine
    df, em = energy
    alerts = marine_alerts(voy, mm) + energy_alerts(df, em)
    for a in alerts:
        ex = explain(a, kb, mode="standard")
        assert ex.headline and ex.confidence_statement
        assert ex.actions, f"no actions for {a['id']}"
        for act in ex.actions:
            for sid in act["source_ids"]:
                assert sid in kb.docs
