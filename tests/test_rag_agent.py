"""Tests for the knowledge base, retrieval, routing and alert explanation."""
from pathlib import Path

import pytest

from uaagent.agent import LLMClient, build_query, explain, format_value
from uaagent.rag import DISCLAIMER, KnowledgeBase
from uaagent.routing import operator_load

KB_DIR = Path(__file__).resolve().parents[1] / "kb"


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase(KB_DIR)


def _alert(kind: str, state: str) -> dict:
    base = {
        "id": f"A-{kind}-{state}",
        "kind": kind,
        "state": state,
        "t": "2025-11-10T14:32:00Z",
        "coverage": 0.9,
    }
    if kind == "compressor_decay":
        base.update(
            domain="marine", unit="",
            estimate=0.968 if state == "confirmed" else 0.976,
            lo=0.962 if state == "confirmed" else 0.970,
            hi=0.971 if state == "confirmed" else 0.982,
            threshold=0.975,
            context={
                "speed_knots": 18.0 if state == "confirmed" else 9.0,
                "fuel_flow_kg_s": 1.31, "T48_C": 548.0, "T2_C": 585.0,
                "P2_bar": 8.9, "interval_width": 0.009, "typical_width": 0.008,
            },
        )
    elif kind == "turbine_decay":
        base.update(
            domain="marine", unit="",
            estimate=0.980 if state == "confirmed" else 0.985,
            lo=0.977 if state == "confirmed" else 0.981,
            hi=0.983 if state == "confirmed" else 0.989,
            threshold=0.985,
            context={
                "speed_knots": 21.0, "fuel_flow_kg_s": 1.42, "T48_C": 571.0,
                "interval_width": 0.006, "typical_width": 0.007,
            },
        )
    elif kind == "reserve_shortfall_risk":
        committed = 8866.0
        upper = 8950.0 if state == "confirmed" else 8900.0
        base.update(
            domain="energy", unit="MW",
            estimate=8200.0, lo=7700.0, hi=8700.0,
            threshold=committed,
            context={
                "forecast_mw": 8200.0, "lo_mw": 7700.0, "hi_mw": 8700.0,
                "upper_mw": upper, "upper85_mw": 8890.0,
                "committed_mw": committed,
                "standing_reserve_mw": 666.0,
                "recommended_reserve_mw": 750.0,
                "margin_mw": committed - upper,
                "extra_units": 5, "unit_mw": 20.0,
                "wind_mw": 1200.0, "horizon_h": 3.0,
                "interval_width": 1000.0, "typical_width": 950.0,
            },
        )
    else:  # forecast_uncertainty_high
        base.update(
            domain="energy", unit="MW",
            estimate=8300.0, lo=7700.0, hi=8950.0,
            threshold=8866.0,
            context={
                "forecast_mw": 8300.0, "lo_mw": 7700.0, "hi_mw": 8950.0,
                "upper_mw": 8950.0, "upper85_mw": 8800.0,
                "committed_mw": 8866.0,
                "standing_reserve_mw": 666.0,
                "recommended_reserve_mw": 1250.0,
                "margin_mw": -84.0,
                "extra_units": 5, "unit_mw": 20.0,
                "wind_mw": 1500.0, "horizon_h": 3.0,
                "interval_width": 1250.0, "typical_width": 780.0,
            },
        )
    return base


ALL_CASES = [
    (kind, state)
    for kind in ("compressor_decay", "turbine_decay",
                 "reserve_shortfall_risk", "forecast_uncertainty_high")
    for state in ("confirmed", "watch")
]


# --------------------------------------------------------------------------
# Knowledge base loading
# --------------------------------------------------------------------------

def test_kb_loads_enough_docs(kb):
    assert kb.n_docs >= 20
    assert kb.n_chunks >= kb.n_docs


def test_every_doc_has_disclaimer():
    for path in sorted(KB_DIR.glob("*.md")):
        assert DISCLAIMER in path.read_text(encoding="utf-8"), path.name


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def test_retrieve_compressor_query(kb):
    hits = kb.retrieve("compressor fouling water wash", k=4)
    assert hits, "no hits"
    top = hits[0]
    assert top.domain == "marine"
    doc = kb.get_doc(top.doc_id)
    assert doc["system"] == "gas_turbine_compressor"


def test_domain_filter(kb):
    hits = kb.retrieve("water wash decay threshold", k=6, domain="energy")
    assert all(h.domain == "energy" for h in hits)


def test_doc_type_filter_and_doc_cap(kb):
    hits = kb.retrieve("water wash", k=10, doc_types=["maintenance_log"])
    assert hits
    assert all(h.doc_type == "maintenance_log" for h in hits)
    assert len({h.doc_id for h in hits}) * 2 >= len(hits)  # <=2 chunks per doc


# --------------------------------------------------------------------------
# explain() across all kinds and states
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind,state", ALL_CASES)
def test_explain_all_cases(kb, kind, state):
    alert = _alert(kind, state)
    ex = explain(alert, kb, mode="standard")
    assert ex.generator == "offline-extractive"
    assert ex.headline
    assert ex.actions, f"no actions for {kind}/{state}"
    for a in ex.actions:
        assert a["text"].strip()
        assert a["source_ids"], "action without citation"
        for sid in a["source_ids"]:
            assert sid in kb.docs, f"unknown source id {sid}"
    st = ex.confidence_statement
    if alert["domain"] == "marine":
        for v in (alert["lo"], alert["hi"], alert["threshold"]):
            assert format_value(v, "marine") in st
    elif alert["kind"] == "reserve_shortfall_risk":
        assert format_value(alert["context"]["upper_mw"], "energy") in st
        assert format_value(alert["context"]["committed_mw"], "energy") in st
    else:
        assert format_value(alert["context"]["lo_mw"], "energy") in st
        assert format_value(alert["context"]["hi_mw"], "energy") in st


def test_energy_statements(kb):
    ex = explain(_alert("reserve_shortfall_risk", "watch"), kb, mode="standard")
    assert "stays below" in ex.confidence_statement
    assert "MW short" in ex.confidence_statement
    ex = explain(_alert("reserve_shortfall_risk", "confirmed"), kb,
                 mode="standard")
    assert "85% bound" in ex.confidence_statement
    ex = explain(_alert("forecast_uncertainty_high", "watch"), kb,
                 mode="standard")
    assert "[7,700, 8,950] MW" in ex.confidence_statement
    assert "1,250 MW wide" in ex.confidence_statement


def test_focus_is_terser_than_deep_dive(kb):
    alert = _alert("compressor_decay", "confirmed")
    focus = explain(alert, kb, mode="focus")
    deep = explain(alert, kb, mode="deep_dive")
    assert len(focus.actions) < len(deep.actions)
    assert len(focus.why) <= len(deep.why)
    assert len(deep.sources) <= 5
    assert any(h.doc_type == "maintenance_log" for h in deep.sources)


def test_query_mentions_confirmation_run_at_low_speed():
    q = build_query(_alert("compressor_decay", "watch"))
    assert "confirmation run" in q
    q2 = build_query(_alert("compressor_decay", "confirmed"))
    assert "water wash" in q2 and "fuel penalty" in q2


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_operator_load_modes():
    assert operator_load(0, 0, False) == (0.0, "deep_dive")
    assert operator_load(2, 1, False) == (0.39, "standard")
    score, mode = operator_load(3, 0, True)
    assert mode == "focus" and abs(score - 0.66) < 1e-9
    assert operator_load(10, 10, True)[0] == 1.0  # capped


# --------------------------------------------------------------------------
# LLM path and fallback
# --------------------------------------------------------------------------

def test_llm_failure_falls_back(kb, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("uaagent.agent.requests.post", boom)
    client = LLMClient("http://127.0.0.1:9", "dummy", "demo-model")
    ex = explain(_alert("compressor_decay", "confirmed"), kb,
                 mode="standard", llm=client)
    assert ex.generator == "offline-extractive"


def test_llm_from_env_unconfigured(monkeypatch):
    for var in ("UAA_LLM_BASE_URL", "UAA_LLM_API_KEY", "UAA_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert LLMClient.from_env() is None
