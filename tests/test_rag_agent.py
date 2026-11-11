"""Tests for the knowledge base, retrieval, routing and alert explanation."""
import json
from pathlib import Path

import pytest
import requests

from uaagent.agent import (
    CSC_BASE_URL,
    CSC_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    LLMClient,
    build_query,
    explain,
    format_value,
    _extract_actions,
    _num,
)
from uaagent.rag import DISCLAIMER, Hit, KnowledgeBase
from uaagent.routing import operator_load

KB_DIR = Path(__file__).resolve().parents[1] / "kb"


@pytest.fixture(autouse=True)
def _isolate_llm_config(monkeypatch, tmp_path):
    """Hermetic LLM config for this module: no user env keys and no real
    ~/.aitta_token, so tests never resolve or call a real endpoint."""
    for var in ("UAA_LLM_BASE_URL", "UAA_LLM_API_KEY", "UAA_LLM_MODEL",
                "AITTA_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


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
    assert "water wash" in q2
    assert "fuel penalty" in q2


def test_query_malformed_speed_adds_no_confirmation():
    for bad in (None, "bad"):
        a = _alert("compressor_decay", "watch")
        a["context"]["speed_knots"] = bad
        assert "confirmation run" not in build_query(a)


def test_query_unknown_kind_confirmed_is_triage_only():
    q = build_query({"kind": "nope", "state": "confirmed", "context": {}})
    assert q == "alert triage escalation"


def test_query_compressor_watch_low_load_exact():
    q = build_query(_alert("compressor_decay", "watch"))
    assert q == (
        "compressor fouling decay kMc T2 P2 inlet filter symptoms "
        "early fouling signs monitoring watch handling "
        "sensor validation check before acting "
        "low load observability confirmation run"
    )


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_operator_load_modes():
    assert operator_load(0, 0, False) == (0.0, "deep_dive")
    assert operator_load(2, 1, False) == (0.39, "standard")
    score, mode = operator_load(3, 0, True)
    assert mode == "focus"
    assert abs(score - 0.66) < 1e-9
    assert operator_load(10, 10, True)[0] == 1.0  # capped


# --------------------------------------------------------------------------
# LLM configuration discovery (env > overrides > ~/.aitta_token)
# --------------------------------------------------------------------------

def test_env_key_defaults_to_csc(monkeypatch):
    monkeypatch.setenv("UAA_LLM_API_KEY", "dummy-env-key")
    client = LLMClient.from_env()
    assert client is not None
    assert client.api_key == "dummy-env-key"
    assert client.base_url == CSC_BASE_URL
    assert client.model == CSC_MODEL


def test_aitta_key_env_defaults_to_csc(monkeypatch):
    monkeypatch.setenv("AITTA_KEY", "dummy-aitta-key")
    client = LLMClient.from_env()
    assert client is not None
    assert client.base_url == CSC_BASE_URL
    assert client.model == CSC_MODEL


def test_explicit_endpoint_and_model_retained(tmp_path, monkeypatch):
    """Explicit keys are honored for custom endpoints even when a local
    CSC token file exists."""
    (tmp_path / ".aitta_token").write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("UAA_LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("UAA_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("UAA_LLM_MODEL", "local-model")
    client = LLMClient.from_env()
    assert client.base_url == "http://127.0.0.1:1234/v1"
    assert client.model == "local-model"
    assert client.api_key == "dummy-key"


def test_explicit_endpoint_without_key_retained(monkeypatch):
    monkeypatch.setenv("UAA_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("UAA_LLM_MODEL", "local-model")
    client = LLMClient.from_env()
    assert client is not None
    assert client.base_url == "http://127.0.0.1:1234/v1"
    assert client.model == "local-model"
    assert client.api_key == ""


def test_token_file_fallback(tmp_path):
    (tmp_path / ".aitta_token").write_text("  dummy-file-token\n",
                                         encoding="utf-8")
    client = LLMClient.from_env()
    assert client is not None
    assert client.api_key == "dummy-file-token"
    assert client.base_url == CSC_BASE_URL
    assert client.model == CSC_MODEL


def test_token_file_not_sent_to_custom_endpoint(tmp_path, monkeypatch):
    """The CSC token file is never attached to a non-CSC base URL."""
    (tmp_path / ".aitta_token").write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("UAA_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("UAA_LLM_MODEL", "local-model")
    client = LLMClient.from_env()
    assert client is not None
    assert client.base_url == "http://127.0.0.1:1234/v1"
    assert client.model == "local-model"
    assert client.api_key == ""


def test_token_file_used_when_base_is_csc(tmp_path, monkeypatch):
    (tmp_path / ".aitta_token").write_text("file-token", encoding="utf-8")
    monkeypatch.setenv("UAA_LLM_BASE_URL", CSC_BASE_URL + "/")
    client = LLMClient.from_env()
    assert client is not None
    assert client.base_url == CSC_BASE_URL
    assert client.api_key == "file-token"
    assert client.model == CSC_MODEL


def test_cross_name_key_precedence(monkeypatch):
    """Env beats overrides across different key names (AITTA vs UAA)."""
    monkeypatch.setenv("AITTA_KEY", "env-aitta")
    client = LLMClient.from_env(
        overrides={"UAA_LLM_API_KEY": "secret-uaa"})
    assert client is not None
    assert client.api_key == "env-aitta"
    monkeypatch.setenv("UAA_LLM_API_KEY", "env-uaa")
    client = LLMClient.from_env(
        overrides={"UAA_LLM_API_KEY": "secret-uaa"})
    assert client.api_key == "env-uaa"


def test_priority_env_over_overrides_over_file(tmp_path, monkeypatch):
    (tmp_path / ".aitta_token").write_text("file-token", encoding="utf-8")
    assert LLMClient.from_env().api_key == "file-token"
    overrides = {"UAA_LLM_API_KEY": "secret-token",
                 "UAA_LLM_BASE_URL": "http://secret/v1",
                 "UAA_LLM_MODEL": "secret-model"}
    client = LLMClient.from_env(overrides=overrides)
    assert client.api_key == "secret-token"
    assert client.base_url == "http://secret/v1"
    assert client.model == "secret-model"
    monkeypatch.setenv("UAA_LLM_API_KEY", "env-token")
    client = LLMClient.from_env(overrides=overrides)
    assert client.api_key == "env-token"
    assert client.base_url == "http://secret/v1"


def test_overrides_key_defaults_to_csc():
    client = LLMClient.from_env(overrides={"AITTA_KEY": "tok"})
    assert client is not None
    assert client.api_key == "tok"
    assert client.base_url == CSC_BASE_URL
    assert client.model == CSC_MODEL


def test_no_config_returns_none():
    """Unconfigured means no client — and from_env performs no network I/O."""
    assert LLMClient.from_env() is None
    assert LLMClient.from_env(overrides={}) is None


def test_unreadable_token_file_is_ignored(tmp_path):
    (tmp_path / ".aitta_token").mkdir()
    assert LLMClient.from_env() is None


def test_invalid_utf8_token_file_is_ignored(tmp_path):
    (tmp_path / ".aitta_token").write_bytes(b"\xff\xfe\x00\x01")
    assert LLMClient.from_env() is None


def test_llm_from_env_unconfigured(monkeypatch):
    for var in ("UAA_LLM_BASE_URL", "UAA_LLM_API_KEY", "UAA_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert LLMClient.from_env() is None


# --------------------------------------------------------------------------
# LLM path and fallback
# --------------------------------------------------------------------------

class _Resp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err


def _chat_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _capture_post(monkeypatch, responder):
    """Patch requests.post; return the list of captured call kwargs."""
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers,
                      "timeout": timeout})
        return responder()

    monkeypatch.setattr("uaagent.agent.requests.post", fake_post)
    return calls


def test_csc_chat_request_shape(monkeypatch):
    calls = _capture_post(
        monkeypatch, lambda: _Resp(_chat_payload('{"headline": "x"}')))
    LLMClient(CSC_BASE_URL, "dummy-token", CSC_MODEL).chat(
        [{"role": "user", "content": "hi"}])
    (call,) = calls
    assert call["url"] == f"{CSC_BASE_URL}/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer dummy-token"
    assert call["json"]["model"] == CSC_MODEL
    assert call["json"]["max_tokens"] == 2048
    assert call["json"]["reasoning_effort"] == "low"
    assert "response_format" not in call["json"]
    assert call["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_csc_other_model_omits_reasoning_effort(monkeypatch):
    """reasoning_effort is only sent for the gpt-oss advisory model."""
    calls = _capture_post(
        monkeypatch, lambda: _Resp(_chat_payload('{"headline": "x"}')))
    LLMClient(CSC_BASE_URL, "dummy-token", "other/model-7b").chat([])
    (call,) = calls
    assert "reasoning_effort" not in call["json"]
    assert "response_format" not in call["json"]


def test_generic_chat_request_keeps_json_mode(monkeypatch):
    calls = _capture_post(monkeypatch, lambda: _Resp(_chat_payload("{}")))
    LLMClient("http://127.0.0.1:1234/v1", "", "local-model").chat([])
    (call,) = calls
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["json"]["max_tokens"] == 2048
    assert "reasoning_effort" not in call["json"]
    assert "Authorization" not in call["headers"]


def _wrap_fence(content: str, variant: str) -> str:
    if variant == "bare":
        return content
    if variant == "json_fence":
        return f"```json\n{content}\n```"
    if variant == "plain_fence":
        return f"```\n{content}\n```"
    if variant == "spaced_json":
        return f"```json   {content}   ```"
    if variant == "open_only":
        return f"```json\n{content}"
    if variant == "outer_ws":
        return f"  ```json\n{content}\n```  "
    raise ValueError(variant)


@pytest.mark.parametrize(
    "variant",
    ["bare", "json_fence", "plain_fence", "spaced_json", "open_only",
     "outer_ws"])
def test_llm_advisory_success(kb, monkeypatch, variant):
    offline = explain(_alert("compressor_decay", "confirmed"), kb,
                      mode="standard")
    doc_id = offline.sources[0].doc_id
    advisory = {
        "headline": "Compressor fouling confirmed - schedule a water wash",
        "why": ["The whole interval sits below the wash threshold."],
        "actions": [{
            "text": "Schedule an online water wash per the service bulletin.",
            "source_ids": [doc_id],
        }],
    }
    _capture_post(monkeypatch,
                  lambda: _Resp(_chat_payload(
                      _wrap_fence(json.dumps(advisory), variant))))
    client = LLMClient(CSC_BASE_URL, "dummy-token", CSC_MODEL)
    ex = explain(_alert("compressor_decay", "confirmed"), kb,
                 mode="standard", llm=client)
    assert ex.generator == f"llm:{CSC_MODEL}"
    assert "gpt-oss" in ex.generator
    assert ex.fallback_reason is None
    assert ex.llm_latency_seconds is not None
    assert ex.llm_latency_seconds >= 0
    assert ex.actions[0]["source_ids"] == [doc_id]


def _raise_timeout():
    raise requests.Timeout("simulated connect timeout")


@pytest.mark.parametrize(
    "responder,reason",
    [
        pytest.param(_raise_timeout, "timeout", id="timeout"),
        pytest.param(lambda: _Resp(status=401), "authentication failed",
                     id="auth_401"),
        pytest.param(lambda: _Resp(status=429), "rate limited",
                     id="rate_limit_429"),
        pytest.param(lambda: _Resp(_chat_payload("{not json")),
                     "invalid response", id="malformed_json"),
        pytest.param(
            lambda: _Resp(_chat_payload(json.dumps({
                "headline": "Advisory",
                "why": ["Grounded bullet."],
                "actions": [{"text": "Do the thing.",
                             "source_ids": ["GHOST-DOC"]}],
            }))),
            "invalid response", id="unknown_citation"),
    ],
)
def test_llm_failures_fall_back_safely(kb, monkeypatch, responder, reason):
    _capture_post(monkeypatch, responder)
    client = LLMClient(CSC_BASE_URL, "dummy-token-secret", CSC_MODEL)
    alert = _alert("compressor_decay", "confirmed")
    ex = explain(alert, kb, mode="standard", llm=client)
    baseline = explain(alert, kb, mode="standard")
    assert ex.generator == "offline-extractive"
    assert ex.fallback_reason == reason
    assert ex.llm_latency_seconds is not None
    assert ex.llm_latency_seconds >= 0
    # the deterministic confidence statement is preserved verbatim
    assert ex.confidence_statement == baseline.confidence_statement
    # categorical reason only — no token, HTTP body or exception text
    rendered = json.dumps({
        "headline": ex.headline, "why": ex.why, "actions": ex.actions,
        "generator": ex.generator, "fallback_reason": ex.fallback_reason,
    })
    assert "dummy-token-secret" not in rendered
    assert "simulated connect timeout" not in rendered
    assert "GHOST-DOC" not in rendered


def test_llm_failure_falls_back(kb, monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("uaagent.agent.requests.post", boom)
    client = LLMClient("http://127.0.0.1:9", "dummy", "demo-model")
    ex = explain(_alert("compressor_decay", "confirmed"), kb,
                 mode="standard", llm=client)
    assert ex.generator == "offline-extractive"
    assert ex.fallback_reason == "request failed"


# --------------------------------------------------------------------------
# Sonar-targeted internals: _num / _extract_actions edge behavior
# --------------------------------------------------------------------------

def test_num_nan_missing_and_special_values():
    ctx = {"nan_f": float("nan"), "nan_s": "nan", "zero": 0,
           "inf": float("inf"), "bad": "abc"}
    assert _num(ctx, "nan_f") is None
    assert _num(ctx, "nan_s") is None
    assert _num(ctx, "missing") is None
    assert _num(ctx, "bad") is None
    assert _num(ctx, "zero") == 0.0
    assert _num(ctx, "inf") == float("inf")


def _hit(doc_id: str, doc_type: str, text: str) -> Hit:
    return Hit(doc_id=doc_id, title="t", doc_type=doc_type,
               domain="marine", section="", text=text, score=1.0)


def test_extract_actions_duplicate_lines_keep_first_source():
    line = "1. Inspect the inlet filter for signs of fouling."
    h1 = _hit("KB-A", "procedure", line)
    h2 = _hit("KB-B", "procedure", line)
    out = _extract_actions([h1, h2], 5, _alert("compressor_decay", "watch"))
    assert len(out) == 1
    assert out[0]["source_ids"] == ["KB-A"]


def test_extract_actions_stop_boundary_after_full_source():
    lines = "\n".join(
        f"{i}. Perform the scheduled inspection step number {i}." for i in range(25))
    h1 = _hit("KB-A", "procedure", lines)
    h2 = _hit("KB-B", "procedure",
              "1. Only-from-second-source maintenance action item.")
    out = _extract_actions([h1, h2], 40, _alert("compressor_decay", "watch"))
    texts = [a["text"] for a in out]
    assert len(texts) == 25  # 24th check fires only after the source ends
    assert not any("Only-from-second-source" in t for t in texts)
