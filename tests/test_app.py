"""Smoke test: every app page renders without exceptions."""
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

import common
from uaagent.agent import LLMClient

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """App tests must never reach a real endpoint or the user's token file
    (a real ~/.aitta_token may exist on a developer machine)."""
    monkeypatch.setattr(
        LLMClient, "from_env",
        classmethod(lambda cls, overrides=None: None))


def _run_page(mod_name, root):
    # AppTest.from_function extracts this function's source into a temp
    # script, so it must be self-contained: all imports inside the body.
    import sys
    import importlib
    for p in (root + "/src", root + "/app"):
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.import_module(mod_name).show()


def _run_advisory(root):
    # Self-contained: AppTest.from_function extracts this function's source.
    import sys
    for p in (root + "/src", root + "/app"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import common as C

    alert = {
        "id": "A-test", "domain": "marine", "kind": "compressor_decay",
        "state": "confirmed", "t": "h050", "step": 50,
        "estimate": 0.968, "lo": 0.962, "hi": 0.971, "threshold": 0.975,
        "coverage": 0.9, "unit": "",
        "context": {"speed_knots": 18.0, "fuel_flow_kg_s": 1.31,
                    "T2_C": 585.0, "P2_bar": 8.9},
    }
    C.render_advisory(alert, "standard")


@pytest.mark.parametrize("page", [
    "page_overview", "page_marine", "page_energy", "page_trust", "page_pilot",
])
def test_page_renders(page):
    at = AppTest.from_function(
        _run_page, kwargs={"mod_name": page, "root": str(ROOT)})
    at.run(timeout=180)
    assert not at.exception, f"{page}: {at.exception}"


def test_full_app_runs():
    """The real entry point (st.navigation) renders its default page.

    at.switch_page only supports file-based pages/* multipage apps, not
    st.Page(function) navigation, so the other pages are covered by the
    per-page wrapper tests above.
    """
    at = AppTest.from_file(str(ROOT / "app" / "app.py"))
    at.run(timeout=180)
    assert not at.exception, at.exception


class _TimeoutLLM(LLMClient):
    def chat(self, messages, timeout=None):
        raise requests.Timeout("simulated connect timeout")


def test_advisory_fallback_warning_and_caption(monkeypatch):
    """A failed configured model warns with a safe reason; the caption
    keeps the measured request time and the cached-timing label."""
    monkeypatch.setattr(
        common, "get_llm_client",
        lambda: _TimeoutLLM("https://aitta-api.csc.fi/openai/v1",
                            "dummy-token", "openai/gpt-oss-120b"))
    at = AppTest.from_function(_run_advisory, kwargs={"root": str(ROOT)})
    at.run(timeout=180)
    assert not at.exception, at.exception
    warnings = [w.value for w in at.warning]
    assert any("unavailable" in w and "timeout" in w for w in warnings)
    assert not any("simulated" in w or "dummy-token" in w
                   for w in warnings)
    captions = [c.value for c in at.caption]
    assert any(
        "Generator: offline-extractive" in c
        and "original request; cached results reuse this timing" in c
        for c in captions
    )
