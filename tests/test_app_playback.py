"""Regression tests for the live-replay scroll/flicker fixes.

Covers the playing-gated fragment scheduler, stable plotly chart
keys/config, the fixed-height feed column, and the slider/session-state
synchronisation that keeps stale widget state from rewinding the replay
step on unrelated reruns.
"""
import ast
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import common as C
import page_marine
from uaagent.agent import LLMClient

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """App tests must never reach a real endpoint or the user's token file
    (a real ~/.aitta_token may exist on a developer machine)."""
    monkeypatch.setattr(
        LLMClient, "from_env",
        classmethod(lambda cls, overrides=None: None))


def _run_page(mod_name, root):
    import sys
    import importlib
    for p in (root + "/src", root + "/app"):
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.import_module(mod_name).show()


def _at(mod_name):
    at = AppTest.from_function(
        _run_page, kwargs={"mod_name": mod_name, "root": str(ROOT)})
    at.run(timeout=180)
    assert not at.exception, f"{mod_name}: {at.exception}"
    return at


def _el(seq, key):
    return next(e for e in seq if e.key == key)


def _tree(mod_name):
    return ast.parse((APP / f"{mod_name}.py").read_text(encoding="utf-8"))


def _call_name(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def test_marine_play_pause_restart():
    at = _at("page_marine")
    assert at.session_state["m_step"] == 380
    assert not at.session_state["m_playing"]

    _el(at.button, "m_play").click().run()
    assert at.session_state["m_playing"]
    step = at.session_state["m_step"]
    at.run()
    assert at.session_state["m_step"] > step

    _el(at.button, "m_play").click().run()
    assert not at.session_state["m_playing"]
    frozen = at.session_state["m_step"]
    at.run()
    assert at.session_state["m_step"] == frozen

    _el(at.button, "m_restart").click().run()
    assert at.session_state["m_step"] == 0
    assert not at.session_state["m_playing"]


def test_marine_jump_to_first_alert_survives_rerun():
    at = _at("page_marine")
    _el(at.button, "m_first").click().run()
    expected = min(a["step"] for a in C.get_marine_alerts())
    assert at.session_state["m_step"] == expected
    at.run()
    assert at.session_state["m_step"] == expected


def test_marine_step_survives_playback_then_unrelated_rerun():
    at = _at("page_marine")
    at.session_state["m_playing"] = True
    at.run()
    at.run()
    at.session_state["m_playing"] = False
    step = at.session_state["m_step"]
    assert step > 380
    _el(at.sidebar.number_input, "m_fuel_price").set_value(900).run()
    assert at.session_state["m_step"] == step


def test_marine_slider_drag_during_playback_is_honored():
    at = _at("page_marine")
    at.session_state["m_playing"] = True
    at.run()
    at.run()
    _el(at.slider, "m_slider").set_value(600).run()
    assert at.session_state["m_step"] >= 600


def test_marine_playback_stops_at_end():
    at = _at("page_marine")
    at.session_state["m_playing"] = True
    at.session_state["m_step"] = 717
    at.run()
    assert not at.exception
    assert at.session_state["m_step"] == 719
    assert not at.session_state["m_playing"]
    at.run()
    assert at.session_state["m_step"] == 719


def test_energy_playback_advances_while_playing():
    at = _at("page_energy")
    at.session_state["e_playing"] = True
    at.run()
    step = at.session_state["e_step"]
    at.run()
    assert at.session_state["e_step"] > step


def test_energy_play_pause_restart():
    at = _at("page_energy")
    assert not at.session_state["e_playing"]

    _el(at.button, "e_play").click().run()
    assert at.session_state["e_playing"]
    step = at.session_state["e_step"]
    at.run()
    assert at.session_state["e_step"] > step

    _el(at.button, "e_play").click().run()
    assert not at.session_state["e_playing"]
    frozen = at.session_state["e_step"]
    at.run()
    assert at.session_state["e_step"] == frozen

    _el(at.button, "e_restart").click().run()
    assert at.session_state["e_step"] == 0
    assert not at.session_state["e_playing"]


def test_energy_step_survives_playback_then_unrelated_rerun():
    at = _at("page_energy")
    at.session_state["e_playing"] = True
    at.run()
    at.run()
    at.session_state["e_playing"] = False
    step = at.session_state["e_step"]
    assert step > 0
    _el(at.sidebar.slider, "e_extra").set_value(3).run()
    assert at.session_state["e_step"] == step


def test_energy_playback_stops_at_end():
    at = _at("page_energy")
    df_full = C.load_energy_test()
    dw = C.load_energy_metrics()["demo_window"]
    start = pd.Timestamp(dw["start"]).date()
    end = pd.Timestamp(dw["end"]).date()
    if (end - start).days > 31:
        end = start + pd.Timedelta(days=31)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df_full[(df_full["decision_time"] >= start_ts)
                 & (df_full["decision_time"] < end_ts)]
    max_step = len(df) - 1
    at.session_state["e_playing"] = True
    at.session_state["e_step"] = max_step - 2
    at.run()
    assert not at.exception
    assert at.session_state["e_step"] == max_step
    assert not at.session_state["e_playing"]
    at.run()
    assert at.session_state["e_step"] == max_step


def test_energy_window_shrink_clamps_step():
    at = _at("page_energy")
    df_full = C.load_energy_test()
    tmin = df_full["decision_time"].min().date()
    at.session_state["e_step"] = len(df_full) - 1
    _el(at.sidebar.date_input, "e_dates").set_value((tmin, tmin)).run()
    start_ts = pd.Timestamp(tmin, tz="UTC")
    end_ts = start_ts + pd.Timedelta(days=1)
    expected = df_full[(df_full["decision_time"] >= start_ts)
                       & (df_full["decision_time"] < end_ts)]
    assert 0 <= at.session_state["e_step"] <= len(expected) - 1


def test_main_fig_alert_markers_and_legend_invariants():
    voy = C.load_marine_voyage()
    metrics = C.load_marine_metrics()
    ua = C.get_marine_alerts()
    naive = C.get_naive_alerts()
    win = voy.iloc[: 601]
    fig = page_marine._main_fig(
        win, ua, naive, True, 560, metrics["thresholds"])
    xset = set(win["step"])

    def kind_prefix(a):
        return ("compressor" if a["kind"].startswith("compressor")
                else "turbine")

    n_ua = len({(kind_prefix(a), a["state"])
                for a in ua if a["step"] in xset})
    n_naive = len({kind_prefix(a) for a in naive if a["step"] in xset})
    nonempty = [t for t in fig.data if t.mode == "markers" and len(t.x) > 0]
    assert n_ua + n_naive > 0
    assert len(nonempty) == n_ua + n_naive
    names = {t.name for t in fig.data}
    assert {"UA watch", "UA confirmed", "conventional alarm",
            "ground truth (simulator)", "90% conformal band",
            "ensemble estimate"} <= names


def test_no_static_fragment_decorator_on_live_panels():
    for mod in ("page_marine", "page_energy"):
        for node in ast.walk(_tree(mod)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not any("fragment" in ast.dump(d)
                               for d in node.decorator_list), (
                    f"{mod}.{node.name} still declares a static "
                    "st.fragment decorator")


def test_fragment_interval_gated_by_playing_flag():
    for mod, flag in (("page_marine", "m_playing"),
                      ("page_energy", "e_playing")):
        hits = [n for n in ast.walk(_tree(mod))
                if isinstance(n, ast.Call)
                and _call_name(n) == "fragment"
                and any(k.arg == "run_every"
                        and isinstance(k.value, ast.IfExp)
                        and flag in ast.dump(k.value)
                        for k in n.keywords)]
        assert len(hits) == 1, (
            f"{mod}: expected one st.fragment(run_every=<cond on {flag}>) "
            f"call, found {len(hits)}")


def test_live_charts_have_stable_distinct_keys_and_scrollzoom_off():
    expected_calls = {"page_marine": 2, "page_energy": 3}
    for mod, count in expected_calls.items():
        calls = [n for n in ast.walk(_tree(mod))
                 if isinstance(n, ast.Call)
                 and _call_name(n) == "plotly_chart"]
        assert len(calls) == count, f"{mod}: {len(calls)} chart calls"
        literal_keys = []
        for n in calls:
            kw = {k.arg: k.value for k in n.keywords}
            assert "key" in kw, f"{mod}: plotly_chart missing key="
            assert "config" in kw, f"{mod}: plotly_chart missing config="
            assert ast.literal_eval(kw["config"]).get("scrollZoom") is False
            if isinstance(kw["key"], ast.Constant):
                literal_keys.append(ast.literal_eval(kw["key"]))
        assert len(literal_keys) == len(set(literal_keys)), (
            f"{mod}: duplicate literal chart keys {literal_keys}")


def test_live_figure_layouts_pin_uirevision():
    for mod in ("page_marine", "page_energy"):
        updates = [n for n in ast.walk(_tree(mod))
                   if isinstance(n, ast.Call)
                   and _call_name(n) == "update_layout"]
        assert updates, mod
        for n in updates:
            assert "uirevision" in {k.arg for k in n.keywords}, (
                f"{mod}: update_layout without uirevision")


def test_feed_column_uses_fixed_height_container():
    for mod, height in (("page_marine", 760), ("page_energy", 500)):
        hits = [n for n in ast.walk(_tree(mod))
                if isinstance(n, ast.Call)
                and _call_name(n) == "container"
                and any(k.arg == "height"
                        and isinstance(k.value, ast.Constant)
                        and k.value.value == height
                        for k in n.keywords)]
        assert hits, f"{mod}: no st.container(height={height}) found"


def test_slider_updates_step_via_callback_not_assignment():
    for mod, step_key, slider_key in (
            ("page_marine", "m_step", "m_slider"),
            ("page_energy", "e_step", "e_slider")):
        src = (APP / f"{mod}.py").read_text(encoding="utf-8")
        assert f"ss.{step_key} = c[" not in src, (
            f"{mod}: slider return value still assigned to {step_key}")
        tree = _tree(mod)
        sliders = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and _call_name(n) == "slider"
                   and any(k.arg == "key"
                           and isinstance(k.value, ast.Constant)
                           and k.value.value == slider_key
                           for k in n.keywords)]
        assert sliders, f"{mod}: no slider keyed {slider_key}"
        assert any("on_change" in {k.arg for k in n.keywords}
                   for n in sliders), (
            f"{mod}: {slider_key} slider lacks on_change callback")
