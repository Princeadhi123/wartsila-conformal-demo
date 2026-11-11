"""Page 2 — Marine console: live replay of the voyage with UA alerts."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import common as C

_TARGETS = {"kMc": "Compressor decay kMc", "kMt": "HP turbine decay kMt"}
_SENSORS = {"T48": "HP turbine exit temp T48, °C", "T2": "Compressor outlet temp T2, °C",
            "P2": "Compressor outlet pressure P2, bar", "mf": "Fuel flow, kg/s"}
_YRANGES = {"kMc": [0.955, 1.005], "kMt": [0.972, 1.004]}
_WINDOW = 240
_TRANSPARENT = "rgba(0,0,0,0)"
_GRID_COLOR = "rgba(255,255,255,0.08)"


def _state_at(row, target: str) -> str:
    if row[f"{target}_confirmed_state"]:
        return "confirmed"
    if row[f"{target}_watch_state"]:
        return "watch"
    return "normal"


def _spark(title: str, x, y, height: int = 150) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines",
        line={"color": C.COLORS["band_line"], "width": 1.4}))
    fig.update_layout(
        title={"text": title, "font": {"size": 12, "color": "#8FA6BC"}},
        height=height, margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT,
        font={"color": "#8FA6BC"}, showlegend=False,
        yaxis={"autorange": True}, uirevision=title,
    )
    fig.update_xaxes(gridcolor=_GRID_COLOR, showticklabels=False)
    fig.update_yaxes(gridcolor=_GRID_COLOR, nticks=3)
    return fig


def _add_alert_markers(fig: go.Figure, ua: list[dict], naive: list[dict],
                       target: str, row: int, xset: set,
                       legend_done: set) -> None:
    kind_prefix = "compressor" if target == "kMc" else "turbine"
    for state, col, lname in (
        ("watch", C.COLORS["watch"], "UA watch"),
        ("confirmed", C.COLORS["confirmed"], "UA confirmed"),
    ):
        pts = [a for a in ua if a["kind"].startswith(kind_prefix)
               and a["state"] == state and a["step"] in xset]
        if pts:
            fig.add_trace(
                go.Scatter(
                    x=[a["step"] for a in pts],
                    y=[a["estimate"] for a in pts], mode="markers",
                    marker={"symbol": "diamond", "size": 13, "color": col,
                            "line": {"color": "#0B1B2B", "width": 1.5}},
                    name=lname, legendgroup=lname,
                    showlegend=lname not in legend_done),
                row=row, col=1)
            legend_done.add(lname)
    pts = [a for a in naive if a["kind"].startswith(kind_prefix)
           and a["step"] in xset]
    if pts:
        fig.add_trace(
            go.Scatter(
                x=[a["step"] for a in pts],
                y=[a["estimate"] for a in pts], mode="markers",
                marker={"symbol": "triangle-up", "size": 13,
                        "color": C.COLORS["confirmed"],
                        "line": {"color": "#0B1B2B", "width": 1.5}},
                name="conventional alarm", legendgroup="naive",
                showlegend="naive" not in legend_done,
                hovertext="conventional alarm"),
            row=row, col=1)
        legend_done.add("naive")


def _main_fig(win: pd.DataFrame, ua: list[dict], naive: list[dict],
              truth: bool, wash_step: int | None,
              thresholds: dict) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
        subplot_titles=list(_TARGETS.values()),
    )
    x = list(win["step"])
    xset = set(x)
    legend_done = set()
    for i, target in enumerate(_TARGETS, start=1):
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(win[f"{target}_hi"]) + list(win[f"{target}_lo"])[::-1],
                fill="toself", fillcolor=C.COLORS["band"],
                line={"width": 0}, hoverinfo="skip",
                name="90% conformal band", legendgroup="band",
                showlegend="band" not in legend_done),
            row=i, col=1)
        legend_done.add("band")
        fig.add_trace(
            go.Scatter(x=x, y=win[f"{target}_mu"], mode="lines",
                       line={"color": C.COLORS["band_line"], "width": 1.2},
                       name="ensemble estimate", legendgroup="est",
                       showlegend="est" not in legend_done),
            row=i, col=1)
        legend_done.add("est")
        if truth:
            fig.add_trace(
                go.Scatter(x=x, y=win[target], mode="lines",
                           line={"color": C.COLORS["healthy"], "width": 1.5,
                                 "dash": "dot"},
                           name="ground truth (simulator)",
                           legendgroup="truth",
                           showlegend="truth" not in legend_done),
                row=i, col=1)
            legend_done.add("truth")
        fig.add_hline(y=thresholds[target], line_dash="dash",
                      line_color=C.COLORS["accent"], row=i, col=1)
        _add_alert_markers(fig, ua, naive, target, i, xset, legend_done)
    if wash_step is not None:
        for i in (1, 2):
            fig.add_vline(x=wash_step, line_dash="dot",
                          line_color=C.COLORS["healthy"], row=i, col=1)
    # legend swatches for marker types even when none are visible yet
    for lname, sym, col in (
        ("conventional alarm", "triangle-up", C.COLORS["confirmed"]),
        ("UA watch", "diamond", C.COLORS["watch"]),
        ("UA confirmed", "diamond", C.COLORS["confirmed"]),
    ):
        if lname not in legend_done:
            fig.add_trace(
                go.Scatter(x=[], y=[], mode="markers",
                           marker={"symbol": sym, "size": 13, "color": col,
                                   "line": {"color": "#0B1B2B", "width": 1.5}},
                           name=lname, showlegend=True),
                row=1, col=1)
    fig.update_layout(
        height=560, margin={"l": 10, "r": 10, "t": 95, "b": 10},
        paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT,
        font={"color": "#E8EEF4"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        uirevision="marine-main",
    )
    fig.update_xaxes(gridcolor=_GRID_COLOR)
    for i, target in enumerate(_TARGETS, start=1):
        fig.update_yaxes(gridcolor=_GRID_COLOR,
                         range=_YRANGES[target], row=i, col=1)
    return fig


def _live(voy, metrics, ua, naive, fuel_price, mode_override, extra):
    ss = st.session_state
    max_step = len(voy) - 1
    if ss.m_playing:
        ss.m_step = min(ss.m_step + ss.m_speed, max_step)
        if ss.m_step >= max_step:
            ss.m_playing = False
            st.rerun()
    step = int(ss.m_step)
    row = voy.iloc[step]

    states = {t: _state_at(row, t) for t in _TARGETS}
    active = sum(s != "normal" for s in states.values()) + extra
    new = sum(1 for a in ua if step - 24 < a["step"] <= step)
    crit = any(s == "confirmed" for s in states.values())
    score, mode = C.operator_load(active, new, crit)
    if mode_override != "Auto":
        mode = mode_override

    # ---- status strip ------------------------------------------------------
    fep = row["fuel_excess_pct"]
    pen = "–" if pd.isna(fep) else f"€{float(fep) / 100 * float(row['mf_healthy']) * 3600 * 24 / 1000 * fuel_price:,.0f}/day"
    st.markdown(
        f"**Hour {step}** · ship speed **{row['v']:.0f} kn** · "
        + C.state_chip(states["kMc"], "compressor ") + " "
        + C.state_chip(states["kMt"], "turbine ") + " "
        + C.chip(f"load {score:.2f} · {C.MODE_LABEL[mode]}",
                 C.COLORS["band_line"])
        + f" · est. fuel penalty **{pen}** *(illustrative)*",
        unsafe_allow_html=True,
    )
    st.caption(
        C.MODE_DESCRIPTIONS[mode]
        + f" (active {active}, new in 24 h {new}, critical {crit})"
    )

    left, right = st.columns([0.65, 0.35])
    with left:
        win = (voy.iloc[: step + 1] if ss.m_full
               else voy.iloc[max(0, step - _WINDOW + 1): step + 1])
        wash_step = 560 if step >= 560 else None
        st.plotly_chart(
            _main_fig(win, ua, naive, ss.m_truth, wash_step,
                      metrics["thresholds"]),
            use_container_width=True, key="m_main_chart",
            config={"scrollZoom": False},
        )
        st.caption(
            "The conformal band widens at low ship speed — the health state "
            "is less observable there."
        )
        cols = st.columns(4)
        for col, (sensor, label) in zip(cols, _SENSORS.items()):
            col.plotly_chart(_spark(label, win["step"], win[sensor]),
                             use_container_width=True,
                             key=f"m_spark_{sensor}",
                             config={"scrollZoom": False})
    with right:
        with st.container(height=760):
            st.markdown("**Conventional alarms (point estimate)**")
            C.render_naive_feed(naive, step, ss.m_truth)
            st.markdown("**Uncertainty-aware agent**")
            sel = C.render_ua_feed(ua, step, "m_alert_sel")
            st.divider()
            st.markdown("**Agent advisory**")
            if sel is None:
                st.caption("Select an alert once raised to see the advisory.")
            else:
                C.render_advisory(sel, mode)


def _on_slider() -> None:
    st.session_state.m_step = st.session_state.m_slider


def show() -> None:
    st.title("Marine console — propulsion health, live replay")
    st.caption(
        "A frigate propulsion plant replays 720 operating hours. The agent "
        "estimates compressor (kMc) and turbine (kMt) decay with calibrated "
        "90% intervals and only 'confirms' when the whole interval is beyond "
        "the action threshold."
    )
    voy = C.load_marine_voyage()
    metrics = C.load_marine_metrics()
    ua = C.get_marine_alerts()
    naive = C.get_naive_alerts()

    ss = st.session_state
    ss.setdefault("m_step", 380)
    ss.setdefault("m_playing", False)
    ss.setdefault("m_speed", 4)
    ss.setdefault("m_truth", False)
    ss.setdefault("m_full", False)
    ss.setdefault("m_extra", 0)
    if not ss.get("m_qp_done"):
        ss.m_qp_done = True
        qp = st.query_params
        if qp.get("hour"):
            ss.m_step = max(0, min(len(voy) - 1, int(qp["hour"])))
        if qp.get("truth") == "1":
            ss.m_truth = True
        if qp.get("extra"):
            ss.m_extra = int(qp["extra"])

    with st.sidebar:
        st.header("Console settings")
        fuel_price = st.number_input(
            "Fuel price (EUR/t, illustrative)", 100, 2000, 700, 50,
            key="m_fuel_price")
        mode_override = st.selectbox(
            "Advisory mode", ["Auto", "Focus", "Standard", "Deep dive"],
            key="m_mode")
        extra = st.slider(
            "Simulated additional concurrent alerts from other assets",
            0, 6, key="m_extra")

    mode_override = {"Focus": "focus", "Standard": "standard",
                     "Deep dive": "deep_dive"}.get(mode_override, "Auto")

    c = st.columns([1, 1, 1.4, 1.4, 3, 1.6, 1.8])
    if c[0].button("Pause" if ss.m_playing else "Play", key="m_play"):
        ss.m_playing = not ss.m_playing
    if c[1].button("Restart", key="m_restart"):
        ss.m_step, ss.m_playing = 0, False
    if c[2].button("Jump to first alert", key="m_first"):
        ss.m_step = min(a["step"] for a in ua)
    if c[3].button("Jump to wash", key="m_wash"):
        ss.m_step = 560
    ss.m_step = max(0, min(int(ss.m_step), len(voy) - 1))
    ss.m_slider = int(ss.m_step)
    c[4].slider("Voyage hour", 0, len(voy) - 1,
                key="m_slider", on_change=_on_slider)
    ss.m_speed = c[5].selectbox("h / tick", [1, 4, 12],
                                index=[1, 4, 12].index(ss.m_speed),
                                key="m_speed_sel")
    with c[6]:
        ss.m_truth = st.toggle("Reveal ground truth (simulator)",
                               value=ss.m_truth, key="m_truth_t")
        ss.m_full = st.toggle("Full history", value=ss.m_full,
                              key="m_full_t")

    st.fragment(run_every=1.0 if ss.m_playing else None)(_live)(
        voy, metrics, ua, naive, fuel_price, mode_override, extra)
    C.footer()
