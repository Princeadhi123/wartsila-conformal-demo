"""Page 3 — Energy console: residual-load forecasting replay."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as C
from uaagent.marine import count_raises, episodes

_MAX_WINDOW_DAYS = 31
_TRAIL_H = 96


def _main_fig(df: pd.DataFrame, alerts: list[dict], step: int) -> go.Figure:
    lo_i = max(0, step - _TRAIL_H + 1)
    shown = df.iloc[lo_i: step + 1]
    fc = df.iloc[lo_i: min(step + 2, len(df))]  # + next forecast horizon
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(fc["target_time"]) + list(fc["target_time"])[::-1],
        y=list(fc["hi"]) + list(fc["lo"])[::-1],
        fill="toself", fillcolor=C.COLORS["band"], line=dict(width=0),
        hoverinfo="skip", name="90% adaptive interval"))
    fig.add_trace(go.Scatter(x=shown["target_time"], y=shown["y"],
                           mode="lines",
                           line=dict(color="#E8EEF4", width=1.2),
                           name="actual residual load"))
    fig.add_trace(go.Scatter(x=fc["target_time"], y=fc["mid"], mode="lines",
                           line=dict(color=C.COLORS["band_line"], width=1.8),
                           name="median forecast"))
    fig.add_trace(go.Scatter(x=fc["target_time"], y=fc["upper95"],
                           mode="lines",
                           line=dict(color=C.COLORS["watch"], width=1.4,
                                     dash="dot"),
                           name="95% upper bound"))
    fig.add_trace(go.Scatter(x=fc["target_time"],
                           y=fc["mid"] + fc["reserve_ref"], mode="lines",
                           line=dict(color=C.COLORS["confirmed"], width=1.4,
                                     dash="dash"),
                           name="committed (forecast + standing reserve)"))
    xlo, xhi = shown["target_time"].min(), fc["target_time"].max()
    legend_done = set()
    for a in alerts:
        if a["step"] > step:
            break
        row = df.iloc[a["step"]]
        if row["target_time"] < xlo:
            continue
        col = (C.COLORS["confirmed"] if a["state"] == "confirmed"
               else C.COLORS["watch"])
        lname = ("UA confirmed" if a["state"] == "confirmed" else "UA watch")
        fig.add_trace(go.Scatter(
            x=[row["target_time"]], y=[row["upper95"]], mode="markers",
            marker=dict(symbol="diamond", size=12, color=col,
                        line=dict(color="#0B1B2B", width=1.5)),
            name=lname, legendgroup=lname,
            showlegend=lname not in legend_done,
            hovertext=f"{C.KIND_LABEL.get(a['kind'], a['kind'])} · {a['state']}"))
        legend_done.add(lname)
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="MW", xaxis_range=[xlo, xhi],
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def _reserve_fig(df: pd.DataFrame, step: int) -> go.Figure:
    shown = df.iloc[max(0, step - _TRAIL_H + 1): step + 1]
    x = shown["target_time"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=shown["reserve_dynamic"], mode="lines",
                           line=dict(color=C.COLORS["band_line"], width=1.8),
                           name="recommended (conformal dynamic)"))
    fig.add_trace(go.Scatter(x=x, y=shown["reserve_ref"], mode="lines",
                           line=dict(color=C.COLORS["watch"], width=1.4),
                           name="standing reserve"))
    fig.add_trace(go.Scatter(x=x, y=shown["reserve_static"], mode="lines",
                           line=dict(color=C.COLORS["muted"], width=1.4,
                                     dash="dash"),
                           name="fixed rule"))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="reserve MW",
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


def _rollcov_fig(df_full: pd.DataFrame) -> go.Figure:
    x = df_full["target_time"]
    fig = go.Figure()
    for col, name, color in (
        ("rollcov_raw", "un-calibrated quantile band", C.COLORS["confirmed"]),
        ("rollcov_static", "static CQR", C.COLORS["watch"]),
        ("rollcov_aci", "adaptive conformal (ACI)", C.COLORS["healthy"]),
    ):
        fig.add_trace(go.Scatter(x=x, y=df_full[col], mode="lines",
                               line=dict(color=color, width=1.4), name=name))
    fig.add_hline(y=0.9, line_dash="dash", line_color="#E8EEF4",
                  annotation_text="stated 90%")
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8EEF4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="rolling 30-day coverage",
        yaxis_range=[0.5, 1.0],
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


@st.fragment(run_every=1.0)
def _live(df, alerts, metrics, mode_override, extra):
    ss = st.session_state
    max_step = len(df) - 1
    if ss.e_playing:
        ss.e_step = min(ss.e_step + ss.e_speed, max_step)
        if ss.e_step >= max_step:
            ss.e_playing = False
    step = min(int(ss.e_step), max_step)
    row = df.iloc[step]
    unit_mw = float(metrics["unit_mw"])

    watch_state = episodes(df["risk_watch"].to_numpy(dtype=bool), persist=2)
    conf_state = episodes(df["risk_confirmed"].to_numpy(dtype=bool), persist=2)
    unc_state = episodes(df["uncertainty_high"].to_numpy(dtype=bool), persist=2)
    active = int(watch_state[step]) + int(unc_state[step]) + extra
    new = sum(1 for a in alerts if step - 24 < a["step"] <= step)
    score, mode = C.operator_load(active, new, bool(conf_state[step]))
    if mode_override != "Auto":
        mode = mode_override

    units = math.ceil(float(row["reserve_dynamic"]) / unit_mw)
    st.markdown(
        f"**Target {row['target_time']:%Y-%m-%d %H:%M UTC}** · forecast "
        f"**{row['mid']:,.0f} MW** · 90% interval "
        f"**[{row['lo']:,.0f}, {row['hi']:,.0f}]** · recommended reserve "
        f"**{row['reserve_dynamic']:,.0f} MW** (≈ {units} × {unit_mw:.0f} MW "
        f"fast-start units, illustrative) · standing reserve "
        f"**{row['reserve_ref']:,.0f} MW** · wind now "
        f"**{row['wind_now']:,.0f} MW** · "
        + C.chip(f"load {score:.2f} · {C.MODE_LABEL[mode]}",
                 C.COLORS["band_line"]),
        unsafe_allow_html=True,
    )
    st.caption(
        C.MODE_DESCRIPTIONS[mode]
        + f" (active {active}, new in 24 h {new}, critical "
        + str(bool(conf_state[step])) + ")"
    )

    st.markdown("**Residual load and forecast interval**")
    st.plotly_chart(_main_fig(df, alerts, step), use_container_width=True)

    left, right = st.columns([0.6, 0.4])
    with left:
        st.markdown("**Reserve: recommended vs standing vs fixed rule**")
        st.plotly_chart(_reserve_fig(df, step), use_container_width=True)
    with right:
        st.markdown("**Uncertainty-aware agent**")
        sel = C.render_ua_feed(alerts, step, "e_alert_sel", max_items=6)
        st.divider()
        st.markdown("**Agent advisory**")
        if sel is None:
            st.caption("Select an alert once raised to see the advisory.")
        else:
            C.render_advisory(sel, mode)


def show() -> None:
    st.title("Energy console — residual-load forecast replay")
    st.caption(
        "Hourly 3 h-ahead residual-load forecasts for Finland. The agent "
        "flags reserve-shortfall risk when the 95% upper bound exceeds "
        "committed capacity, and flags hours where the forecast interval "
        "is unusually wide."
    )
    df_full = C.load_energy_test()
    metrics = C.load_energy_metrics()
    ss = st.session_state

    tmin = df_full["decision_time"].min()
    tmax = df_full["decision_time"].max()
    dw = metrics["demo_window"]
    d_start = pd.Timestamp(dw["start"])
    d_end = pd.Timestamp(dw["end"])
    if d_start.tzinfo is None:
        d_start = d_start.tz_localize("UTC")
        d_end = d_end.tz_localize("UTC")

    with st.sidebar:
        st.header("Replay window")
        picked = st.date_input(
            f"Date range (max {_MAX_WINDOW_DAYS} days)",
            value=(d_start.date(), d_end.date()),
            min_value=tmin.date(), max_value=tmax.date(), key="e_dates")
        mode_override = st.selectbox(
            "Advisory mode", ["Auto", "Focus", "Standard", "Deep dive"],
            key="e_mode")
        extra = st.slider(
            "Simulated additional concurrent alerts from other assets",
            0, 6, key="e_extra")

    mode_override = {"Focus": "focus", "Standard": "standard",
                     "Deep dive": "deep_dive"}.get(mode_override, "Auto")

    if isinstance(picked, tuple) and len(picked) == 2:
        start, end = picked
    else:
        start, end = d_start.date(), d_end.date()
    if (end - start).days > _MAX_WINDOW_DAYS:
        end = start + pd.Timedelta(days=_MAX_WINDOW_DAYS)
        st.caption(f"Window capped at {_MAX_WINDOW_DAYS} days.")
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df_full[(df_full["decision_time"] >= start_ts)
                 & (df_full["decision_time"] < end_ts)].reset_index(drop=True)
    if df.empty:
        st.warning("No data in the selected window.")
        C.footer()
        return
    alerts = C.get_energy_alerts(df)

    # Default position: 12 h before the worst fixed-rule shortfall so the
    # event unfolds when Play is pressed.
    default_step = max(0, int(df["shortfall_static"].to_numpy().argmax()) - 12)
    ss.setdefault("e_step", default_step)
    ss.setdefault("e_playing", False)
    ss.setdefault("e_speed", 24)
    ss.setdefault("e_extra", 0)
    if not ss.get("e_qp_done"):
        ss.e_qp_done = True
        qp = st.query_params
        if qp.get("hour"):
            ss.e_step = max(0, min(len(df) - 1, int(qp["hour"])))
        if qp.get("extra"):
            ss.e_extra = int(qp["extra"])

    c = st.columns([1, 1, 3, 1.6])
    if c[0].button("Pause" if ss.e_playing else "Play", key="e_play"):
        ss.e_playing = not ss.e_playing
    if c[1].button("Restart", key="e_restart"):
        ss.e_step, ss.e_playing = 0, False
    ss.e_step = min(
        c[2].slider("Replay hour", 0, len(df) - 1,
                    min(int(ss.e_step), len(df) - 1), key="e_slider"),
        len(df) - 1)
    ss.e_speed = c[3].selectbox("h / tick", [4, 24, 48],
                                index=[4, 24, 48].index(ss.e_speed)
                                if ss.e_speed in [4, 24, 48] else 1,
                                key="e_speed_sel")

    _live(df, alerts, metrics, mode_override, extra)

    st.subheader("Does the stated 90% hold as conditions drift?")
    st.caption(
        "Rolling 30-day empirical coverage over the full test period. The "
        "un-calibrated model band and the static CQR band drift with the "
        "weather regime; online adaptive conformal keeps the stated level."
    )
    st.plotly_chart(_rollcov_fig(df_full), use_container_width=True)
    C.footer()
