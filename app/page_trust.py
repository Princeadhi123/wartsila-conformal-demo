"""Page 4 — Trust & validation: evidence tables and calibration charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import common as C
from uaagent.alerts import interval_stats_at_level

_BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
             font=dict(color="#E8EEF4"), margin=dict(l=10, r=10, t=30, b=10))


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return f"{x * 100:.1f}%" if x >= 0.005 else f"{x * 100:.2f}%"


def _f4(x) -> str:
    return "–" if x is None else f"{x:.4f}"


def _mw(x) -> str:
    return "–" if x is None else f"{x:,.0f}"


def _num(x, spec: str = ".0f") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "–"
    return format(x, spec)


def _marine_tables(mm: dict) -> None:
    st.subheader("Marine — held-out test split")
    rows = []
    for t in ("kMc", "kMt"):
        s = mm["targets"][t]["test"]
        rows.append({
            "target": t,
            "RMSE": _f4(s["rmse"]),
            "cov. split": _pct(s["coverage_split"]),
            "cov. adaptive": _pct(s["coverage_adaptive"]),
            "width split": _f4(s["mean_width_split"]),
            "width adaptive": _f4(s["mean_width_adaptive"]),
        })
    st.dataframe(pd.DataFrame(rows).set_index("target"),
                 width="stretch")

    st.markdown("**Alert quality — conventional vs uncertainty-aware**")
    rows = []
    for t in ("kMc", "kMt"):
        s = mm["targets"][t]["test"]
        rows.append({
            "target": t,
            "FA conv.": _pct(s["naive_false_alarm_rate"]),
            "FA UA": _pct(s["ua_false_alarm_rate"]),
            "prec. conv.": _pct(s["naive_precision"]),
            "prec. UA": _pct(s["ua_precision"]),
            "det. conv.": _pct(s["naive_detection_rate"]),
            "det. UA conf.": _pct(s["ua_confirmed_rate_when_degraded"]),
            "det. UA w+c": _pct(s["ua_watch_or_confirmed_when_degraded"]),
            "watch | healthy": _pct(s["ua_watch_rate_when_healthy"]),
        })
    st.dataframe(pd.DataFrame(rows).set_index("target"),
                 width="stretch")

    st.markdown("**Voyage replay — alert episodes**")
    rows = []
    for t in ("kMc", "kMt"):
        ep = mm["targets"][t]["voyage"]["episodes"]
        for label, key in (("conventional", "naive"),
                           ("UA watch", "ua_watch"),
                           ("UA confirmed", "ua_confirmed")):
            e = ep[key]
            rows.append({
                "target": t, "feed": label,
                "raises": e["n_raises"],
                "raised while still above threshold": e["n_false_raises"],
                "first true raise (h)": _num(e["first_true_raise"]),
                "detection delay (h)": _num(e["detection_delay_steps"]),
                "hours in false alarm": e["steps_in_false_alarm"],
            })
    st.dataframe(pd.DataFrame(rows).set_index(["target", "feed"]),
                 width="stretch")
    st.caption(
        "For the watch tier, raises while still above threshold are the "
        "intended early warning — they ask for evidence, not action."
    )
    st.info(
        "Trade-off, stated plainly: on held-out data the conventional alarm "
        "is wrong about 1 time in 20 when the plant is healthy; a confirmed "
        "uncertainty-aware alert almost never is. The price is that "
        "confirmed alerts come later and cover fewer degraded points — the "
        "watch tier keeps the early warning and tells the operator to "
        "gather evidence rather than act."
    )


def _marine_charts(mm: dict) -> None:
    st.subheader("Conditional coverage by ship speed")
    c1, c2 = st.columns(2)
    for t, col in zip(("kMc", "kMt"), (c1, c2)):
        bs = pd.DataFrame(mm["targets"][t]["test"]["by_speed"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bs["v"], y=bs["coverage_split"],
                             name="constant-width split",
                             marker_color=C.COLORS["muted"]))
        fig.add_trace(go.Bar(x=bs["v"], y=bs["coverage_adaptive"],
                             name="adaptive",
                             marker_color=C.COLORS["band_line"]))
        fig.add_hline(y=0.9, line_dash="dash", line_color=C.COLORS["accent"])
        fig.update_layout(title=f"{t} — coverage by speed", barmode="group",
                          height=280, yaxis_range=[0.5, 1.02], **_BASE)
        col.plotly_chart(fig, width="stretch")
        wfig = go.Figure()
        wfig.add_trace(go.Scatter(x=bs["v"], y=bs["width_adaptive"],
                                  mode="lines+markers",
                                  line=dict(color=C.COLORS["band_line"]),
                                  name="adaptive width"))
        wfig.add_trace(go.Scatter(x=bs["v"], y=bs["width_split"],
                                  mode="lines+markers",
                                  line=dict(color=C.COLORS["muted"],
                                            dash="dash"),
                                  name="split width"))
        wfig.update_layout(title=f"{t} — mean interval width by speed",
                           height=220, **_BASE)
        col.plotly_chart(wfig, width="stretch")
    st.caption(
        "Constant-width intervals under-cover at 3 kn, exactly where "
        "estimation is hardest; the adaptive band widens there and keeps "
        "the promised level."
    )


def _calibration_charts(mm: dict, em: dict) -> None:
    st.subheader("Calibration — nominal vs empirical")
    c1, c2 = st.columns(2)
    square = dict(scaleanchor="x", scaleratio=1, range=[0.5, 1.0])
    fig = go.Figure()
    for t, color in (("kMc", C.COLORS["band_line"]),
                     ("kMt", C.COLORS["watch"])):
        cc = pd.DataFrame(mm["targets"][t]["test"]["calibration_curve"])
        fig.add_trace(go.Scatter(x=cc["nominal"], y=cc["adaptive"],
                                 mode="lines+markers",
                                 line=dict(color=color),
                                 name=f"{t} adaptive"))
        fig.add_trace(go.Scatter(x=cc["nominal"], y=cc["split"],
                                 mode="lines+markers",
                                 line=dict(color=color, dash="dot"),
                                 name=f"{t} split"))
    fig.add_trace(go.Scatter(x=[0.5, 1], y=[0.5, 1], mode="lines",
                           line=dict(color="#E8EEF4", dash="dash"),
                           name="perfect"))
    fig.update_layout(title="Marine (held-out test)", height=380,
                      xaxis=dict(title="nominal", range=[0.5, 1.0]),
                      yaxis=dict(title="empirical", **square), **_BASE)
    c1.plotly_chart(fig, width="stretch")

    cc = pd.DataFrame(em["calibration_curve"])
    fig2 = go.Figure()
    for key in [k for k in cc.columns if k != "nominal"]:
        fig2.add_trace(go.Scatter(x=cc["nominal"], y=cc[key],
                                  mode="lines+markers",
                                  line=dict(color=C.COLORS["band_line"]),
                                  name=key))
    fig2.add_trace(go.Scatter(x=[0.5, 1], y=[0.5, 1], mode="lines",
                            line=dict(color="#E8EEF4", dash="dash"),
                            name="perfect"))
    fig2.update_layout(title="Energy (static CQR)", height=380,
                       xaxis=dict(title="nominal", range=[0.5, 1.0]),
                       yaxis=dict(title="empirical", **square), **_BASE)
    c2.plotly_chart(fig2, width="stretch")


def _interactive(mm: dict) -> None:
    st.subheader("Try it — pick the confidence level")
    st.caption(
        "Intervals recomputed live from the calibration scores and the "
        "held-out test predictions; nothing is precomputed for this widget."
    )
    level = st.slider("Nominal coverage", 50, 99, 90, 1) / 100.0
    scores = C.load_cal_scores()
    test = C.load_marine_test()
    cols = st.columns(len(mm["targets"]))
    for col, t in zip(cols, mm["targets"]):
        s = interval_stats_at_level(
            test, t, scores[f"{t}_norm"], mm["thresholds"][t], level)
        col.markdown(f"**{t}** · nominal {level:.0%}")
        col.metric("empirical coverage", _pct(s["coverage"]))
        col.metric("mean width", _f4(s["mean_width"]))
        col.metric("false-alarm rate", _pct(s["false_alarm_rate"]),
                   help="share of healthy test rows flagged confirmed")
        col.metric("confirmed detection", _pct(s["confirmed_detection"]),
                   help="share of degraded test rows flagged confirmed")


def _energy_section(em: dict) -> None:
    st.subheader("Energy — held-out test period")
    cov = em["coverage"]
    wid = em["mean_width_mw"]
    wm = em["worst_month_coverage"]
    rows = [
        ("un-calibrated quantile band", _pct(cov["raw_quantile_regression"]),
         _mw(wid["raw_quantile_regression"]), _pct(wm["coverage_raw"])),
        ("constant-width split", _pct(cov["split_constant_width"]),
         _mw(wid["split_constant_width"]), "–"),
        ("static CQR", _pct(cov["cqr_static"]), _mw(wid["cqr_static"]),
         _pct(wm["coverage_static"])),
        ("adaptive conformal (ACI)", _pct(cov["cqr_aci"]),
         _mw(wid["cqr_aci"]), _pct(wm["coverage_aci"])),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["method", "empirical coverage",
                                    "mean width (MW)",
                                    "worst-month coverage"])
        .set_index("method"),
        width="stretch")
    acc = em["accuracy"]
    st.markdown(
        f"Forecast accuracy: model MAE **{acc['mae_model']:,.0f} MW**, RMSE "
        f"**{acc['rmse_model']:,.0f} MW** vs persistence MAE "
        f"**{acc['mae_persistence']:,.0f} MW** — the model is much better "
        f"than persistence, and only the calibrated intervals make that "
        f"skill usable for reserve sizing."
    )

    bm = pd.DataFrame(em["by_month"])
    fig = go.Figure()
    for col_, name, color in (
        ("coverage_raw", "un-calibrated", C.COLORS["confirmed"]),
        ("coverage_static", "static CQR", C.COLORS["watch"]),
        ("coverage_aci", "adaptive (ACI)", C.COLORS["healthy"]),
    ):
        fig.add_trace(go.Scatter(x=bm["month"], y=bm[col_], mode="lines",
                               name=name, line=dict(color=color, width=1.6)))
    fig.add_hline(y=0.9, line_dash="dash", line_color="#E8EEF4")
    fig.update_layout(title="Monthly empirical coverage", height=300,
                      yaxis_range=[0.5, 1.0], **_BASE)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Is the uncertainty signal informative?**")
    rows = []
    for flag, label in (("risk_watch", "reserve risk (watch)"),
                        ("risk_confirmed", "reserve risk (confirmed)"),
                        ("uncertainty_high", "high uncertainty")):
        a = em["alerts"][flag]
        rows.append({
            "flag": label,
            "share of hours": _pct(a["share_of_hours"]),
            "MAE flagged (MW)": _mw(a["mean_abs_error_mw_when_flagged"]),
            "MAE not flagged (MW)": _mw(a["mean_abs_error_mw_when_not_flagged"]),
        })
    st.dataframe(pd.DataFrame(rows).set_index("flag"),
                 width="stretch")
    st.caption(
        "Flagged hours carry materially larger forecast error — the "
        "uncertainty signal identifies the difficult hours before the "
        "error happens."
    )

    st.markdown("**Reserve policies, exactly as measured**")
    r = em["reserve"]
    rows = []
    for label, key in (("fixed rule", "fixed_rule"),
                       ("conformal dynamic", "conformal_dynamic"),
                       ("fixed rule, matched shortfall",
                        "fixed_rule_matched_to_dynamic_shortfall")):
        d = r[key]
        rows.append({
            "policy": label,
            "mean reserve (MW)": _mw(d["mean_reserve_mw"]),
            "shortfall freq.": _pct(d["shortfall_frequency"]),
            "mean shortfall (MW)": _mw(d["mean_shortfall_mw_when_short"]),
            "p99 shortfall (MW)": _mw(d["p99_shortfall_mw"]),
            "max shortfall (MW)": _mw(d["max_shortfall_mw"]),
        })
    st.dataframe(pd.DataFrame(rows).set_index("policy"),
                 width="stretch")
    st.info(
        "Honest reading: on this public dataset the dynamic rule keeps the "
        "stated risk level and cuts the 99th-percentile shortfall, but "
        "holds more reserve on average than a fixed rule tuned in "
        "hindsight. Without weather forecasts or plant-level signals the "
        "intervals are calibrated but not yet sharp — sharper inputs are "
        "exactly what a pilot with operational data would add."
    )


def show() -> None:
    st.title("Trust & validation")
    st.caption(
        "Every number on this page is measured on held-out data and read "
        "from the pipeline artifacts at runtime."
    )
    mm = C.load_marine_metrics()
    em = C.load_energy_metrics()
    _marine_tables(mm)
    _marine_charts(mm)
    _calibration_charts(mm, em)
    _interactive(mm)
    st.divider()
    _energy_section(em)
    C.footer()
