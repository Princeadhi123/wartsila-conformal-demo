"""Page 1 — Overview: pitch, KPI cards, how it works, architecture."""
from __future__ import annotations

import streamlit as st

import common as C


def _kpi(col, title, value, sub="", fs="1.25rem"):
    col.markdown(
        f'<div style="background:{C.COLORS["band"]};border:1px solid '
        f'{C.COLORS["band_line"]}44;border-radius:8px;padding:14px 16px;'
        f'height:150px">'
        f'<div style="color:{C.COLORS["muted"]};font-size:0.8rem">{title}</div>'
        f'<div style="font-size:{fs};font-weight:700;color:#E8EEF4;'
        f'white-space:nowrap">{value}</div>'
        f'<div style="color:{C.COLORS["muted"]};font-size:0.8rem">{sub}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def show() -> None:
    st.title("Uncertainty-Aware Decision Agent — demo")
    st.markdown(
        "**An AI operator assistant that knows — and tells you — how sure "
        "it is.**"
    )

    mm = C.load_marine_metrics()
    em = C.load_energy_metrics()
    kb = C.load_kb()
    t = mm["targets"]["kMc"]["test"]
    naive_fa = t["naive_false_alarm_rate"]
    ua_fa = t["ua_false_alarm_rate"]
    ratio = naive_fa / ua_fa if ua_fa else float("inf")
    cov_ad = t["coverage_adaptive"]
    raw_qr = em["coverage"]["raw_quantile_regression"]
    aci = em["coverage"]["cqr_aci"]

    c1, c2, c3, c4 = st.columns(4)
    _kpi(
        c1, "False alarms, held-out marine test",
        f"{naive_fa:.1%} → {ua_fa:.2%}",
        f"conventional → uncertainty-aware, ~{ratio:.0f}x fewer",
    )
    _kpi(
        c2, "Marine 90% interval",
        f"90% promised → {cov_ad:.1%} delivered",
        "adaptive conformal on unseen operating points",
        fs="1.05rem",
    )
    _kpi(
        c3, "Energy 90% band honesty",
        f"{raw_qr:.0%} → {aci:.1%}",
        "un-calibrated AI band vs adaptive conformal",
    )
    _kpi(
        c4, "Grounded knowledge base",
        f"{kb.n_docs} docs · {kb.n_chunks} chunks",
        "every recommended action cites its source document",
    )

    st.subheader("How it works")
    h1, h2, h3 = st.columns(3)
    h1.markdown(
        "**1 · Calibrated prediction**\n\n"
        "Model ensembles produce point estimates; split and adaptive "
        "conformal calibration turn them into intervals with a stated "
        "coverage — the stated coverage is verified on held-out data."
    )
    h2.markdown(
        "**2 · Explain**\n\n"
        "Alerts are raised on interval evidence (confirmed / watch), then "
        "a retrieval agent explains each alert with troubleshooting steps "
        "cited to knowledge-base documents."
    )
    h3.markdown(
        "**3 · Adapt**\n\n"
        "An operator-load router watches alert pressure and shortens or "
        "expands advisories: terse under high load, full reasoning — "
        "including similar past cases — when the operator has time."
    )

    st.subheader("Architecture")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR
            bgcolor="transparent"
            node [shape=box, style="rounded,filled", fillcolor="#12283D",
                  fontcolor="#E8EEF4", color="#4DA3FF", fontname="sans-serif"]
            edge [color="#8FA6BC", fontcolor="#8FA6BC", fontname="sans-serif"]
            telemetry [label="Telemetry"]
            ensemble [label="Model ensemble"]
            conformal [label="Conformal calibration"]
            alertlogic [label="Alert logic\nconfirmed / watch / normal"]
            rag [label="RAG agent"]
            kb [label="Knowledge base"]
            console [label="Operator console"]
            router [label="Operator-load router"]
            telemetry -> ensemble -> conformal -> alertlogic -> rag -> console
            kb -> rag
            router -> rag
            alertlogic -> router
        }
        """,
        width="stretch",
    )

    with st.expander("What is real in this demo"):
        n = mm["n"]
        st.markdown(
            f"- **Marine data:** public UCI naval propulsion simulator dataset; "
            f"{n['train']:,} training rows, {n['test']:,} held-out test rows, "
            f"plus a {n['voyage']}-hour replay voyage. Sensor noise is injected "
            f"(documented assumption): "
            + ", ".join(
                f"{k} {v:.1%}" for k, v in list(mm["noise_rel"].items())[:5]
            )
            + ", …"
        )
        st.markdown(
            f"- **Energy data:** Open Power System Data, Finland — "
            f"{em['n']['test']:,} hourly test rows, "
            f"{em['horizon_h']:g} h-ahead residual-load forecasting."
        )
        st.markdown(
            "- **Knowledge base:** synthetic markdown documents written from "
            "general engineering practice — not real manufacturer "
            "documentation."
        )
        st.markdown(
            "- **Advisories:** deterministic extractive generator by default; "
            "an optional LLM path (OpenAI-compatible endpoint) is used only "
            "when configured, and always falls back offline."
        )
        st.markdown(
            "- **Numbers on every page are read from the pipeline artifacts "
            "at runtime** — nothing is hardcoded."
        )
    C.footer()
