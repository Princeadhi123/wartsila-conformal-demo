"""Page 5 — Pilot proposal."""
from __future__ import annotations

import streamlit as st

import common as C

_PHASES = [
    ("Phase 1 — Ideation & data mapping",
     "Pick one subsystem where alert fatigue hurts — e.g. engine telemetry "
     "or hybrid grid balancing. Map data availability, label a small set of "
     "historical events, and agree the decision points the agent should "
     "support."),
    ("Phase 2 — Prototyping on CSC/LUMI",
     "Multi-model ensembles with conformal calibration trained on the "
     "chosen subsystem on academic HPC, so no production environment is "
     "touched. Offline evaluation against the held-out history."),
    ("Phase 3 — Validation in a simulated operator environment",
     "Replay realistic operations with operators in the loop. Measure "
     "decision speed and accuracy with and without calibrated uncertainty "
     "and grounded advisories."),
]

_COMPARISON = [
    ("Prediction data",
     "Public naval-propulsion simulator (UCI CBM dataset)",
     "Historical operational data under NDA"),
    ("Energy data",
     "Open Power System Data, Finland — public residual load",
     "Real balancing / telemetry feeds"),
    ("Knowledge base",
     "Synthetic documents from general engineering practice",
     "Real manuals, procedures and maintenance logs"),
    ("Advisory generator",
     "Deterministic extractive composer (+ optional LLM endpoint)",
     "Evaluated LLM ensemble with the same grounding contract"),
    ("Evaluation",
     "Replay voyage / replay grid — no operator",
     "Shadow-mode evaluation with operators"),
]

_NEEDS = [
    "One historical dataset for the chosen subsystem (telemetry, alarms or "
    "dispatch logs).",
    "A domain contact who can sanity-check findings — a few hours a month.",
    "An NDA — the university has a standard template ready.",
]

_QUESTIONS = [
    "Which subsystem hurts most from alert fatigue today?",
    "What alert-precision level would operators actually trust?",
    "Where do remote-operations engineers lose the most time right now?",
]


def show() -> None:
    st.title("Pilot proposal")
    st.markdown(
        "**Uncertainty-Aware AI Decision Agents for Real-Time Marine and "
        "Energy Operations** — from public-data prototype to a scoped pilot."
    )

    st.subheader("Three phases")
    cols = st.columns(3)
    for col, (title, body) in zip(cols, _PHASES):
        col.markdown(
            f'<div style="background:{C.COLORS["band"]};border:1px solid '
            f'{C.COLORS["band_line"]}44;border-radius:8px;padding:14px 16px;'
            f'min-height:170px"><b>{title}</b><br><br>{body}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("3-year timeline")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**Year 1**\n\nData mapping and solution architecture.")
    t2.markdown("**Year 2**\n\nPrototyping on CSC/LUMI with multi-model "
                "ensembles.")
    t3.markdown("**Year 3**\n\nValidation in a simulated operator "
                "environment and tech transfer.")

    st.subheader("In this demo vs in a pilot")
    rows = "".join(
        f"<tr><td style='padding:6px 10px'><b>{a}</b></td>"
        f"<td style='padding:6px 10px'>{b}</td>"
        f"<td style='padding:6px 10px'>{c}</td></tr>"
        for a, b, c in _COMPARISON
    )
    st.markdown(
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th style='text-align:left;padding:6px 10px'>Aspect</th>"
        "<th style='text-align:left;padding:6px 10px'>In this demo</th>"
        "<th style='text-align:left;padding:6px 10px'>"
        "In a pilot with the partner</th></tr>"
        + rows + "</table>",
        unsafe_allow_html=True,
    )

    st.subheader("What we would need")
    for n in _NEEDS:
        st.markdown(f"- {n}")

    st.subheader("Open questions for discussion")
    for q in _QUESTIONS:
        st.markdown(f"- {q}")

    st.divider()
    st.markdown(
        "**Prince Das Adhikary** — Doctoral Researcher, Turku Research "
        "Institute for Learning Analytics, University of Turku"
    )
    C.footer()
