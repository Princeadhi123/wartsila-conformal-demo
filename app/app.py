"""Uncertainty-Aware Decision Agent — demo app (streamlit run app/app.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(
    page_title="Uncertainty-Aware Decision Agent — demo",
    layout="wide",
)

import page_energy      # noqa: E402
import page_marine      # noqa: E402
import page_overview    # noqa: E402
import page_pilot       # noqa: E402
import page_trust       # noqa: E402



def overview():
    page_overview.show()


def marine_console():
    page_marine.show()


def energy_console():
    page_energy.show()


def trust_validation():
    page_trust.show()


def pilot_proposal():
    page_pilot.show()


nav = st.navigation(
    [
        st.Page(overview, title="Overview", default=True),
        st.Page(marine_console, title="Marine console"),
        st.Page(energy_console, title="Energy console"),
        st.Page(trust_validation, title="Trust & validation"),
        st.Page(pilot_proposal, title="Pilot proposal"),
    ]
)
nav.run()
