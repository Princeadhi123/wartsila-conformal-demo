"""Shared loaders, constants and renderers for the Streamlit demo app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (SRC, ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from uaagent.agent import (                              # noqa: E402
    KIND_LABEL,
    MODE_LABEL,
    LLMClient,
    explain,
    format_value,
)
from uaagent.alerts import (                              # noqa: E402
    energy_alerts,
    marine_alerts,
    marine_naive_alerts,
)
from uaagent.rag import KnowledgeBase                     # noqa: E402
from uaagent.routing import MODE_DESCRIPTIONS, operator_load  # noqa: E402

ART = ROOT / "artifacts"

COLORS = {
    "accent": "#FF7A1A",
    "confirmed": "#E5484D",
    "watch": "#F5A524",
    "healthy": "#3DD68C",
    "band": "rgba(77, 163, 255, 0.28)",
    "band_line": "#4DA3FF",
    "muted": "#8FA6BC",
}

FOOTER = (
    "Research demo · University of Turku · public datasets (UCI naval "
    "propulsion simulator; Open Power System Data, Finland) · knowledge-base "
    "documents are synthetic · no Wärtsilä data used"
)

_STATE_COLOR = {"confirmed": COLORS["confirmed"], "watch": COLORS["watch"],
                "normal": COLORS["healthy"]}


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_marine_voyage() -> pd.DataFrame:
    return pd.read_csv(ART / "marine_voyage.csv")


@st.cache_data(show_spinner=False)
def load_marine_test() -> pd.DataFrame:
    return pd.read_csv(ART / "marine_test.csv")


@st.cache_data(show_spinner=False)
def load_marine_metrics() -> dict:
    return json.loads((ART / "marine_metrics.json").read_text())


@st.cache_data(show_spinner=False)
def load_energy_test() -> pd.DataFrame:
    return pd.read_csv(ART / "energy_test.csv", parse_dates=["decision_time", "target_time"])


@st.cache_data(show_spinner=False)
def load_energy_metrics() -> dict:
    return json.loads((ART / "energy_metrics.json").read_text())


@st.cache_data(show_spinner=False)
def load_cal_scores() -> dict:
    npz = np.load(ART / "marine_cal_scores.npz")
    return {k: npz[k] for k in npz.files}


@st.cache_resource(show_spinner=False)
def load_kb() -> KnowledgeBase:
    return KnowledgeBase(ROOT / "kb")


@st.cache_data(show_spinner=False)
def get_marine_alerts() -> list[dict]:
    return marine_alerts(load_marine_voyage(), load_marine_metrics())


@st.cache_data(show_spinner=False)
def get_naive_alerts() -> list[dict]:
    return marine_naive_alerts(load_marine_voyage(), load_marine_metrics())


def get_energy_alerts(df: pd.DataFrame) -> list[dict]:
    return energy_alerts(df, load_energy_metrics())


_LLM_SECRET_KEYS = ("UAA_LLM_BASE_URL", "UAA_LLM_MODEL",
                    "UAA_LLM_API_KEY", "AITTA_KEY")


def _llm_secret_overrides() -> dict:
    """Optional LLM config from root-level st.secrets keys.

    A missing or unreadable secrets file yields no overrides; secrets are
    never copied into os.environ (env vars still win inside from_env).
    """
    try:
        secrets = st.secrets
        return {k: str(secrets[k]) for k in _LLM_SECRET_KEYS if k in secrets}
    except Exception:
        return {}


def get_llm_client() -> "LLMClient | None":
    """The configured advisory LLM client, or None when unconfigured.

    Resolved on every rerun — cheap file reads — so token rotation or
    endpoint changes take effect without restarting the app.
    """
    return LLMClient.from_env(overrides=_llm_secret_overrides())


@st.cache_data(show_spinner=False, ttl=300)
def explain_cached(alert: dict, mode: str, llm_base_url: str = "",
                   llm_model: str = "", _llm: "LLMClient | None" = None):
    """Explain one alert; results are cached for 5 minutes.

    The endpoint and model participate in the cache key so a model change
    never replays another backend's advisory; the client itself (_llm) is
    excluded from hashing. Fallbacks are cached for the same TTL.
    """
    return explain(alert, load_kb(), mode=mode, llm=_llm)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

_STATUS_WIDTHS = {
    "m_replay_status": (1093, 1129),
    "e_replay_status": (1825, 1129),
}


def reserve_replay_status_space() -> None:
    rules = []
    for key, widths in _STATUS_WIDTHS.items():
        rules.append(f".st-key-{key}{{container-type:inline-size;}}")
        for idx, (width, lh) in enumerate(zip(widths, (1.6, 1.4)),
                                          start=1):
            for div, lines in ((1, 2), (2, 3), (3, 4), (4, 6)):
                rules.append(
                    f"@container (max-width:{round(width / div)}px)"
                    f"{{.st-key-{key} > div:nth-child({idx})"
                    f"{{min-height:{lines * lh}rem;}}}}")
    st.html("<style>" + "".join(rules) + "</style>")


def reserve_chart_space(heights: dict[str, int]) -> None:
    st.html(
        "<style>" + "".join(
            f".st-key-{key}{{min-height:{h}px;}}"
            for key, h in heights.items())
        + "</style>"
    )


def chip(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;border:1px solid {color};'
        f'color:{color};padding:2px 10px;border-radius:12px;'
        f'font-size:0.8rem;font-weight:600;white-space:nowrap">{text}</span>'
    )


def state_chip(state: str, label: str = "") -> str:
    return chip(f"{label}{state.upper()}", _STATE_COLOR.get(state, COLORS["muted"]))


def render_confidence(text: str) -> None:
    st.markdown(
        f'<div style="background:{COLORS["band"]};border-left:3px solid '
        f'{COLORS["band_line"]};padding:10px 14px;border-radius:4px;'
        f'font-size:1.0rem">{text}</div>',
        unsafe_allow_html=True,
    )


def render_advisory(alert: dict, mode: str) -> None:
    """Render a full Explanation for one alert."""
    llm = get_llm_client()
    ex = explain_cached(
        alert, mode,
        llm_base_url=llm.base_url if llm else "",
        llm_model=llm.model if llm else "",
        _llm=llm,
    )
    st.markdown(f"#### {ex.headline}")
    if ex.fallback_reason:
        waited = (f" after {ex.llm_latency_seconds:.1f} s"
                  if ex.llm_latency_seconds is not None else "")
        st.warning(
            f"Configured LLM advisory unavailable ({ex.fallback_reason})"
            f"{waited} — showing the deterministic offline advisory instead."
        )
    render_confidence(ex.confidence_statement)
    if ex.why:
        st.markdown("**Why**")
        for w in ex.why:
            st.markdown(f"- {w}")
    if ex.actions:
        st.markdown("**Recommended actions**")
        for i, a in enumerate(ex.actions, 1):
            ids = ", ".join(a["source_ids"])
            st.markdown(f"{i}. {a['text']} `[{ids}]`")
    with st.expander("Sources", expanded=False):
        for h in ex.sources:
            st.markdown(
                f"**{h.doc_id}** — {h.title} · *{h.section}* "
                f"({h.doc_type}, score {h.score:.2f})"
            )
            st.caption(h.text[:600])
    caption = (
        f"Generator: {ex.generator} · Mode: "
        f"{MODE_LABEL.get(ex.mode, ex.mode)}"
    )
    if ex.llm_latency_seconds is not None:
        caption += (
            f" · LLM request {ex.llm_latency_seconds:.1f} s "
            "(original request; cached results reuse this timing)"
        )
    st.caption(caption)


def ua_alert_label(a: dict) -> str:
    kind = KIND_LABEL.get(a["kind"], a["kind"])
    d = a["domain"]
    est = format_value(a["estimate"], d)
    lo, hi = format_value(a["lo"], d), format_value(a["hi"], d)
    unit = f" {a['unit']}" if a["unit"] else ""
    t = str(a["t"])
    t = t[0].upper() + t[1:]
    return f"{t} · {kind} · {est} [{lo}, {hi}]{unit}"


def render_ua_feed(alerts: list[dict], step: int, key: str,
                   max_items: int = 8) -> dict | None:
    """UA alert feed; returns the alert selected for the advisory."""
    shown = [a for a in alerts if a["step"] <= step][::-1][:max_items]
    if not shown:
        st.info("No uncertainty-aware alerts raised yet.")
        return None
    for a in shown:
        col = _STATE_COLOR[a["state"]]
        st.markdown(
            f"<small>{chip(a['state'].upper(), col)} {ua_alert_label(a)}</small>",
            unsafe_allow_html=True,
        )
    # Default to the latest alert; a manual pick survives until the next new
    # alert arrives, then the selection snaps back to the newest.
    latest = shown[0]["id"]
    if st.session_state.get(f"{key}_latest") != latest:
        st.session_state[f"{key}_latest"] = latest
        st.session_state[key] = 0
    sel = st.selectbox(
        "Advisory for",
        options=list(range(len(shown))),
        format_func=lambda i: ua_alert_label(shown[i]),
        key=key,
    )
    return shown[sel]


def render_naive_feed(naive: list[dict], step: int, show_truth: bool) -> None:
    shown = [a for a in naive if a["step"] <= step][::-1]
    if not shown:
        st.info("No conventional alarms raised yet.")
        return
    for a in shown[:8]:
        kind = KIND_LABEL.get(a["kind"], a["kind"])
        line = (
            f"{chip('ALARM', COLORS['muted'])} Hour {a['step']} · {kind} · "
            f"est {a['estimate']:.3f} < {a['threshold']}"
        )
        if show_truth:
            line += (" · " + chip("FALSE ALARM", COLORS["confirmed"])
                     if a["is_false"]
                     else " · " + chip("TRUE", COLORS["healthy"]))
        st.markdown(f"<small>{line}</small>", unsafe_allow_html=True)


def footer() -> None:
    st.divider()
    st.caption(FOOTER)
