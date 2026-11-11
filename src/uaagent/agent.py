"""Alert explanation layer: grounded RAG composer with an optional LLM path.

The alert contract (produced by the data/ML core) is a plain dict::

    {"id": str, "domain": "marine"|"energy",
     "kind": "compressor_decay"|"turbine_decay"|"reserve_shortfall_risk"
             |"forecast_uncertainty_high",
     "state": "confirmed"|"watch", "t": str,
     "estimate": float, "lo": float, "hi": float, "threshold": float,
     "coverage": float, "unit": str,
     "context": dict[str, float|str]}
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .rag import Hit, KnowledgeBase
from .routing import MODE_DESCRIPTIONS

# ---------------------------------------------------------------------------
# Static vocabulary
# ---------------------------------------------------------------------------

_QUANTITY = {
    "compressor_decay": "compressor decay coefficient kMc",
    "turbine_decay": "turbine decay coefficient kMt",
    "reserve_shortfall_risk": "residual-load upper forecast",
    "forecast_uncertainty_high": "forecast interval width",
}

_HEADLINE = {
    ("compressor_decay", "confirmed"): "Compressor decay confirmed - schedule a water wash",
    ("compressor_decay", "watch"): "Possible compressor fouling - watch and gather evidence",
    ("turbine_decay", "confirmed"): "HP turbine decay confirmed - inspect and consider de-rating",
    ("turbine_decay", "watch"): "Possible HP turbine degradation - monitor and validate sensors",
    ("reserve_shortfall_risk", "confirmed"): "Reserve shortfall risk - secure additional fast capacity",
    ("reserve_shortfall_risk", "watch"): "Reserve margin may go negative - review commitments",
    ("forecast_uncertainty_high", "confirmed"): "Forecast uncertainty unusually high - plan conservatively",
    ("forecast_uncertainty_high", "watch"): "Forecast uncertainty elevated - monitor interval width",
}

KIND_LABEL = {
    "compressor_decay": "Compressor decay",
    "turbine_decay": "HP turbine decay",
    "reserve_shortfall_risk": "Reserve shortfall risk",
    "forecast_uncertainty_high": "Forecast uncertainty high",
}
MODE_LABEL = {"focus": "Focus", "standard": "Standard", "deep_dive": "Deep dive"}

# context key -> (label, unit, format spec or "mw" for thousands-separated)
_CTX_LABELS = {
    "margin_mw": ("Reserve margin", "MW", "mw"),
    "upper_mw": ("Forecast upper bound", "MW", "mw"),
    "committed_mw": ("Committed reserve", "MW", "mw"),
    "forecast_mw": ("Residual-load forecast", "MW", "mw"),
    "fuel_flow_kg_s": ("Fuel flow", "kg/s", ".2f"),
    "T2_C": ("Compressor outlet temperature T2", "°C", ".0f"),
    "P2_bar": ("Compressor outlet pressure P2", "bar", ".1f"),
    "T48_C": ("HP turbine exit temperature T48", "°C", ".0f"),
    "speed_knots": ("Ship speed", "kn", ".0f"),
    "lever_position": ("Lever position", "", ".1f"),
    "wind_mw": ("Wind generation", "MW", "mw"),
    "horizon_h": ("Forecast horizon", "h", ".0f"),
    "interval_width": ("Prediction interval width", "", ".3f"),
    "typical_width": ("Typical interval width", "", ".3f"),
}

# keywords that make an extracted sentence a better action for this alert.
# "hi" words weigh 3, "lo" words 1, "neg" words -4 so that watch states lead
# with evidence-gathering and confirmed states lead with the decision.
_ACTION_HINTS = {
    ("compressor_decay", "confirmed"): {
        "hi": ["wash", "schedule", "offline", "fuel", "penalty", "filter",
               "soak", "rinse", "inject"],
        "lo": ["confirm", "monitor", "log", "verify", "check"],
        "neg": []},
    ("compressor_decay", "watch"): {
        "hi": ["validate", "sensor", "confirmation", "monitor", "verify",
               "check", "load point", "spread", "trend", "compare",
               "tighten", "re-estimate", "filter"],
        "lo": ["schedule", "inspect"],
        "neg": ["wash", "de-rate", "inject", "offline", "repair", "dispatch"]},
    ("turbine_decay", "confirmed"): {
        "hi": ["borescope", "inspect", "de-rate", "t48", "rating", "repair",
               "classify", "photograph", "tip"],
        "lo": ["monitor", "verify", "check", "record", "document"],
        "neg": []},
    ("turbine_decay", "watch"): {
        "hi": ["validate", "sensor", "monitor", "spread", "trend", "compare",
               "check", "verify", "exclude", "re-estimate", "confirmation"],
        "lo": ["inspect", "schedule"],
        "neg": ["de-rate", "repair", "wash"]},
    ("reserve_shortfall_risk", "confirmed"): {
        "hi": ["upper bound", "fewest", "size", "margin", "reserve", "commit"],
        "lo": ["start", "issue", "synchronise", "load", "unload", "dispatch",
               "standby", "availability", "readiness", "pre-position"],
        "neg": []},
    ("reserve_shortfall_risk", "watch"): {
        "hi": ["monitor", "margin", "review", "verify", "check", "track",
               "record", "confirm", "standby", "availability", "readiness",
               "pre-position", "preserve"],
        "lo": ["schedule", "prefer"],
        "neg": ["issue", "dispatch", "load the unit", "start command"]},
    ("forecast_uncertainty_high", "confirmed"): {
        "hi": ["size", "upper bound", "flexible", "verify", "re-run",
               "monitor", "feeds", "width"],
        "lo": ["record", "increase", "prefer", "check"],
        "neg": []},
    ("forecast_uncertainty_high", "watch"): {
        "hi": ["monitor", "verify", "width", "review", "check", "record",
               "feeds", "re-run"],
        "lo": ["prefer", "size", "increase"],
        "neg": ["long-start", "load the unit", "dispatch"]},
}

# Imperative starters accepted when extracting action sentences from chunks.
_VERBS = {
    "act", "allow", "apply", "avoid", "bring", "check", "classify", "clean",
    "commit", "compare", "confirm", "contact", "de-rate", "derate", "dispatch",
    "do", "document", "drain", "engage", "ensure", "escalate", "estimate",
    "exclude", "follow", "give", "group", "hold", "increase", "inject",
    "inspect", "isolate", "issue", "keep", "log", "maintain", "measure",
    "monitor", "motor", "never", "open", "perform", "photograph",
    "pre-position", "pre-start", "prefer", "prepare", "preserve",
    "re-baseline", "re-check", "re-estimate", "re-run", "record", "reduce",
    "release", "replace", "restore", "review", "rinse", "run", "schedule",
    "secure", "shift", "shorten", "shut", "size", "split", "start", "stop",
    "tighten", "track", "unload", "use", "validate", "verify", "wait", "watch",
}

_MODE_CAPS = {  # mode -> (max why bullets, max actions, max sources)
    "focus": (1, 2, 2),
    "standard": (3, 4, 3),
    "deep_dive": (None, 6, 5),
}

_ACTIONABLE_TYPES = ("procedure", "ops_guideline", "service_bulletin")
_LOG_TYPE = "maintenance_log"


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    headline: str
    confidence_statement: str
    why: list[str]
    actions: list[dict]  # {"text": str, "source_ids": list[str]}
    sources: list[Hit]
    mode: str
    generator: str
    llm_latency_seconds: float | None = None
    fallback_reason: str | None = None


def format_value(v, domain: str = "marine") -> str:
    """Fixed-decimal number formatting used in confidence statements.

    Marine coefficients get 3 decimals; energy megawatts get integer
    thousands separators.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{x:,.0f}" if domain == "energy" else f"{x:.3f}"


def _fmt_ctx(v, spec: str = "g") -> str:
    """Compact formatting for context/why values."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if spec == "mw":
        return f"{x:,.0f}"
    return format(x, spec)


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

_QUERY_TERMS = {
    # kind -> (common, confirmed-state, watch-state) query fragments
    "compressor_decay": (
        "compressor fouling decay kMc T2 P2 inlet filter symptoms",
        "water wash decision criteria fuel penalty online offline wash procedure",
        "early fouling signs monitoring watch handling",
    ),
    "turbine_decay": (
        "HP turbine degradation kMt T48 erosion hot corrosion tip clearance",
        "borescope inspection de-rate continued operation",
        "T48 trend monitoring watch handling",
    ),
    "reserve_shortfall_risk": (
        "reserve sizing upper bound forecast interval spinning fast-start reserve",
        "fast-start gas engine genset dispatch BESS hybrid ramp",
        "monitor reserve margin watch handling",
    ),
    "forecast_uncertainty_high": (
        "forecast uncertainty wide interval conservative dispatch upper bound",
        "unusually wide interval response",
        "monitor interval width watch handling",
    ),
}


def _is_low_load(ctx: dict) -> bool:
    try:
        return float(ctx.get("speed_knots", 99.0)) < 12.0
    except (TypeError, ValueError):
        return False


def build_query(alert: dict) -> str:
    """Keyword query tuned per alert kind/state and context."""
    kind = alert.get("kind", "")
    state = alert.get("state", "watch")
    ctx = alert.get("context") or {}
    parts: list[str] = []
    terms = _QUERY_TERMS.get(kind)
    if terms:
        base, confirmed, watch = terms
        parts.append(base)
        parts.append(confirmed if state == "confirmed" else watch)
    if kind in ("compressor_decay", "turbine_decay"):
        parts.append("sensor validation check before acting")
        if _is_low_load(ctx):
            parts.append("low load observability confirmation run")
    if state == "confirmed":
        parts.append("alert triage escalation")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Deterministic composer helpers
# ---------------------------------------------------------------------------

def _confidence_statement(alert: dict) -> str:
    if alert.get("domain") == "energy":
        return _energy_confidence(alert)
    lo, hi, thr = alert["lo"], alert["hi"], alert["threshold"]
    qty = _QUANTITY.get(alert.get("kind", ""), "estimate")
    cov = alert.get("coverage")
    cov_txt = f"{cov:.0%}" if isinstance(cov, (int, float)) else "the stated"
    unit = alert.get("unit") or ""
    u = f" {unit}" if unit else ""
    if alert.get("state") == "confirmed":
        side = "below" if float(hi) < float(thr) else "above"
        return (
            f"With {cov_txt} coverage the {qty} lies in "
            f"[{format_value(lo)}, {format_value(hi)}]{u}; the entire interval "
            f"is {side} the {format_value(thr)}{u} action threshold."
        )
    return (
        f"With {cov_txt} coverage the {qty} lies in "
        f"[{format_value(lo)}, {format_value(hi)}]{u}; the interval straddles "
        f"the {format_value(thr)}{u} threshold - not enough evidence to act yet."
    )


def _energy_confidence(alert: dict) -> str:
    """Energy-domain statements keyed on the alert's settled context keys."""
    ctx = alert.get("context") or {}
    kind = alert.get("kind", "")
    cov = alert.get("coverage")
    cov_txt = f"{cov:.0%}" if isinstance(cov, (int, float)) else "90%"
    h = ctx.get("horizon_h")
    h_txt = f"{float(h):g} h" if h is not None else "the target horizon"
    fmt = lambda v: format_value(v, "energy")

    if kind == "reserve_shortfall_risk":
        return _reserve_confidence(alert, ctx, h_txt)
    if kind == "forecast_uncertainty_high":
        return _uncertainty_confidence(alert, ctx, cov_txt, h_txt)
    return (
        f"With {cov_txt} coverage the estimate lies in "
        f"[{fmt(alert['lo'])}, {fmt(alert['hi'])}] MW against the "
        f"{fmt(alert['threshold'])} MW threshold."
    )


def _reserve_confidence(alert: dict, ctx: dict, h_txt: str) -> str:
    fmt = lambda v: format_value(v, "energy")
    upper = ctx.get("upper_mw")
    committed = ctx.get("committed_mw")
    short = ctx.get("margin_mw")
    short = (-float(short)) if short is not None else float(upper) - float(committed)
    units = ctx.get("extra_units")
    unit_mw = ctx.get("unit_mw")
    s = (
        f"With 95% confidence residual load in {h_txt} stays below "
        f"{fmt(upper)} MW. Scheduled generation plus standing reserve "
        f"covers {fmt(committed)} MW"
    )
    if short > 0:
        units_txt = (
            f" (≈ {int(units)} × {float(unit_mw):g} MW fast-start units)"
            if units and unit_mw else ""
        )
        s += f" — {fmt(short)} MW short of that bound{units_txt}."
    else:
        s += " — sufficient to cover that bound."
    if alert.get("state") == "confirmed" and ctx.get("upper85_mw") is not None:
        s += (
            f" Even the less cautious 85% bound ({fmt(ctx['upper85_mw'])} MW) "
            f"exceeds committed capacity."
        )
    return s


def _uncertainty_confidence(alert: dict, ctx: dict, cov_txt: str,
                            h_txt: str) -> str:
    fmt = lambda v: format_value(v, "energy")
    lo_mw = ctx.get("lo_mw", alert["lo"])
    hi_mw = ctx.get("hi_mw", alert["hi"])
    width = ctx.get("interval_width", float(hi_mw) - float(lo_mw))
    tw = ctx.get("typical_width")
    s = (
        f"The {cov_txt} interval for residual load in {h_txt} is "
        f"[{fmt(lo_mw)}, {fmt(hi_mw)}] MW — {fmt(width)} MW wide"
    )
    try:
        ratio = float(width) / float(tw) if tw else None
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = None
    if ratio:
        s += f", {ratio:.1f}× the typical width"
    s += ". Treat the point forecast with caution."
    return s


def _headline(alert: dict) -> str:
    return _HEADLINE.get(
        (alert.get("kind", ""), alert.get("state", "watch")),
        f"{KIND_LABEL.get(alert.get('kind', ''), alert.get('kind', 'Alert'))} "
        f"- {alert.get('state', 'watch')}",
    )


def _num(ctx: dict, key: str):
    try:
        v = float(ctx[key])
    except (KeyError, TypeError, ValueError):
        return None
    return None if math.isnan(v) else v  # NaN -> missing


def _why_bullets(alert: dict) -> list[str]:
    """Evidence bullets ordered by diagnostic value (most informative first)."""
    if alert.get("domain") == "energy":
        return _why_bullets_energy(alert)
    return _why_bullets_marine(alert)


def _context_bullets(ctx: dict, keys) -> list[str]:
    """Format raw context signals in the given order via _CTX_LABELS."""
    out: list[str] = []
    for key in keys:
        if key in ctx and key in _CTX_LABELS:
            label, unit, spec = _CTX_LABELS[key]
            out.append(
                f"{label}: {_fmt_ctx(ctx[key], spec)} {unit}".rstrip())
    return out


def _interval_width_ratio(ctx: dict):
    iw, tw = _num(ctx, "interval_width"), _num(ctx, "typical_width")
    return (iw / tw) if (iw is not None and tw) else None


def _why_bullets_marine(alert: dict) -> list[str]:
    ctx = alert.get("context") or {}
    bullets: list[str] = []
    speed = _num(ctx, "speed_knots")
    ratio = _interval_width_ratio(ctx)
    # (a) low-load observability first - it determines how much to trust the alert
    if speed is not None and speed < 12:
        bullets.append(
            f"The alert arose at low load ({speed:g} kn) where health "
            f"observability is reduced; a confirmation run at higher steady "
            f"load is advisable."
        )
    # (b) interval unusually wide
    if ratio is not None and ratio > 1.3:
        bullets.append(
            f"The prediction interval is {ratio:.1f}x wider than typical, so "
            f"the estimate carries less evidence than usual."
        )
    # (c) fuel penalty evidence
    fep = _num(ctx, "fuel_excess_pct")
    if fep is not None:
        direction = "above" if fep >= 0 else "below"
        bullets.append(
            f"Fuel flow is about {abs(fep):.1f}% {direction} the healthy "
            f"baseline at this speed."
        )
    # then raw signals in diagnostic order
    bullets.extend(_context_bullets(ctx, (
        "T2_C", "P2_bar", "T48_C", "fuel_flow_kg_s", "speed_knots",
        "lever_position")))
    if ratio is None or ratio <= 1.3:
        bullets.extend(_context_bullets(
            ctx, ("interval_width", "typical_width")))
    if not bullets:
        bullets.append(
            f"Estimate {format_value(alert['estimate'])} with interval "
            f"[{format_value(alert['lo'])}, {format_value(alert['hi'])}] "
            f"against the {format_value(alert['threshold'])} threshold."
        )
    return bullets


def _why_bullets_energy(alert: dict) -> list[str]:
    ctx = alert.get("context") or {}
    fmt = lambda v: _fmt_ctx(v, "mw")
    bullets: list[str] = []
    # margin / extra units first
    margin = _num(ctx, "margin_mw")
    units, unit_mw = ctx.get("extra_units"), _num(ctx, "unit_mw")
    if margin is not None and margin < 0:
        s = f"Committed capacity is {fmt(-margin)} MW short of the 95% upper bound"
        if units and unit_mw:
            s += f" — about {int(units)} × {unit_mw:g} MW fast-start units needed"
        bullets.append(s + ".")
    elif margin is not None:
        bullets.append(
            f"Committed capacity covers the 95% upper bound with "
            f"{fmt(margin)} MW to spare."
        )
    # interval width vs typical
    iw, tw = _num(ctx, "interval_width"), _num(ctx, "typical_width")
    if iw is not None and tw:
        ratio = iw / tw
        note = f"The 90% interval is {fmt(iw)} MW wide — {ratio:.1f}× the typical {fmt(tw)} MW"
        bullets.append(note + ("." if ratio <= 1.3 else " — uncertainty is elevated."))
    # wind level, forecast, horizon
    wind = _num(ctx, "wind_mw")
    if wind is not None:
        bullets.append(f"Wind generation now: {fmt(wind)} MW.")
    fc = _num(ctx, "forecast_mw")
    if fc is not None:
        bullets.append(f"Residual-load forecast: {fmt(fc)} MW.")
    h = _num(ctx, "horizon_h")
    if h is not None:
        bullets.append(f"Forecast horizon: {h:g} h.")
    if not bullets:
        bullets.append(
            f"Estimate {fmt(alert['estimate'])} MW with interval "
            f"[{fmt(alert['lo'])}, {fmt(alert['hi'])}] MW."
        )
    return bullets


_MD_STRIP = re.compile(r"[*_`#]")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+")


def _clean_action(text: str) -> str:
    t = _MD_STRIP.sub("", text)
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".!?":
        t += "."
    return t


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _candidate_sentences(line: str) -> list[str]:
    """Action candidates from one chunk line.

    Numbered/bulleted items are kept whole; prose paragraphs are split into
    sentences and only imperative-starting sentences are kept.
    """
    numbered = bool(re.match(r"^\d+[.)]\s", line))
    marked = bool(_LIST_MARKER.match(line))
    body = _LIST_MARKER.sub("", line).strip()
    if not body:
        return []
    if numbered or marked:
        return [body]
    return [
        s for s in _SENTENCE_SPLIT.split(body)
        if s.split(" ", 1)[0].strip(":,.").lower() in _VERBS
    ]


def _clean_action_candidates(hit: Hit):
    """Yield cleaned eligible (20-180 char) action candidates from a chunk."""
    for raw in hit.text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ">")):
            continue
        for cand in _candidate_sentences(line):
            cleaned = _clean_action(cand)
            if 20 <= len(cleaned) <= 180:
                yield cleaned


def _action_hint_score(key: str, hints) -> int:
    return (3 * sum(1 for w in hints.get("hi", []) if w in key)
            + sum(1 for w in hints.get("lo", []) if w in key)
            - 4 * sum(1 for w in hints.get("neg", []) if w in key))


def _extract_actions(sources: list[Hit], cap: int, alert: dict) -> list[dict]:
    """Pull imperative / numbered-step lines out of the retrieved chunks."""
    hints = _ACTION_HINTS.get((alert.get("kind", ""), alert.get("state", "")), [])
    preferred = [h for h in sources if h.doc_type in _ACTIONABLE_TYPES]
    rest = [h for h in sources if h.doc_type not in _ACTIONABLE_TYPES]
    candidates: list[tuple[int, int, dict]] = []  # (hint score, order, action)
    seen: set[str] = set()
    order = 0
    for hit in preferred + rest:
        for cleaned in _clean_action_candidates(hit):
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((_action_hint_score(key, hints), order,
                               {"text": cleaned, "source_ids": [hit.doc_id]}))
            order += 1
        if len(candidates) >= 24:
            break
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [a for _, _, a in candidates[:cap]]


def _select_sources(pool: list[Hit], cap: int, ensure_log: bool) -> list[Hit]:
    sel = pool[:cap]
    if ensure_log and not any(h.doc_type == _LOG_TYPE for h in sel):
        log = next((h for h in pool[cap:] if h.doc_type == _LOG_TYPE), None)
        if log is not None:
            sel = (sel[:-1] + [log]) if len(sel) == cap else sel + [log]
    return sel


def _retrieve_pool(alert: dict, kb: KnowledgeBase, mode: str) -> list[Hit]:
    query = build_query(alert)
    domain = alert.get("domain")
    pool = list(kb.retrieve(query, k=8, domain=domain))
    # cross-cutting (triage/escalation) docs live outside the alert domain
    pool += [
        h for h in kb.retrieve(query, k=3)
        if h.doc_id not in {x.doc_id for x in pool}
    ]
    if mode == "deep_dive":
        logs = kb.retrieve(query, k=2, domain=domain, doc_types=[_LOG_TYPE])
        pool += [h for h in logs if h.doc_id not in {x.doc_id for x in pool}]
    # dedupe by (doc_id, section) keeping the best score, then re-rank
    best: dict[tuple[str, str], Hit] = {}
    for h in pool:
        key = (h.doc_id, h.section)
        if key not in best or h.score > best[key].score:
            best[key] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def _explain_offline(alert: dict, sources: list[Hit], pool: list[Hit],
                     mode: str, conf: str) -> Explanation:
    max_why, max_actions, _ = _MODE_CAPS.get(mode, _MODE_CAPS["standard"])
    why = _why_bullets(alert)
    if max_why is not None:
        why = why[:max_why]
    actions = _extract_actions(pool, max_actions, alert)
    if mode == "deep_dive":
        log = next((h for h in sources if h.doc_type == _LOG_TYPE), None)
        if log is not None:
            entry = {
                "text": (
                    f"Review the similar past case in \"{log.title}\" before "
                    f"committing resources."
                ),
                "source_ids": [log.doc_id],
            }
            pos = min(3, len(actions))
            actions = actions[:pos] + [entry] + actions[pos:]
            actions = actions[:max_actions]
    return Explanation(
        headline=_headline(alert),
        confidence_statement=conf,
        why=why,
        actions=actions,
        sources=sources,
        mode=mode,
        generator="offline-extractive",
    )


# ---------------------------------------------------------------------------
# Optional LLM path (OpenAI-compatible /chat/completions)
# ---------------------------------------------------------------------------

# CSC Aitta research inference endpoint, per
# https://docs.lumi-supercomputer.eu/laif/inference/aitta/
CSC_BASE_URL = "https://aitta-api.csc.fi/openai/v1"
CSC_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT_SECONDS = 30.0  # connect/read timeout; no retries
_MAX_TOKENS = 2048
_AITTA_TOKEN_FILE = ".aitta_token"


def _aitta_token() -> "str | None":
    """Read the local Aitta token file if it exists and is readable."""
    try:
        token = (Path.home() / _AITTA_TOKEN_FILE).read_text(
            encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return token or None


class LLMClient:
    """Minimal OpenAI-compatible chat client; no secrets are logged."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    @classmethod
    def from_env(cls, overrides: "dict[str, str] | None" = None
                 ) -> "LLMClient | None":
        """Resolve LLM configuration without mutating os.environ.

        Per endpoint key: a non-empty environment variable, then the
        overrides mapping (e.g. Streamlit secrets read by the app). Api
        key order: env UAA_LLM_API_KEY, env AITTA_KEY, overrides
        UAA_LLM_API_KEY, overrides AITTA_KEY, then ~/.aitta_token — the
        file is only consulted when no base is configured or the base is
        the CSC endpoint, so a CSC token is never sent to another host.
        A discovered api key defaults missing endpoint settings to the
        CSC Aitta service; an explicit base+model pair also works with
        no key (local OpenAI-compatible servers). Returns None when
        neither holds — this never performs network I/O.
        """
        overrides = overrides or {}

        def _cfg(key: str) -> "str | None":
            return os.environ.get(key) or overrides.get(key) or None

        base = _cfg("UAA_LLM_BASE_URL")
        model = _cfg("UAA_LLM_MODEL")
        api_key = (
            os.environ.get("UAA_LLM_API_KEY")
            or os.environ.get("AITTA_KEY")
            or overrides.get("UAA_LLM_API_KEY")
            or overrides.get("AITTA_KEY")
            or None
        )
        if api_key is None and (
                base is None or base.rstrip("/") == CSC_BASE_URL):
            api_key = _aitta_token()
        if api_key:
            base = base or CSC_BASE_URL
            model = model or CSC_MODEL
        if not (base and model):
            return None
        return cls(base, api_key or "", model)

    def chat(self, messages: list[dict],
             timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": _MAX_TOKENS,
        }
        if self.base_url != CSC_BASE_URL:
            # Aitta's JSON-mode support is unverified and the prompt already
            # requests a JSON object, so json mode stays opt-in elsewhere.
            payload["response_format"] = {"type": "json_object"}
        if self.model == CSC_MODEL:
            # gpt-oss supports a reasoning effort knob; low keeps the
            # advisory interactive.
            payload["reasoning_effort"] = "low"
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


_LLM_MODE_INSTRUCTION = {
    "focus": "The operator is under high load: give at most 1 why bullet and 2 actions, terse wording.",
    "standard": "Give at most 3 why bullets and 4 actions, concise wording.",
    "deep_dive": "Give a full why list and up to 6 actions; include a similar past case from any log source.",
}


def _llm_messages(alert: dict, sources: list[Hit], mode: str) -> list[dict]:
    chunks = "\n\n".join(
        f"[{h.doc_id}] {h.title} :: {h.section}\n{h.text}" for h in sources
    )
    user = (
        f"ALERT (JSON):\n{json.dumps(alert, indent=2, default=str)}\n\n"
        f"MODE: {mode} - {MODE_DESCRIPTIONS.get(mode, '')} "
        f"{_LLM_MODE_INSTRUCTION.get(mode, '')}\n\n"
        f"KNOWLEDGE CHUNKS:\n{chunks}\n\n"
        "Write an operator advisory grounded ONLY in the chunks above.\n"
        "- Restate the alert's interval bounds and threshold exactly as given.\n"
        "- Every action must cite the doc ids it came from in \"source_ids\".\n"
        "- Reply with a JSON object: "
        "{\"headline\": str, \"why\": [str, ...], "
        "\"actions\": [{\"text\": str, \"source_ids\": [str, ...]}, ...]}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You write concise, factual advisories for an operations "
                "centre. Use only the supplied knowledge chunks and cite "
                "[doc_id] per action. Output valid JSON only."
            ),
        },
        {"role": "user", "content": user},
    ]


def _strip_code_fence(raw: str) -> str:
    """Remove an optional ``` / ```json wrapper using string operations."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    raw = raw[3:]
    if raw.startswith("json"):
        raw = raw[4:]
    raw = raw.lstrip()
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def _explain_llm(alert: dict, sources: list[Hit], mode: str,
                 llm: LLMClient, conf: str) -> Explanation:
    max_why, max_actions, _ = _MODE_CAPS.get(mode, _MODE_CAPS["standard"])
    raw = _strip_code_fence(llm.chat(_llm_messages(alert, sources, mode),
                                     timeout=REQUEST_TIMEOUT_SECONDS))
    data = json.loads(raw)
    known = {h.doc_id for h in sources}
    headline = str(data["headline"]).strip()
    why = [str(w).strip() for w in data["why"] if str(w).strip()]
    if max_why is not None:
        why = why[:max_why]
    actions: list[dict] = []
    for a in data.get("actions", []):
        text = str(a.get("text", "")).strip()
        ids = [str(i) for i in a.get("source_ids", [])]
        if not text or not ids:
            raise ValueError("LLM action missing text or citations")
        if any(i not in known for i in ids):
            raise ValueError(f"LLM cited unknown doc id: {ids}")
        actions.append({"text": text, "source_ids": ids})
    if not headline or not why or not actions:
        raise ValueError("LLM returned an incomplete advisory")
    actions = actions[:max_actions]
    return Explanation(
        headline=headline,
        confidence_statement=conf,
        why=why,
        actions=actions,
        sources=sources,
        mode=mode,
        generator=f"llm:{llm.model}",
    )


def _fallback_reason(exc: Exception) -> str:
    """Categorical reason for an LLM failure — never exception text, HTTP
    bodies, headers or tokens."""
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            return "authentication failed"
        if status == 429:
            return "rate limited"
        return "service error"
    if isinstance(exc, ValueError):
        # includes malformed JSON from the body and rejected citations
        return "invalid response"
    if isinstance(exc, requests.RequestException):
        return "request failed"
    return "invalid response"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def explain(
    alert: dict,
    kb: KnowledgeBase,
    mode: str = "standard",
    llm: "LLMClient | None" = None,
) -> Explanation:
    """Explain an alert with a source-grounded advisory.

    Offline (default) path is a deterministic extractive composer. If an
    LLMClient is configured (explicitly, via UAA_LLM_*/AITTA_KEY env vars
    or ~/.aitta_token) it is tried first; any failure falls back to the
    offline composer with a categorical fallback_reason. The measured
    duration of the attempted LLM call (including response validation)
    is recorded in llm_latency_seconds on both paths.
    """
    if llm is None:
        llm = LLMClient.from_env()
    _, _, max_sources = _MODE_CAPS.get(mode, _MODE_CAPS["standard"])
    pool = _retrieve_pool(alert, kb, mode)
    sources = _select_sources(pool, max_sources, ensure_log=(mode == "deep_dive"))
    conf = _confidence_statement(alert)
    if llm is not None and sources:
        started = time.perf_counter()
        try:
            ex = _explain_llm(alert, sources, mode, llm, conf)
        except Exception as exc:
            ex = _explain_offline(alert, sources, pool, mode, conf)
            ex.fallback_reason = _fallback_reason(exc)
        ex.llm_latency_seconds = time.perf_counter() - started
        return ex
    return _explain_offline(alert, sources, pool, mode, conf)
