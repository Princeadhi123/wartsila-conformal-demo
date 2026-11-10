"""Operator-load estimation and advisory-verbosity routing."""
from __future__ import annotations

MODE_DESCRIPTIONS: dict[str, str] = {
    "focus": (
        "Operator under high load: a terse advisory with only the top actions. "
        "Details deferred until the load drops."
    ),
    "standard": (
        "Normal workload: a balanced advisory with the key evidence and the "
        "main recommended actions."
    ),
    "deep_dive": (
        "Low workload: a full advisory with all evidence, extended actions and "
        "similar past cases from the operations logbooks."
    ),
}


def operator_load(
    active_alerts: int,
    new_alerts_last_window: int,
    has_critical: bool,
) -> tuple[float, str]:
    """Return (load score in [0, 1], verbosity mode).

    Modes: ``focus`` for high load (short advisories), ``standard`` for normal
    load, ``deep_dive`` when the operator has spare capacity.
    """
    score = 0.12 * active_alerts + 0.15 * new_alerts_last_window + (0.3 if has_critical else 0.0)
    score = min(1.0, score)
    if score >= 0.6:
        mode = "focus"
    elif score >= 0.25:
        mode = "standard"
    else:
        mode = "deep_dive"
    return score, mode
