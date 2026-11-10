"""Conformal prediction utilities (all methods are from the published literature).

- Split conformal prediction: Vovk et al. (2005); Lei et al. (2018)
- Locally adaptive (normalised) split conformal: Papadopoulos et al. (2008); Lei et al. (2018)
- Conformalized quantile regression (CQR): Romano, Patterson & Candes (2019)
- Adaptive conformal inference (ACI) for time series: Gibbs & Candes (2021)
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of calibration scores.

    Returns +inf when the calibration set is too small to certify the level.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = scores.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[max(k, 1) - 1])


def split_scores(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y) - np.asarray(mu))


def normalized_scores(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y) - np.asarray(mu)) / np.asarray(sigma)


def cqr_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    return np.maximum(np.asarray(q_lo) - y, y - np.asarray(q_hi))


def interval_split(mu: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return mu - q, mu + q


def interval_normalized(mu: np.ndarray, sigma: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return mu - q * sigma, mu + q * sigma


def interval_cqr(q_lo: np.ndarray, q_hi: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    return q_lo - q, q_hi + q


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def aci_cqr(
    y: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    init_scores: np.ndarray,
    alpha: float,
    gamma: float = 0.005,
    window: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online adaptive conformal inference on top of CQR scores.

    At each step t the interval is built ONLY from scores observed before t
    (initial calibration scores + realised test scores so far), using the
    current miscoverage level alpha_t. After y_t is revealed,
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t).

    `window` restricts the score pool to the most recent scores.
    Returns (lo, hi, alpha_t trajectory).
    """
    y, q_lo, q_hi = map(lambda a: np.asarray(a, dtype=float), (y, q_lo, q_hi))
    pool = list(np.asarray(init_scores, dtype=float))
    n = y.size
    lo, hi, alphas = np.empty(n), np.empty(n), np.empty(n)
    a_t = alpha
    max_score = float(np.max(pool))
    for t in range(n):
        alphas[t] = a_t
        recent = np.asarray(pool[-window:] if window else pool)
        if a_t >= 1:
            q = -np.inf
        elif a_t <= 0:
            q = np.inf
        else:
            q = conformal_quantile(recent, a_t)
        if not np.isfinite(q):
            # fall back to the widest / narrowest finite interval seen so far
            q = max_score * 1.5 if q > 0 else float(np.min(recent))
        lo[t], hi[t] = q_lo[t] - q, q_hi[t] + q
        err = 0.0 if lo[t] <= y[t] <= hi[t] else 1.0
        a_t = a_t + gamma * (alpha - err)
        s = max(q_lo[t] - y[t], y[t] - q_hi[t])
        pool.append(s)
        max_score = max(max_score, s)
    return lo, hi, alphas
