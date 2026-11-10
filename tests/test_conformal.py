"""Tests for the conformal machinery (uses the lead's settled conformal.py)."""
import numpy as np

from uaagent.conformal import conformal_quantile, aci_cqr
from uaagent.marine import episodes


def test_conformal_quantile_finite_sample_index():
    scores = np.arange(1, 20, dtype=float)
    # (n+1)(1-alpha) = 20*0.9 = 18 -> 18th smallest score
    assert conformal_quantile(scores, 0.1) == 18.0
    # (n+1)(1-alpha) = 20*0.95 = 19 -> 19th smallest
    assert conformal_quantile(scores, 0.05) == 19.0
    # n=9, alpha=0.05: ceil(10*0.95)=10 > 9 -> cannot certify -> +inf
    assert np.isinf(conformal_quantile(np.arange(1, 10, dtype=float), 0.05))


def test_split_conformal_coverage_synthetic():
    rng = np.random.default_rng(0)
    cal_scores = np.abs(rng.normal(0.0, 1.0, 2000))
    mu_test = np.zeros(5000)
    y_test = rng.normal(0.0, 1.0, 5000)
    q = conformal_quantile(cal_scores, 0.1)
    cov = float(np.mean(np.abs(y_test - mu_test) <= q))
    assert cov >= 0.88
    assert cov <= 0.95  # sanity: not absurdly conservative


def test_aci_tracks_doubling_noise():
    rng = np.random.default_rng(1)
    y = np.r_[rng.normal(0.0, 1.0, 3000), rng.normal(0.0, 2.0, 3000)]
    init_scores = np.abs(rng.normal(0.0, 1.0, 1000))
    q_lo = np.zeros(y.size)
    q_hi = np.zeros(y.size)
    lo, hi, _ = aci_cqr(y, q_lo, q_hi, init_scores, alpha=0.1, window=1000)
    cov_aci = float(np.mean((y >= lo) & (y <= hi)))
    assert abs(cov_aci - 0.9) <= 0.02
    # static CQR cannot adapt: falls well below target after the shift
    q_static = conformal_quantile(init_scores, 0.1)
    cov_static = float(np.mean(np.abs(y) <= q_static))
    assert cov_static < 0.85


def test_episodes_debounce():
    flag = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    state = episodes(flag, persist=3)
    assert state[2] == False and state[4] == True   # raises on 3rd True
    assert state[6] == True and state[8] == False   # clears on 3rd False
    assert not state[9:] .any()                     # single blip never raises
    from uaagent.marine import count_raises
    assert count_raises(state) == [4]
