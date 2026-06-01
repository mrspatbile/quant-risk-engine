"""
Tests for quant_risk.utils.calibration — realized_volatility, realized_correlation.

All tests use synthetic price series. No API calls, no files.
"""

import numpy as np
import pandas as pd
import pytest

from quant_risk.utils.calibration import realized_volatility, realized_correlation


# ── synthetic price series ────────────────────────────────────────────────────

def _geometric_brownian(S0: float, mu: float, sigma: float, n: int, seed: int) -> pd.Series:
    rng  = np.random.default_rng(seed)
    dt   = 1 / 252
    Z    = rng.standard_normal(n)
    log_r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = S0 * np.exp(np.cumsum(log_r))
    prices = np.concatenate([[S0], prices])
    idx = pd.date_range("2024-01-01", periods=n + 1, freq="B")
    return pd.Series(prices, index=idx)


@pytest.fixture
def prices_a():
    return _geometric_brownian(100.0, 0.08, 0.20, 504, seed=1)


@pytest.fixture
def prices_b_high_corr(prices_a):
    # asset B shares the same log returns as A plus tiny independent noise → high correlation
    rng = np.random.default_rng(42)
    log_r_a = np.log(prices_a / prices_a.shift(1)).dropna().values
    noise   = rng.standard_normal(len(log_r_a)) * 0.001   # 10× smaller than A's returns
    log_r_b = log_r_a + noise
    prices  = prices_a.iloc[0] * np.exp(np.concatenate([[0.0], np.cumsum(log_r_b)]))
    return pd.Series(prices, index=prices_a.index)


@pytest.fixture
def prices_b_uncorr():
    return _geometric_brownian(50.0, 0.05, 0.15, 504, seed=99)


# ── realized_volatility ───────────────────────────────────────────────────────

class TestRealizedVolatility:

    def test_returns_float(self, prices_a):
        assert isinstance(realized_volatility(prices_a), float)

    def test_positive(self, prices_a):
        assert realized_volatility(prices_a) > 0

    def test_approx_input_sigma(self, prices_a):
        # generated with sigma=0.20; realised vol should be in reasonable range
        vol = realized_volatility(prices_a)
        assert 0.10 < vol < 0.35

    def test_window_subset(self, prices_a):
        vol_full   = realized_volatility(prices_a)
        vol_window = realized_volatility(prices_a, window=63)
        # Both are valid vols — just check they're reasonable
        assert vol_window > 0
        assert isinstance(vol_window, float)

    def test_window_shorter_series(self, prices_a):
        # window=20 uses only last 20 returns
        vol = realized_volatility(prices_a, window=20)
        assert vol > 0

    def test_annualisation_scales(self, prices_a):
        vol_252 = realized_volatility(prices_a, annualisation=252)
        vol_365 = realized_volatility(prices_a, annualisation=365)
        # vol_365 > vol_252 (scaled by sqrt(365/252))
        assert vol_365 > vol_252

    def test_fewer_than_two_obs_raises(self):
        with pytest.raises(ValueError):
            realized_volatility(pd.Series([100.0]))


# ── realized_correlation ──────────────────────────────────────────────────────

class TestRealizedCorrelation:

    def test_returns_float(self, prices_a, prices_b_uncorr):
        assert isinstance(realized_correlation(prices_a, prices_b_uncorr), float)

    def test_in_range(self, prices_a, prices_b_uncorr):
        rho = realized_correlation(prices_a, prices_b_uncorr)
        assert -1.0 <= rho <= 1.0

    def test_high_corr_near_one(self, prices_a, prices_b_high_corr):
        rho = realized_correlation(prices_a, prices_b_high_corr)
        assert rho > 0.90

    def test_self_corr_is_one(self, prices_a):
        rho = realized_correlation(prices_a, prices_a)
        assert abs(rho - 1.0) < 1e-10

    def test_window_subset(self, prices_a, prices_b_uncorr):
        rho = realized_correlation(prices_a, prices_b_uncorr, window=63)
        assert -1.0 <= rho <= 1.0

    def test_unaligned_index_uses_common(self):
        idx_a = pd.date_range("2024-01-01", periods=100, freq="B")
        idx_b = pd.date_range("2024-02-01", periods=100, freq="B")
        s_a   = pd.Series(np.random.default_rng(1).random(100) + 99, index=idx_a)
        s_b   = pd.Series(np.random.default_rng(2).random(100) + 49, index=idx_b)
        # Should not raise — uses common dates
        rho = realized_correlation(s_a, s_b)
        assert -1.0 <= rho <= 1.0

    def test_no_common_obs_raises(self):
        idx_a = pd.date_range("2023-01-01", periods=10, freq="B")
        idx_b = pd.date_range("2024-01-01", periods=10, freq="B")
        s_a   = pd.Series(range(10), index=idx_a, dtype=float)
        s_b   = pd.Series(range(10), index=idx_b, dtype=float)
        with pytest.raises(ValueError):
            realized_correlation(s_a, s_b)

    def test_window_too_small_raises(self, prices_a, prices_b_uncorr):
        with pytest.raises(ValueError):
            realized_correlation(prices_a, prices_b_uncorr, window=1)
