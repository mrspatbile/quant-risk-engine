"""
Tests for quant_risk.curves -- OISCurve and NSSCurve.

All tests use saved processed files or synthetic data -- no ECB API calls.
Run with: pytest tests/test_curves.py -v
"""

import pytest
import numpy as np
import pandas as pd
from quant_risk.curves.base import DiscountCurve
from quant_risk.curves.ois import OISCurve
from quant_risk.curves.nss import NSSCurve


# ------------------------------------------------------------------
# OISCurve tests
# ------------------------------------------------------------------

class TestOISCurve:

    def test_loads(self, ois_curve):
        assert ois_curve is not None

    def test_currency(self, ois_curve):
        assert ois_curve.currency == "EUR"

    def test_valuation_date_format(self, ois_curve):
        # should be YYYY-MM-DD
        parts = ois_curve.valuation_date.split("-")
        assert len(parts) == 3

    def test_discount_factor_at_zero(self, ois_curve):
        # P(0,0) should be 1
        assert abs(ois_curve.discount(1/365) - 1.0) < 0.01

    def test_discount_factor_decreasing(self, ois_curve):
        # discount factors must be monotonically decreasing
        maturities = [1, 2, 5, 10]
        dfs = [ois_curve.discount(T) for T in maturities]
        assert all(dfs[i] > dfs[i+1] for i in range(len(dfs)-1))

    def test_discount_factor_between_zero_and_one(self, ois_curve):
        for T in [1, 5, 10]:
            df = ois_curve.discount(T)
            assert 0 < df < 1

    def test_zero_rate_positive(self, ois_curve):
        for T in [1, 5, 10]:
            assert ois_curve.zero_rate(T) > 0

    def test_forward_rate_consistent_with_discount(self, ois_curve):
        # P(0,T2) = P(0,T1) * exp(-f(T1,T2) * (T2-T1) / 100)
        T1, T2 = 2.0, 5.0
        df1 = ois_curve.discount(T1)
        df2 = ois_curve.discount(T2)
        fwd = ois_curve.forward_rate(T1, T2)
        df2_implied = df1 * np.exp(-fwd / 100 * (T2 - T1))
        assert abs(df2 - df2_implied) < 1e-3

    def test_instantaneous_forward_positive(self, ois_curve):
        for T in [1, 5, 10]:
            assert ois_curve.instantaneous_forward(T) > 0

    def test_discount_vector(self, ois_curve):
        maturities = np.array([1.0, 2.0, 5.0])
        dfs = ois_curve.discount_vector(maturities)
        assert len(dfs) == 3
        assert all(0 < df < 1 for df in dfs)

    def test_describe(self, ois_curve):
        desc = ois_curve.describe()
        assert "OISCurve" in desc
        assert "EUR" in desc


# ------------------------------------------------------------------
# NSSCurve tests
# ------------------------------------------------------------------

@pytest.fixture
def nss_params():
    """Synthetic NSS parameters -- no API call needed."""
    return {
        "beta0": 1.205,
        "beta1": 0.758,
        "beta2": 1.826,
        "beta3": 7.616,
        "tau1" : 0.842,
        "tau2" : 14.951,
    }

@pytest.fixture
def nss_curve(nss_params):
    return NSSCurve(nss_params, valuation_date="2026-03-24")


class TestNSSCurve:

    def test_loads(self, nss_curve):
        assert nss_curve is not None

    def test_currency(self, nss_curve):
        assert nss_curve.currency == "EUR"

    def test_valuation_date(self, nss_curve):
        assert nss_curve.valuation_date == "2026-03-24"

    def test_short_rate(self, nss_curve, nss_params):
        expected = nss_params["beta0"] + nss_params["beta1"]
        assert abs(nss_curve.short_rate - expected) < 1e-3

    def test_long_rate(self, nss_curve, nss_params):
        assert abs(nss_curve.long_rate - nss_params["beta0"]) < 1e-3

    def test_asymptote(self, nss_curve, nss_params):
        # at very long maturities zero rate -> beta0
        assert abs(nss_curve.zero_rate(1000) - nss_params["beta0"]) < 0.15

    def test_discount_factor_decreasing(self, nss_curve):
        maturities = [1, 2, 5, 10, 20]
        dfs = [nss_curve.discount(T) for T in maturities]
        assert all(dfs[i] > dfs[i+1] for i in range(len(dfs)-1))

    def test_discount_factor_between_zero_and_one(self, nss_curve):
        for T in [1, 5, 10, 20]:
            df = nss_curve.discount(T)
            assert 0 < df < 1

    def test_forward_rate_consistent(self, nss_curve):
        T1, T2 = 2.0, 5.0
        df1 = nss_curve.discount(T1)
        df2 = nss_curve.discount(T2)
        fwd = nss_curve.forward_rate(T1, T2)
        df2_implied = df1 * np.exp(-fwd / 100 * (T2 - T1))
        assert abs(df2 - df2_implied) < 1e-3

    def test_parameters_roundtrip(self, nss_curve, nss_params):
        for key, val in nss_params.items():
            assert abs(nss_curve.parameters[key] - val) < 1e-10

    def test_zero_rate_at_known_maturity(self, nss_curve):
        # 10Y rate should be approximately 3.07% for these parameters
        zr = nss_curve.zero_rate(10.0)
        assert 2.5 < zr < 3.5

    def test_describe(self, nss_curve):
        desc = nss_curve.describe()
        assert "NSSCurve" in desc
        assert "EUR" in desc
        
    @pytest.mark.skip(reason="live ECB API")
    def test_from_ecb_with_date(self):
        nss = NSSCurve.from_ecb(rating='AAA', last_n=60, date='2026-03-24')
        assert nss.valuation_date <= '2026-03-24'


# ------------------------------------------------------------------
# DiscountCurve._select_params_row tests
# ------------------------------------------------------------------

class TestSelectParamsRow:

    @pytest.fixture
    def sample_params(self):
        """Time series of sample parameters indexed by date."""
        dates = pd.date_range("2026-03-20", periods=5, freq="D")
        data = pd.DataFrame(
            {"value": [1.0, 2.0, 3.0, 4.0, 5.0]},
            index=dates
        )
        return data

    def test_select_last_row_when_date_none(self, sample_params):
        row, val_date = DiscountCurve._select_params_row(sample_params, date=None)
        assert val_date == "2026-03-24"
        assert row["value"] == 5.0

    def test_select_exact_date_match(self, sample_params):
        row, val_date = DiscountCurve._select_params_row(sample_params, date="2026-03-22")
        assert val_date == "2026-03-22"
        assert row["value"] == 3.0

    def test_fallback_to_prior_date(self, sample_params):
        # Request 2026-03-23 12:00 (between 2026-03-23 00:00 and 2026-03-24 00:00)
        # Should fall back to 2026-03-23 00:00
        row, val_date = DiscountCurve._select_params_row(sample_params, date="2026-03-23 12:00:00")
        assert val_date == "2026-03-23"
        assert row["value"] == 4.0

    def test_timestamp_input(self, sample_params):
        ts = pd.Timestamp("2026-03-21")
        row, val_date = DiscountCurve._select_params_row(sample_params, date=ts)
        assert val_date == "2026-03-21"
        assert row["value"] == 2.0

    def test_no_data_before_target_raises_error(self, sample_params):
        with pytest.raises(ValueError, match="No data available"):
            DiscountCurve._select_params_row(sample_params, date="2026-03-19")

    def test_normalises_string_dates_in_index(self):
        # Index as strings, not Timestamps
        data = pd.DataFrame(
            {"value": [1.0, 2.0]},
            index=["2026-03-21", "2026-03-22"]
        )
        row, val_date = DiscountCurve._select_params_row(data, date="2026-03-22")
        assert val_date == "2026-03-22"
        assert row["value"] == 2.0

    def test_returns_iso_string_format(self, sample_params):
        _, val_date = DiscountCurve._select_params_row(sample_params, date=None)
        # Should be YYYY-MM-DD
        parts = val_date.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year
        assert len(parts[1]) == 2  # month
        assert len(parts[2]) == 2  # day