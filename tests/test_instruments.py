"""
Tests for quant_risk.instruments -- Bond.
Run with: pytest tests/test_instruments.py -v
"""

import pytest
from quant_risk.instruments.bond import Bond
from quant_risk.curves.ois import OISCurve


@pytest.fixture
def curve():
    return OISCurve.from_processed()


@pytest.fixture
def bund():
    return Bond(
        isin          = "DE0001102580",
        face_value    = 1_000_000,
        coupon_rate   = 2.60,
        issue_date    = "2023-08-15",
        maturity_date = "2034-08-15",
    )


class TestBond:

    def test_describe(self, bund):
        assert "DE0001102580" in bund.describe()
        assert "EUR" in bund.describe()

    def test_currency(self, bund):
        assert bund.currency == "EUR"

    def test_notional(self, bund):
        assert bund.notional == 1_000_000

    def test_price_keys(self, bund, curve):
        p = bund.price(curve)
        assert all(k in p for k in [
            "clean_price", "dirty_price", "accrued",
            "ytm", "duration", "convexity", "dv01"
        ])

    def test_dirty_above_clean(self, bund, curve):
        p = bund.price(curve)
        assert p["dirty_price"] > p["clean_price"]

    def test_accrued_positive(self, bund, curve):
        p = bund.price(curve)
        assert p["accrued"] > 0

    def test_ytm_positive(self, bund, curve):
        assert bund.price(curve)["ytm"] > 0

    def test_duration_positive(self, bund, curve):
        assert bund.price(curve)["duration"] > 0

    def test_dv01_positive(self, bund, curve):
        assert bund.dv01(curve) > 0

    def test_z_spread_positive(self, bund, curve):
        # bond priced below OIS theoretical -- positive z-spread
        p = bund.price(curve)
        zs = bund.z_spread(p["clean_price"] - 0.15, curve)
        assert zs > 0

    def test_z_spread_negative(self, bund, curve):
        # bond priced above OIS theoretical -- negative z-spread
        p = bund.price(curve)
        zs = bund.z_spread(p["clean_price"] + 0.15, curve)
        assert zs < 0

    def test_cash_flows_columns(self, bund):
        cf = bund.cash_flows()
        assert all(c in cf.columns for c in ["date", "amount", "type"])

    def test_cash_flows_principal(self, bund):
        cf = bund.cash_flows()
        assert (cf["type"] == "principal").sum() == 1

    def test_cash_flows_coupons(self, bund):
        cf = bund.cash_flows()
        assert (cf["type"] == "coupon").sum() > 0


# ------------------------------------------------------------------
# IRSwap tests
# ------------------------------------------------------------------

from quant_risk.instruments.swap import IRSwap

@pytest.fixture
def swap():
    return IRSwap(
        notional       = 10_000_000,
        maturity_years = 5,
        fixed_rate     = 2.50,
        valuation_date = "2026-03-24",
    )

class TestIRSwap:

    def test_describe(self, swap):
        assert "Payer" in swap.describe()
        assert "EUR" in swap.describe()

    def test_currency(self, swap):
        assert swap.currency == "EUR"

    def test_notional(self, swap):
        assert swap.notional == 10_000_000

    def test_par_rate_positive(self, swap, curve):
        assert swap.par_rate(curve) > 0

    def test_npv_at_par_is_zero(self, curve):
        par = IRSwap(
            notional       = 10_000_000,
            maturity_years = 5,
            fixed_rate     = 2.4889,
            valuation_date = "2026-03-24",
        )
        assert abs(par.price(curve)["npv"]) < 100

    def test_price_keys(self, swap, curve):
        p = swap.price(curve)
        assert all(k in p for k in [
            "npv", "fixed_leg_npv", "float_leg_npv",
            "par_rate", "fixed_rate"
        ])

    def test_fixed_plus_float_equals_npv(self, swap, curve):
        p = swap.price(curve)
        assert abs(p["fixed_leg_npv"] + p["float_leg_npv"] - p["npv"]) < 1

    def test_dv01_negative_for_payer(self, swap, curve):
        # payer swap loses value when rates rise
        assert swap.dv01(curve) < 0

    def test_dv01_positive_for_receiver(self, curve):
        receiver = IRSwap(
            notional       = 10_000_000,
            maturity_years = 5,
            fixed_rate     = 2.50,
            valuation_date = "2026-03-24",
            pay_fixed      = False,
        )
        assert receiver.dv01(curve) > 0

    def test_key_rate_dv01_5y_dominates(self, swap, curve):
        kr = swap.key_rate_dv01(curve)
        assert abs(kr["5Y"]) == max(abs(v) for v in kr.values())