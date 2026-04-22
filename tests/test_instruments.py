"""
Tests for quant_risk.instruments -- Bond.
Run with: pytest tests/test_instruments.py -v
"""

import pytest
import numpy as np
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


# ------------------------------------------------------------------
# CreditDefaultSwap tests
# ------------------------------------------------------------------

import QuantLib as ql
from quant_risk.instruments.cds import CreditDefaultSwap

@pytest.fixture
def valuation_date():
    d = ql.Date(24, 3, 2026)
    ql.Settings.instance().evaluationDate = d
    return d

@pytest.fixture
def maturity(valuation_date):
    return ql.Date(20, 3, 2031)   # 5Y

@pytest.fixture
def cds_flat(valuation_date, maturity):
    return CreditDefaultSwap.from_flat_spread(
        valuation_date = valuation_date,
        maturity       = maturity,
        notional_      = 10_000_000,
        par_spread     = 0.0150,   # 150bps
        coupon         = 0.0100,   # 100bps standard IG
        recovery       = 0.40,
    )

@pytest.fixture
def cds_bootstrapped(valuation_date, maturity, curve):
    return CreditDefaultSwap.from_market_quotes(
        valuation_date  = valuation_date,
        maturity        = maturity,
        notional_       = 10_000_000,
        market_tenors   = [1, 2, 3, 5, 7, 10],
        market_spreads  = [0.0080, 0.0110, 0.0130, 0.0150, 0.0165, 0.0180],
        coupon          = 0.0100,
        recovery        = 0.40,
    )

class TestCreditDefaultSwap:

    # ── basic properties ──────────────────────────────────────────────────────

    def test_currency(self, cds_flat):
        assert cds_flat.currency == 'EUR'

    def test_notional(self, cds_flat):
        assert cds_flat.notional == 10_000_000

    def test_describe(self, cds_flat):
        d = cds_flat.describe()
        assert 'Buyer' in d
        assert 'EUR' in d
        assert '150' in d or '100' in d

    # ── pricing ───────────────────────────────────────────────────────────────

    def test_npv_positive_for_buyer_spread_above_coupon(self, cds_flat, curve):
        # par spread 150bps > coupon 100bps -- buyer has positive NPV
        assert cds_flat.price(curve) > 0

    def test_npv_negative_for_seller(self, valuation_date, maturity, curve):
        cds_seller = CreditDefaultSwap.from_flat_spread(
            valuation_date  = valuation_date,
            maturity        = maturity,
            notional_       = 10_000_000,
            par_spread      = 0.0150,
            coupon          = 0.0100,
            protection_buyer = False,
        )
        assert cds_seller.price(curve) < 0

    def test_npv_zero_at_par_spread_equals_coupon(self, valuation_date, maturity, curve):
        # par spread = coupon = 100bps -- NPV should be near zero
        cds_par = CreditDefaultSwap.from_flat_spread(
            valuation_date = valuation_date,
            maturity       = maturity,
            notional_      = 10_000_000,
            par_spread     = 0.0100,
            coupon         = 0.0100,
        )
        assert abs(cds_par.price(curve)) < 5000   # small residual from OIS discounting

    def test_par_spread_recovers_input(self, cds_flat, curve):
        # par spread from pricer should match the 150bps input
        assert abs(cds_flat.par_spread(curve) - 0.0150) < 0.0005

    # ── upfront and PUF ───────────────────────────────────────────────────────

    def test_upfront_positive_when_spread_above_coupon(self, cds_flat, curve):
        # spread 150bps > coupon 100bps -- buyer pays upfront
        assert cds_flat.upfront(curve) > 0

    def test_upfront_negative_when_spread_below_coupon(self, valuation_date, maturity, curve):
        cds_tight = CreditDefaultSwap.from_flat_spread(
            valuation_date = valuation_date,
            maturity       = maturity,
            notional_      = 10_000_000,
            par_spread     = 0.0060,   # 60bps < 100bps coupon
            coupon         = 0.0100,
        )
        assert cds_tight.upfront(curve) < 0

    def test_puf_sign_consistent_with_upfront(self, cds_flat, curve):
        assert cds_flat.puf(curve) > 0   # spread > coupon

    # ── sensitivities ─────────────────────────────────────────────────────────

    def test_cs01_positive_for_buyer(self, cds_flat, curve):
        # protection buyer gains when spreads widen
        assert cds_flat.cs01(curve) > 0

    def test_cs01_negative_for_seller(self, valuation_date, maturity, curve):
        cds_seller = CreditDefaultSwap.from_flat_spread(
            valuation_date   = valuation_date,
            maturity         = maturity,
            notional_        = 10_000_000,
            par_spread       = 0.0150,
            coupon           = 0.0100,
            protection_buyer = False,
        )
        assert cds_seller.cs01(curve) < 0

    def test_ir_pv01_small_relative_to_cs01(self, cds_flat, curve):
        # IR PV01 should be much smaller than CS01 for CDS
        assert abs(cds_flat.dv01(curve)) < abs(cds_flat.cs01(curve))

    def test_duration_positive(self, cds_flat, curve):
        # risky annuity always positive
        assert cds_flat.duration(curve) > 0

    def test_duration_less_than_maturity(self, cds_flat, curve):
        # risky annuity < maturity in years due to default risk discounting
        assert cds_flat.duration(curve) < 5.0

    # ── jump to default ───────────────────────────────────────────────────────

    def test_jtd_keys(self, cds_flat, curve):
        jtd = cds_flat.jump_to_default(curve)
        assert all(k in jtd for k in ['lgd', 'pv_premium', 'jtd_buyer', 'jtd_seller'])

    def test_lgd_equals_one_minus_recovery_times_notional(self, cds_flat, curve):
        jtd = cds_flat.jump_to_default(curve)
        assert abs(jtd['lgd'] - 0.60 * 10_000_000) < 1

    def test_jtd_buyer_positive(self, cds_flat, curve):
        # default is a gain for the protection buyer
        assert cds_flat.jump_to_default(curve)['jtd_buyer'] > 0

    def test_jtd_seller_negative(self, cds_flat, curve):
        # default is a loss for the protection seller
        assert cds_flat.jump_to_default(curve)['jtd_seller'] < 0

    def test_jtd_buyer_plus_seller_zero(self, cds_flat, curve):
        jtd = cds_flat.jump_to_default(curve)
        assert abs(jtd['jtd_buyer'] + jtd['jtd_seller']) < 1

    # ── cash flows ────────────────────────────────────────────────────────────

    def test_cash_flows_columns(self, cds_flat):
        cf = cds_flat.cash_flows()
        assert all(c in cf.columns for c in ['date', 'amount', 'type'])

    def test_cash_flows_all_premium(self, cds_flat):
        cf = cds_flat.cash_flows()
        assert (cf['type'] == 'premium').all()

    def test_cash_flows_quarterly(self, cds_flat):
        # 5Y CDS -- approximately 20 quarterly payments
        cf = cds_flat.cash_flows()
        assert 18 <= len(cf) <= 22

    def test_cash_flows_amounts_positive(self, cds_flat):
        cf = cds_flat.cash_flows()
        assert (cf['amount'] > 0).all()

    # ── bootstrapped curve ────────────────────────────────────────────────────

    def test_bootstrapped_npv_finite(self, cds_bootstrapped, curve):
        assert np.isfinite(cds_bootstrapped.price(curve))

    def test_bootstrapped_cs01_positive_for_buyer(self, cds_bootstrapped, curve):
        assert cds_bootstrapped.cs01(curve) > 0

    def test_bootstrapped_par_spread_at_5y(self, cds_bootstrapped, curve):
        # 5Y market spread is 150bps -- par spread should be close
        assert abs(cds_bootstrapped.par_spread(curve) - 0.0150) < 0.0010