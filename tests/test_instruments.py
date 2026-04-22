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



# ------------------------------------------------------------------
# FXForward tests
# ------------------------------------------------------------------

import QuantLib as ql
from quant_risk.instruments.fx_forward import FXForward

@pytest.fixture
def fx_setup(valuation_date):
    """Build USD discount handle matching notebook construction."""
    day_count_fx = ql.Actual360()
    us_calendar  = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

    # USD rates from notebook -- SOFR + Treasury CMT March 24 2026
    usd_dates = [
        valuation_date,
        valuation_date + ql.Period(91,  ql.Days),   # 3M
        valuation_date + ql.Period(182, ql.Days),   # 6M
        valuation_date + ql.Period(365, ql.Days),   # 1Y
        valuation_date + ql.Period(730, ql.Days),   # 2Y
        valuation_date + ql.Period(1825, ql.Days),  # 5Y
        valuation_date + ql.Period(3650, ql.Days),  # 10Y
    ]
    usd_rates = [
        0.03630,   # SOFR overnight
        0.03740,   # 3M
        0.03780,   # 6M
        0.03810,   # 1Y
        0.03900,   # 2Y
        0.04030,   # 5Y
        0.04390,   # 10Y
    ]
    usd_zc = ql.ZeroCurve(
        usd_dates, usd_rates, day_count_fx,
        us_calendar, ql.Linear(), ql.Continuous
    )
    usd_zc.enableExtrapolation()
    usd_handle = ql.YieldTermStructureHandle(usd_zc)

    spot_eurusd = 1.1578
    return usd_handle, spot_eurusd

@pytest.fixture
def fx_maturity(valuation_date):
    return valuation_date + ql.Period(365, ql.Days)   # 1Y

@pytest.fixture
def fwd_inception(valuation_date, fx_maturity, fx_setup):
    usd_handle, spot = fx_setup
    return FXForward.at_inception(
        notional_foreign = 1_000_000,
        spot_rate        = spot,
        maturity_date    = fx_maturity,
        valuation_date   = valuation_date,
        usd_disc_handle  = usd_handle,
    )

@pytest.fixture
def fwd_seasoned(valuation_date, fx_maturity, fx_setup):
    usd_handle, spot = fx_setup
    return FXForward.seasoned(
        notional_foreign = 1_000_000,
        spot_rate        = spot,
        f0               = 1.1792,   # contractual rate from notebook
        maturity_date    = fx_maturity,
        valuation_date   = valuation_date,
        usd_disc_handle  = usd_handle,
    )

class TestFXForward:

    # ── basic properties ──────────────────────────────────────────────────────

    def test_currency(self, fwd_inception):
        assert fwd_inception.currency == 'EUR'

    def test_notional(self, fwd_inception):
        assert fwd_inception.notional == 1_000_000

    def test_describe_contains_key_info(self, fwd_inception):
        d = fwd_inception.describe()
        assert 'USD' in d
        assert 'EUR' in d
        assert 'Exporter' in d

    # ── forward rate ──────────────────────────────────────────────────────────

    def test_forward_rate_above_spot(self, fwd_inception, curve):
        # USD rates > EUR rates -- forward EUR/USD > spot
        assert fwd_inception.forward_rate(curve) > fwd_inception._spot_rate

    def test_forward_rate_close_to_notebook(self, fwd_inception, curve):
        # notebook shows 1Y forward ~1.1792
        assert abs(fwd_inception.forward_rate(curve) - 1.1792) < 0.01

    # ── NPV ───────────────────────────────────────────────────────────────────

    def test_npv_zero_at_inception(self, fwd_inception, curve):
        # forward priced at market rate -- NPV should be near zero
        assert abs(fwd_inception.price(curve)) < 100

    def test_npv_positive_when_eur_weakens(self, valuation_date, fx_maturity, fx_setup, curve):
        # F_0 > F_t means EUR weakened -- exporter gains
        usd_handle, spot = fx_setup
        fwd = FXForward.seasoned(
            notional_foreign = 1_000_000,
            spot_rate        = spot,
            f0               = 1.2500,   # locked in high rate -- EUR weaker than now
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
        )
        assert fwd.price(curve) > 0

    def test_npv_negative_when_eur_strengthens(self, valuation_date, fx_maturity, fx_setup, curve):
        # F_0 < F_t means EUR strengthened -- exporter loses
        usd_handle, spot = fx_setup
        fwd = FXForward.seasoned(
            notional_foreign = 1_000_000,
            spot_rate        = spot,
            f0               = 1.0500,   # locked in low rate -- EUR stronger than now
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
        )
        assert fwd.price(curve) < 0

    def test_npv_flips_for_importer(self, valuation_date, fx_maturity, fx_setup, curve):
        usd_handle, spot = fx_setup
        fwd_imp = FXForward.seasoned(
            notional_foreign = 1_000_000,
            spot_rate        = spot,
            f0               = 1.2500,
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
            exporter         = False,
        )
        fwd_exp = FXForward.seasoned(
            notional_foreign = 1_000_000,
            spot_rate        = spot,
            f0               = 1.2500,
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
            exporter         = True,
        )
        assert abs(fwd_imp.price(curve) + fwd_exp.price(curve)) < 1

    # ── delta FX ──────────────────────────────────────────────────────────────

    def test_delta_fx_negative_for_exporter(self, fwd_seasoned, curve):
        # exporter loses when EUR strengthens (spot rises)
        assert fwd_seasoned.delta_fx(curve) < 0

    def test_delta_fx_analytical_vs_numerical(self, fwd_seasoned, curve):
        # analytical and numerical should be close
        analytical = fwd_seasoned.delta_fx(curve)
        numerical  = fwd_seasoned.delta_fx_numerical(curve)
        assert abs(analytical - numerical) / abs(numerical) < 0.15   # within 15%

    def test_delta_fx_scales_with_notional(self, valuation_date, fx_maturity, fx_setup, curve):
        usd_handle, spot = fx_setup
        fwd_large = FXForward.seasoned(
            notional_foreign = 10_000_000,
            spot_rate        = spot,
            f0               = 1.1792,
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
        )
        fwd_small = FXForward.seasoned(
            notional_foreign = 1_000_000,
            spot_rate        = spot,
            f0               = 1.1792,
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
        )
        assert abs(fwd_large.delta_fx(curve) / fwd_small.delta_fx(curve) - 10) < 0.01

    # ── delta IR ──────────────────────────────────────────────────────────────

    def test_delta_ir_eur_negative_for_exporter(self, fwd_seasoned, curve):
        # when EUR rates rise, F_t falls, (F_0 - F_t) rises -- exporter gains
        # dv01 measures EUR rate sensitivity -- positive for exporter
        assert fwd_seasoned.dv01(curve) > 0

    def test_delta_ir_usd_positive_for_exporter(self, fwd_seasoned, curve):
        # when USD rates rise, F_t rises, (F_0 - F_t) falls -- exporter loses
        assert fwd_seasoned.delta_ir_usd(curve) > 0

    def test_delta_ir_small_relative_to_delta_fx(self, fwd_seasoned, curve):
        # IR sensitivity (1bp) << FX sensitivity (1%)
        assert abs(fwd_seasoned.dv01(curve)) < abs(fwd_seasoned.delta_fx(curve))

    # ── duration ─────────────────────────────────────────────────────────────

    def test_duration_close_to_one_year(self, fwd_seasoned, curve):
        assert abs(fwd_seasoned.duration(curve) - 1.0) < 0.05

    # ── hedge effectiveness ───────────────────────────────────────────────────

    def test_hedge_effectiveness_keys(self, fwd_seasoned, curve):
        h = fwd_seasoned.hedge_effectiveness(10_000_000, curve)
        assert all(k in h for k in [
            'portfolio_eur', 'delta_unhedged', 'delta_hedge',
            'delta_hedged', 'effectiveness_pct', 'hedge_ratio_pct'
        ])

    def test_hedge_effectiveness_below_100(self, fwd_seasoned, curve):
        h = fwd_seasoned.hedge_effectiveness(10_000_000, curve)
        assert 0 < h['effectiveness_pct'] < 100

    def test_hedge_reduces_delta(self, fwd_seasoned, curve):
        h = fwd_seasoned.hedge_effectiveness(10_000_000, curve)
        assert abs(h['delta_hedged']) < abs(h['delta_unhedged'])

    # ── basis ─────────────────────────────────────────────────────────────────

    def test_basis_cost_zero_when_no_basis(self, valuation_date, fx_maturity, fx_setup, curve):
        usd_handle, spot = fx_setup
        fwd_no_basis = FXForward.seasoned(
            notional_foreign = 1_000_000,
            spot_rate        = spot,
            f0               = 1.1792,
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
            basis_bps        = 0.0,
        )
        assert abs(fwd_no_basis.basis_cost(curve)) < 1

    def test_basis_cost_positive_for_negative_basis(self, valuation_date, fx_maturity, fx_setup, curve):
        usd_handle, spot = fx_setup
        fwd_basis = FXForward.seasoned(
            notional_foreign = 10_000_000,
            spot_rate        = spot,
            f0               = 1.1792,
            maturity_date    = fx_maturity,
            valuation_date   = valuation_date,
            usd_disc_handle  = usd_handle,
            basis_bps        = -20.0,
        )
        assert fwd_basis.basis_cost(curve) > 0

    # ── cash flows ────────────────────────────────────────────────────────────

    def test_cash_flows_columns(self, fwd_seasoned):
        cf = fwd_seasoned.cash_flows()
        assert all(c in cf.columns for c in ['date', 'amount', 'type'])

    def test_cash_flows_single_settlement(self, fwd_seasoned):
        assert len(fwd_seasoned.cash_flows()) == 1

    def test_cash_flows_type_settlement(self, fwd_seasoned):
        assert fwd_seasoned.cash_flows()['type'].iloc[0] == 'settlement'


# ------------------------------------------------------------------
# VanillaOption tests -- append to tests/test_instruments.py
# ------------------------------------------------------------------

from quant_risk.instruments.option import VanillaOption

@pytest.fixture
def opt_valuation_date():
    d = ql.Date(24, 3, 2026)
    ql.Settings.instance().evaluationDate = d
    return d

@pytest.fixture
def opt_expiry(opt_valuation_date):
    return opt_valuation_date + ql.Period(91, ql.Days)   # ~3M

@pytest.fixture
def call_option(opt_valuation_date, opt_expiry):
    return VanillaOption(
        spot          = 5250.0,
        strike        = 5250.0,   # ATM
        expiry_date   = opt_expiry,
        valuation_date = opt_valuation_date,
        sigma         = 0.165,
        option_type   = 'call',
        notional_     = 1000.0,
        div_yield     = 0.030,
    )

@pytest.fixture
def put_option(opt_valuation_date, opt_expiry):
    return VanillaOption(
        spot          = 5250.0,
        strike        = 5250.0,   # ATM
        expiry_date   = opt_expiry,
        valuation_date = opt_valuation_date,
        sigma         = 0.165,
        option_type   = 'put',
        notional_     = 1000.0,
        div_yield     = 0.030,
    )


class TestVanillaOption:

    # ── basic properties ──────────────────────────────────────────────────────

    def test_currency(self, call_option):
        assert call_option.currency == 'EUR'

    def test_notional(self, call_option):
        assert call_option.notional == 1000.0

    def test_describe_call(self, call_option):
        d = call_option.describe()
        assert 'Call' in d
        assert 'EUR' in d
        assert '5250' in d

    def test_describe_put(self, put_option):
        assert 'Put' in put_option.describe()

    # ── pricing ───────────────────────────────────────────────────────────────

    def test_call_price_positive(self, call_option, curve):
        assert call_option.price(curve) > 0

    def test_put_price_positive(self, put_option, curve):
        assert put_option.price(curve) > 0

    def test_call_price_above_intrinsic(self, call_option, curve):
        # time value -- price > intrinsic (0 for ATM)
        assert call_option.price(curve) > 0

    def test_itm_call_above_atm_call(self, opt_valuation_date, opt_expiry, curve):
        atm = VanillaOption(5250, 5250, opt_expiry, opt_valuation_date, 0.165, 'call', 1.0, 0.03)
        itm = VanillaOption(5250, 4750, opt_expiry, opt_valuation_date, 0.165, 'call', 1.0, 0.03)
        assert itm.price(curve) > atm.price(curve)

    def test_otm_call_below_atm_call(self, opt_valuation_date, opt_expiry, curve):
        atm = VanillaOption(5250, 5250, opt_expiry, opt_valuation_date, 0.165, 'call', 1.0, 0.03)
        otm = VanillaOption(5250, 5750, opt_expiry, opt_valuation_date, 0.165, 'call', 1.0, 0.03)
        assert otm.price(curve) < atm.price(curve)

    def test_put_call_parity(self, call_option, put_option, curve):
        # C - P = S*e^(-qT) - K*e^(-rT)
        C = call_option.price(curve) / call_option.notional
        P = put_option.price(curve)  / put_option.notional
        S = 5250.0
        K = 5250.0
        T = call_option._T
        q = 0.030
        disc_handle = call_option._disc_handle_from_curve(curve)
        r = -np.log(disc_handle.discount(call_option._expiry_date)) / T
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((C - P) - rhs) < 0.01

    def test_price_increases_with_vol(self, opt_valuation_date, opt_expiry, curve):
        low_vol  = VanillaOption(5250, 5250, opt_expiry, opt_valuation_date, 0.10, 'call', 1.0, 0.03)
        high_vol = VanillaOption(5250, 5250, opt_expiry, opt_valuation_date, 0.30, 'call', 1.0, 0.03)
        assert high_vol.price(curve) > low_vol.price(curve)

    # ── greeks ────────────────────────────────────────────────────────────────

    def test_call_delta_between_zero_and_one(self, call_option, curve):
        d = call_option.delta(curve) / call_option.notional
        assert 0 < d < 1

    def test_put_delta_between_minus_one_and_zero(self, put_option, curve):
        d = put_option.delta(curve) / put_option.notional
        assert -1 < d < 0

    def test_atm_delta_close_to_half(self, call_option, curve):
        d = call_option.delta(curve) / call_option.notional
        assert abs(d - 0.5) < 0.10

    def test_gamma_positive_for_long_call(self, call_option, curve):
        assert call_option.gamma(curve) > 0

    def test_gamma_positive_for_long_put(self, put_option, curve):
        assert put_option.gamma(curve) > 0

    def test_call_put_gamma_equal(self, call_option, put_option, curve):
        # gamma is same for call and put with same parameters
        assert abs(call_option.gamma(curve) - put_option.gamma(curve)) < 0.001

    def test_vega_positive_for_long_call(self, call_option, curve):
        assert call_option.vega(curve) > 0

    def test_vega_positive_for_long_put(self, put_option, curve):
        assert put_option.vega(curve) > 0

    def test_call_put_vega_equal(self, call_option, put_option, curve):
        assert abs(call_option.vega(curve) - put_option.vega(curve)) < 0.01

    def test_theta_negative_for_long_call(self, call_option, curve):
        assert call_option.theta(curve) < 0

    def test_theta_negative_for_long_put(self, put_option, curve):
        assert put_option.theta(curve) < 0

    def test_rho_positive_for_call(self, call_option, curve):
        assert call_option.dv01(curve) > 0

    def test_rho_negative_for_put(self, put_option, curve):
        assert put_option.dv01(curve) < 0

    def test_duration_close_to_expiry_tenor(self, call_option, curve):
        assert abs(call_option.duration(curve) - 0.25) < 0.05

    # ── greeks summary ────────────────────────────────────────────────────────

    def test_greeks_summary_keys(self, call_option, curve):
        g = call_option.greeks_summary(curve)
        assert all(k in g for k in ['price', 'delta', 'gamma', 'vega', 'theta', 'rho'])

    # ── implied volatility ────────────────────────────────────────────────────

    def test_implied_vol_round_trip(self, call_option, curve):
        disc_handle  = call_option._disc_handle_from_curve(curve)
        r            = -np.log(disc_handle.discount(call_option._expiry_date)) / call_option._T
        market_price = call_option._bsm_price(r)
        iv           = call_option.implied_vol(market_price, r=r)
        assert abs(iv - call_option._sigma) < 0.0001

    # ── FRTB curvature ────────────────────────────────────────────────────────

    def test_frtb_curvature_keys(self, call_option, curve):
        cvr = call_option.frtb_curvature(curve)
        assert all(k in cvr for k in ['V', 'V_up', 'V_down', 'cvr_up', 'cvr_down', 'cvr'])

    def test_frtb_cvr_non_negative(self, call_option, curve):
        # CVR is always >= 0 for long options (positive gamma)
        assert call_option.frtb_curvature(curve)['cvr'] >= 0

    def test_frtb_v_up_above_v_down_for_call(self, call_option, curve):
        cvr = call_option.frtb_curvature(curve)
        assert cvr['V_up'] > cvr['V_down']

    # ── cash flows ────────────────────────────────────────────────────────────

    def test_cash_flows_columns(self, call_option):
        cf = call_option.cash_flows()
        assert all(c in cf.columns for c in ['date', 'amount', 'type'])

    def test_cash_flows_single_row(self, call_option):
        assert len(call_option.cash_flows()) == 1

    def test_atm_intrinsic_zero(self, call_option):
        # ATM option -- intrinsic is zero
        assert call_option.cash_flows()['amount'].iloc[0] == 0.0