# tests/test_xva.py

"""
Tests for src/quant_risk/risk/xva.py — XVAEngine and Trade.

Coverage
--------
Trade
  - Construction with valid parameters

XVAEngine
  - Construction and describe()
  - exposure_profile(): shapes, EE >= 0, NEE <= 0, caching (same object returned)
  - cva(): positive, monotone in spread, monotone in recovery
  - dva(): non-negative
  - fva(): two_way CSA → FVA=0; one_way_post → FBA=0; none → FCA and FBA both nonzero
  - mva(): non-negative; √MPOR scaling
  - xva(): all required keys present; total_XVA = CVA - DVA + FVA + MVA
  - cs01(): positive for payer IRS
  - dva01(): non-negative
  - Netting: two exactly offsetting trades reduce CVA below sum of individual CVAs
"""

import numpy as np
import pandas as pd
import pytest

from quant_risk.curves.ois import OISCurve
from quant_risk.models.rates import HullWhiteProcess
from quant_risk.models.simulator import MCSimulator
from quant_risk.risk.xva import Trade, XVAEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _synthetic_flat_ois(rate: float = 0.025) -> OISCurve:
    """Flat OIS curve at rate (decimal), synthetic, no API calls."""
    tenors  = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
    labels  = ["1M","2M","3M","6M","9M","12M","2Y","3Y","5Y","10Y","10Y+"]
    data = pd.DataFrame(
        {
            "years":           tenors,
            "zero_rate_pct":   [rate] * len(tenors),
            "discount_factor": [np.exp(-rate * t) for t in tenors],
            "valuation_date":  ["2026-03-24"] * len(tenors),
        },
        index=labels,
    )
    data.index.name = "maturity"
    return OISCurve(data)


KAPPA = 0.10
SIGMA = 0.005   # 0.5%/√yr in decimal
RATE  = 0.025   # 2.5% in decimal
# Small sim — speed over precision in unit tests
N_PATHS = 200
N_STEPS = 24   # bi-monthly over 2Y
T_SIM   = 5.0
EXP_DATES = np.arange(0.5, 3.0, 0.5)   # 0.5Y, 1.0Y, 1.5Y, 2.0Y, 2.5Y
MPOR_BASE_DAYS = 10


@pytest.fixture(scope="module")
def flat_ois() -> OISCurve:
    return _synthetic_flat_ois(RATE)


@pytest.fixture(scope="module")
def sim(flat_ois) -> MCSimulator:
    return MCSimulator(
        process   = HullWhiteProcess(curve=flat_ois, kappa=KAPPA, sigma=SIGMA),
        x0        = RATE,
        T         = T_SIM,
        n_steps   = N_STEPS,
        n_paths   = N_PATHS,
        antithetic= True,
        seed      = 0,
    )


@pytest.fixture(scope="module")
def payer_trade() -> Trade:
    return Trade(
        payment_dates = np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        year_fracs    = np.ones(5),
        K             = RATE,
        notional      = 1_000_000,
        is_payer      = True,
    )


@pytest.fixture(scope="module")
def engine(sim, flat_ois, payer_trade) -> XVAEngine:
    return XVAEngine(
        simulator = sim,
        trades    = [payer_trade],
        exp_dates = EXP_DATES,
        kappa     = KAPPA,
        sigma     = SIGMA,
        curve     = flat_ois,
    )


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

class TestTrade:
    def test_construction(self):
        tr = Trade(
            payment_dates = np.array([1.0, 2.0]),
            year_fracs    = np.ones(2),
            K             = 0.025,
            notional      = 500_000,
            is_payer      = False,
        )
        assert tr.K == 0.025
        assert tr.notional == 500_000
        assert not tr.is_payer


# ---------------------------------------------------------------------------
# XVAEngine — construction and describe
# ---------------------------------------------------------------------------

class TestXVAEngineBasics:
    def test_describe_contains_engine(self, engine):
        assert "XVAEngine" in engine.describe()

    def test_describe_contains_trade_count(self, engine):
        assert "1 trade" in engine.describe()

    def test_describe_contains_kappa(self, engine):
        assert f"{KAPPA:.4f}" in engine.describe()


# ---------------------------------------------------------------------------
# Exposure profile
# ---------------------------------------------------------------------------

class TestExposureProfile:
    def test_profile_keys(self, engine):
        prof = engine.exposure_profile()
        assert {'dates', 'mtm', 'EE', 'NEE', 'PFE', 'EE_disc', 'EPE'}.issubset(prof)

    def test_ee_shape(self, engine):
        prof = engine.exposure_profile()
        assert prof['EE'].shape == (len(EXP_DATES),)

    def test_mtm_shape(self, engine):
        prof = engine.exposure_profile()
        assert prof['mtm'].shape == (N_PATHS, len(EXP_DATES))

    def test_ee_nonnegative(self, engine):
        # EE = E[max(V,0)] >= 0 by definition
        assert (engine.exposure_profile()['EE'] >= 0).all()

    def test_nee_nonpositive(self, engine):
        # NEE = E[min(V,0)] <= 0 by definition
        assert (engine.exposure_profile()['NEE'] <= 0).all()

    def test_profile_cached(self, engine):
        # Same Python object must be returned on repeated calls
        prof1 = engine.exposure_profile()
        prof2 = engine.exposure_profile()
        assert prof1 is prof2


# ---------------------------------------------------------------------------
# CVA
# ---------------------------------------------------------------------------

class TestCVA:
    def test_cva_positive(self, engine):
        assert engine.cva(60.0, 0.40)['CVA'] > 0

    def test_cva_keys(self, engine):
        result = engine.cva(60.0, 0.40)
        assert {'CVA', 'h', 'pd_period', 'cva_by_period'}.issubset(result)

    def test_cva_increases_with_spread(self, engine):
        cva_lo = engine.cva(30.0, 0.40)['CVA']
        cva_hi = engine.cva(120.0, 0.40)['CVA']
        assert cva_hi > cva_lo

    def test_cva_decreases_with_recovery(self, engine):
        cva_lo_r = engine.cva(60.0, 0.20)['CVA']
        cva_hi_r = engine.cva(60.0, 0.60)['CVA']
        assert cva_lo_r > cva_hi_r

    def test_cva_zero_spread_is_zero(self, engine):
        # h = 0 → PD = 0 → CVA = 0
        assert engine.cva(0.0, 0.40)['CVA'] == pytest.approx(0.0, abs=1e-10)

    def test_cva_by_period_sums_to_cva(self, engine):
        result = engine.cva(60.0, 0.40)
        np.testing.assert_allclose(
            result['cva_by_period'].sum(), result['CVA'], rtol=1e-10
        )


# ---------------------------------------------------------------------------
# DVA
# ---------------------------------------------------------------------------

class TestDVA:
    def test_dva_nonnegative(self, engine):
        assert engine.dva(45.0, 0.40)['DVA'] >= 0

    def test_dva_keys(self, engine):
        result = engine.dva(45.0, 0.40)
        assert {'DVA', 'h_own', 'pd_own', 'dva_by_period'}.issubset(result)

    def test_dva_zero_spread_is_zero(self, engine):
        assert engine.dva(0.0, 0.40)['DVA'] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# FVA
# ---------------------------------------------------------------------------

class TestFVA:
    def test_fva_two_way_is_zero(self, engine):
        result = engine.fva(30.0, 20.0, 'two_way')
        assert result['FVA'] == 0.0
        assert result['FCA'] == 0.0
        assert result['FBA'] == 0.0

    def test_fva_one_way_post_fba_is_zero(self, engine):
        result = engine.fva(30.0, 20.0, 'one_way_post')
        assert result['FBA'] == 0.0
        np.testing.assert_array_equal(result['fba_by_period'], 0.0)

    def test_fva_one_way_post_fca_positive(self, engine):
        assert engine.fva(30.0, 20.0, 'one_way_post')['FCA'] > 0

    def test_fva_none_fca_and_fba_nonneg(self, engine):
        result = engine.fva(30.0, 20.0, 'none')
        assert result['FCA'] >= 0
        assert result['FBA'] >= 0

    def test_fva_keys(self, engine):
        result = engine.fva(30.0, 20.0, 'none')
        assert {'FVA', 'FCA', 'FBA', 'fca_by_period', 'fba_by_period'}.issubset(result)

    def test_fva_equals_fca_minus_fba(self, engine):
        result = engine.fva(30.0, 20.0, 'none')
        np.testing.assert_allclose(
            result['FVA'], result['FCA'] - result['FBA'], rtol=1e-10
        )

    def test_fva_zero_spread_is_zero(self, engine):
        result = engine.fva(0.0, 0.0, 'none')
        assert result['FVA'] == pytest.approx(0.0, abs=1e-10)



# ---------------------------------------------------------------------------
# MVA
# ---------------------------------------------------------------------------

class TestMVA:
    def test_mva_nonnegative(self, engine):
        assert engine.mva(64.0, 10, 25.0, MPOR_BASE_DAYS)['MVA'] >= 0

    def test_mva_keys(self, engine):
        result = engine.mva(64.0, 10, 25.0, MPOR_BASE_DAYS)
        assert {'MVA', 'eim_disc', 'mva_by_period'}.issubset(result)

    def test_mva_eim_disc_nonneg(self, engine):
        # Expected initial margin is always non-negative (IM >= 0)
        assert (engine.mva(64.0, 10, 25.0, MPOR_BASE_DAYS)['eim_disc'] >= 0).all()

    def test_mva_scales_with_sqrt_mpor(self, engine):
        # MVA(MPOR=5) ≈ MVA(MPOR=10) / sqrt(2)
        mva_10 = engine.mva(64.0, 10, 25.0, MPOR_BASE_DAYS)['MVA']
        mva_5  = engine.mva(64.0,  5, 25.0, MPOR_BASE_DAYS)['MVA']
        np.testing.assert_allclose(mva_5, mva_10 / np.sqrt(2), rtol=0.01)

    def test_mva_zero_spread_is_zero(self, engine):
        assert engine.mva(64.0, 10, 0.0, MPOR_BASE_DAYS)['MVA'] == pytest.approx(0.0, abs=1e-10)

    def test_mva_zero_rw_is_zero(self, engine):
        assert engine.mva(0.0, 10, 25.0, MPOR_BASE_DAYS)['MVA'] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Full XVA report
# ---------------------------------------------------------------------------

class TestXVA:
    def test_xva_keys(self, engine):
        result = engine.xva(
            cds_spread_bps=60.0, recovery=0.40,
            own_cds_spread_bps=45.0, own_recovery=0.40,
            funding_spread_bps=30.0, invest_spread_bps=20.0, csa_type='none',
            risk_weight_bps=64.0, mpor_days=10, im_funding_spread_bps=25.0,
            mpor_base_days=MPOR_BASE_DAYS,
        )
        assert {'CVA','DVA','BCVA','FVA','FCA','FBA','MVA','total_XVA','details'}.issubset(result)

    def test_xva_total_equals_sum(self, engine):
        result = engine.xva(
            60.0, 0.40, 45.0, 0.40, 30.0, 20.0, 'none', 64.0, 10, 25.0, MPOR_BASE_DAYS,
        )
        expected = result['CVA'] - result['DVA'] + result['FVA'] + result['MVA']
        np.testing.assert_allclose(result['total_XVA'], expected, rtol=1e-10)

    def test_xva_bcva_equals_cva_minus_dva(self, engine):
        result = engine.xva(
            60.0, 0.40, 45.0, 0.40, 30.0, 20.0, 'none', 64.0, 10, 25.0, MPOR_BASE_DAYS,
        )
        np.testing.assert_allclose(
            result['BCVA'], result['CVA'] - result['DVA'], rtol=1e-10
        )

    def test_xva_two_way_csa_fva_zero(self, engine):
        result = engine.xva(
            60.0, 0.40, 45.0, 0.40, 30.0, 20.0, 'two_way', 64.0, 10, 25.0, MPOR_BASE_DAYS,
        )
        assert result['FVA'] == 0.0


# ---------------------------------------------------------------------------
# Sensitivities
# ---------------------------------------------------------------------------

class TestSensitivities:
    def test_cs01_positive_for_payer(self, engine):
        # CVA increases when counterparty CDS spread widens
        assert engine.cs01(60.0, 0.40) > 0

    def test_cs01_scales_with_bump(self, engine):
        # CS01 at 2 bps should be approximately 2× the 1 bp CS01
        cs01_1 = engine.cs01(60.0, 0.40, bump_bps=1.0)
        cs01_2 = engine.cs01(60.0, 0.40, bump_bps=2.0)
        np.testing.assert_allclose(cs01_2, 2.0 * cs01_1, rtol=0.05)

    def test_dva01_nonneg(self, engine):
        # DVA increases when own CDS spread widens
        assert engine.dva01(45.0, 0.40) >= 0


# ---------------------------------------------------------------------------
# Netting benefit
# ---------------------------------------------------------------------------

class TestNetting:
    def test_perfect_hedge_reduces_cva(self, sim, flat_ois):
        """A payer + matching receiver in the same netting set should give
        lower CVA than the payer alone, because net exposure is zero."""
        pay_dates = np.array([1.0, 2.0, 3.0])
        yf        = np.ones(3)

        tr_pay = Trade(pay_dates, yf, RATE, 500_000, True)
        tr_rec = Trade(pay_dates, yf, RATE, 500_000, False)

        exp = np.arange(0.5, 3.5, 0.5)
        kappa, sigma = 0.10, 0.005

        eng_pay  = XVAEngine(sim, [tr_pay],         exp, kappa, sigma, flat_ois)
        eng_net  = XVAEngine(sim, [tr_pay, tr_rec],  exp, kappa, sigma, flat_ois)

        cva_pay = eng_pay.cva(60.0, 0.40)['CVA']
        cva_net = eng_net.cva(60.0, 0.40)['CVA']

        # Net CVA should be close to zero (perfectly offsetting MTMs)
        assert cva_net < cva_pay

    def test_netting_reduces_mva(self, sim, flat_ois):
        """Offsetting DV01 signs in the same SIMM bucket reduce net IM."""
        pay_dates_5 = np.arange(1.0, 5.001, 1.0)
        pay_dates_3 = np.arange(1.0, 3.001, 1.0)
        yf_5 = np.ones(5); yf_3 = np.ones(3)

        tr_pay = Trade(pay_dates_5, yf_5, RATE, 1_000_000, True)
        tr_rec = Trade(pay_dates_3, yf_3, RATE, 1_000_000, False)

        exp   = np.arange(0.5, 3.5, 0.5)
        kappa, sigma = 0.10, 0.005

        eng_pay  = XVAEngine(sim, [tr_pay],         exp, kappa, sigma, flat_ois)
        eng_net  = XVAEngine(sim, [tr_pay, tr_rec],  exp, kappa, sigma, flat_ois)

        mva_pay = eng_pay.mva(64.0, 10, 25.0, MPOR_BASE_DAYS)['MVA']
        mva_net = eng_net.mva(64.0, 10, 25.0, MPOR_BASE_DAYS)['MVA']

        # Net MVA must be less than single-trade MVA (DV01 partially cancels)
        assert mva_net < mva_pay
