# tests/test_models.py

"""
Tests for src/quant_risk/models/base.py, models/rates.py, and models/equity.py.

Coverage
--------
StochasticProcess (base)
  - _draw_normals: shape, antithetic symmetry, seed reproducibility, odd-path error

VasicekProcess
  - output shape and initial condition
  - antithetic symmetry (paths[i] and paths[n//2+i] use opposite noise at step 0)
  - seed reproducibility
  - long-run mean reversion (E[r(∞)] → θ)
  - negative kappa / sigma validation

HullWhiteProcess
  - output shape and initial condition
  - antithetic symmetry
  - seed reproducibility
  - calibration: with σ → 0 the expected short rate path tracks f(0,t)
  - negative kappa / sigma validation
"""

import numpy as np
import pandas as pd
import pytest

from quant_risk.curves.ois import OISCurve
from quant_risk.models.rates import CIRProcess, HullWhiteProcess, VasicekProcess
from conftest import _flat_ois


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_curve() -> OISCurve:
    """Flat 2.5% OIS curve for Hull-White calibration tests."""
    return _flat_ois(0.025)


@pytest.fixture
def vasicek() -> VasicekProcess:
    """Standard Vasicek instance for reuse across tests."""
    return VasicekProcess(kappa=0.1, theta=0.025, sigma=0.005)


@pytest.fixture
def hw(flat_curve) -> HullWhiteProcess:
    """Hull-White calibrated to the flat 2.5% curve."""
    return HullWhiteProcess(curve=flat_curve, kappa=0.1, sigma=0.005)


# ---------------------------------------------------------------------------
# StochasticProcess._draw_normals (tested via VasicekProcess)
# ---------------------------------------------------------------------------

class TestDrawNormals:
    def test_shape_plain(self, vasicek):
        Z = vasicek._draw_normals(100, 50, antithetic=False, seed=0)
        assert Z.shape == (100, 50)

    def test_shape_antithetic(self, vasicek):
        Z = vasicek._draw_normals(100, 50, antithetic=True, seed=0)
        assert Z.shape == (100, 50)

    def test_antithetic_symmetry(self, vasicek):
        # The bottom half must be the exact negation of the top half
        Z = vasicek._draw_normals(100, 50, antithetic=True, seed=42)
        np.testing.assert_array_equal(Z[:50], -Z[50:])

    def test_seed_reproducibility(self, vasicek):
        Z1 = vasicek._draw_normals(100, 50, antithetic=False, seed=7)
        Z2 = vasicek._draw_normals(100, 50, antithetic=False, seed=7)
        np.testing.assert_array_equal(Z1, Z2)

    def test_different_seeds_differ(self, vasicek):
        Z1 = vasicek._draw_normals(100, 50, antithetic=False, seed=1)
        Z2 = vasicek._draw_normals(100, 50, antithetic=False, seed=2)
        assert not np.array_equal(Z1, Z2)

    def test_odd_paths_antithetic_raises(self, vasicek):
        with pytest.raises(ValueError, match="even"):
            vasicek._draw_normals(101, 50, antithetic=True, seed=0)


# ---------------------------------------------------------------------------
# VasicekProcess
# ---------------------------------------------------------------------------

class TestVasicekProcess:
    def test_output_shape(self, vasicek):
        paths = vasicek.simulate(x0=0.025, T=1.0, n_steps=252, n_paths=100, seed=0)
        assert paths.shape == (100, 253)

    def test_initial_condition(self, vasicek):
        paths = vasicek.simulate(x0=0.025, T=1.0, n_steps=252, n_paths=200, seed=0)
        # Every path starts at x0
        assert np.all(paths[:, 0] == 0.025)

    def test_seed_reproducibility(self, vasicek):
        p1 = vasicek.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=99)
        p2 = vasicek.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, vasicek):
        p1 = vasicek.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=1)
        p2 = vasicek.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=2)
        assert not np.array_equal(p1, p2)

    def test_antithetic_first_step_sum(self, vasicek):
        # At step 1, paths[i,1] + paths[n//2+i,1] = 2 * deterministic_drift
        # because the stochastic terms cancel: +Z and -Z.
        # deterministic_drift = x0 * exp(-κdt) + θ(1 - exp(-κdt))
        T, n_steps, n_paths = 1.0, 1, 100
        paths = vasicek.simulate(
            x0=0.025, T=T, n_steps=n_steps, n_paths=n_paths,
            antithetic=True, seed=0
        )
        dt = T / n_steps
        exp_kdt = np.exp(-vasicek.kappa * dt)
        deterministic = 0.025 * exp_kdt + vasicek.theta * (1.0 - exp_kdt)
        # sum of antithetic pair at step 1 == 2 * deterministic part
        pair_sums = paths[:50, 1] + paths[50:, 1]
        np.testing.assert_allclose(pair_sums, 2.0 * deterministic, rtol=1e-10)

    def test_long_run_mean_reversion(self):
        # For large T, E[r(T)] → θ regardless of r0
        proc = VasicekProcess(kappa=1.0, theta=0.03, sigma=0.003)
        paths = proc.simulate(x0=0.005, T=20.0, n_steps=1000, n_paths=5000, seed=0)
        mean_terminal = paths[:, -1].mean()
        # With κ=1.0 and T=20, exp(-κT) ≈ 2e-9, so E[r(T)] ≈ θ=0.03
        assert abs(mean_terminal - 0.03) < 0.0005

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError):
            VasicekProcess(kappa=-0.1, theta=0.025, sigma=0.005)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            VasicekProcess(kappa=0.1, theta=0.025, sigma=-0.005)

    def test_describe_contains_name(self, vasicek):
        assert "Vasicek" in vasicek.describe()


# ---------------------------------------------------------------------------
# HullWhiteProcess
# ---------------------------------------------------------------------------

class TestHullWhiteProcess:
    def test_output_shape(self, hw):
        paths = hw.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=100, seed=0)
        assert paths.shape == (100, 53)

    def test_initial_condition(self, hw):
        paths = hw.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=100, seed=0)
        assert np.all(paths[:, 0] == 0.025)

    def test_seed_reproducibility(self, hw):
        p1 = hw.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=7)
        p2 = hw.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_antithetic_shape(self, hw):
        paths = hw.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=100,
                            antithetic=True, seed=0)
        assert paths.shape == (100, 53)

    def test_antithetic_first_step_sum(self, hw):
        # Same logic as Vasicek: at step 1, noise cancels, leaving 2 * drift
        T, n_steps, n_paths = 1.0, 1, 100
        paths = hw.simulate(
            x0=0.025, T=T, n_steps=n_steps, n_paths=n_paths,
            antithetic=True, seed=0,
        )
        dt = T / n_steps
        theta_0 = hw._theta(0.0)
        # Euler drift: r0 + (θ(0) - κ r0) dt
        deterministic = 0.025 + (theta_0 - hw.kappa * 0.025) * dt
        pair_sums = paths[:50, 1] + paths[50:, 1]
        np.testing.assert_allclose(pair_sums, 2.0 * deterministic, rtol=1e-10)

    def test_calibration_flat_curve(self, hw):
        # With a flat curve at 2.5% and very small sigma, E[r(t)] ≈ 2.5% at all t.
        # Use low sigma to suppress stochastic noise.
        proc = HullWhiteProcess(
            curve=_flat_ois(0.025),
            kappa=0.1,
            sigma=0.0001,   # near-deterministic
        )
        paths = proc.simulate(x0=0.025, T=5.0, n_steps=60, n_paths=2000, seed=0)
        # Check mean at a few time points against the flat forward rate (0.025 = 2.5%)
        for col in [12, 24, 60]:   # 1Y, 2Y, 5Y
            mean_r = paths[:, col].mean()
            assert abs(mean_r - 0.025) < 0.001, (
                f"E[r] at column {col} = {mean_r:.6f}, expected ≈ 0.025"
            )

    def test_invalid_kappa_raises(self, flat_curve):
        with pytest.raises(ValueError):
            HullWhiteProcess(curve=flat_curve, kappa=0.0, sigma=0.005)

    def test_invalid_sigma_raises(self, flat_curve):
        with pytest.raises(ValueError):
            HullWhiteProcess(curve=flat_curve, kappa=0.1, sigma=0.0)  # sigma=0 invalid

    def test_describe_contains_name(self, hw):
        assert "Hull-White" in hw.describe()


# ---------------------------------------------------------------------------
# GBMProcess
# ---------------------------------------------------------------------------

from quant_risk.models.equity import GBMProcess, LocalVolProcess


class TestGBMProcess:
    @pytest.fixture
    def gbm(self):
        return GBMProcess(r=0.025, sigma=0.20)

    def test_output_shape(self, gbm):
        paths = gbm.simulate(x0=100.0, T=1.0, n_steps=252, n_paths=100, seed=0)
        assert paths.shape == (100, 253)

    def test_initial_condition(self, gbm):
        paths = gbm.simulate(x0=100.0, T=1.0, n_steps=252, n_paths=100, seed=0)
        np.testing.assert_allclose(paths[:, 0], 100.0)

    def test_positive_prices(self, gbm):
        # GBM exact simulation preserves S > 0 for all paths
        paths = gbm.simulate(x0=100.0, T=5.0, n_steps=250, n_paths=500, seed=0)
        assert (paths > 0).all()

    def test_risk_neutral_expectation(self, gbm):
        # E[S(T)] = S0 * exp((r-q)/100 * T)
        T     = 1.0
        S0    = 100.0
        paths = gbm.simulate(x0=S0, T=T, n_steps=252, n_paths=10000,
                              antithetic=True, seed=0)
        expected = S0 * np.exp((gbm.r - gbm.q) * T)
        # Allow 0.5% tolerance
        assert abs(paths[:, -1].mean() - expected) < 0.5 * expected / 100

    def test_seed_reproducibility(self, gbm):
        p1 = gbm.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=50, seed=7)
        p2 = gbm.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=50, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_antithetic_shape(self, gbm):
        paths = gbm.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=100,
                             antithetic=True, seed=0)
        assert paths.shape == (100, 53)

    def test_antithetic_log_symmetry(self, gbm):
        # Antithetic pairs have log-returns that sum to 2 × deterministic drift
        n_paths, n_steps = 100, 1
        paths = gbm.simulate(x0=100.0, T=1.0, n_steps=n_steps, n_paths=n_paths,
                             antithetic=True, seed=0)
        dt         = 1.0
        drift_step = ((gbm.r - gbm.q) - gbm.sigma ** 2 / 2) * dt
        log_ret_base = np.log(paths[:50, 1] / paths[:50, 0])
        log_ret_anti = np.log(paths[50:, 1] / paths[50:, 0])
        # Sum of log-returns of antithetic pair = 2 × drift (noise cancels)
        np.testing.assert_allclose(
            log_ret_base + log_ret_anti, 2 * drift_step, rtol=1e-10
        )

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            GBMProcess(r=0.025, sigma=0.0)

    def test_describe_contains_name(self, gbm):
        assert "GBM" in gbm.describe()


# ---------------------------------------------------------------------------
# LocalVolProcess
# ---------------------------------------------------------------------------

class TestLocalVolProcess:
    @pytest.fixture
    def flat_lv(self):
        # Constant local vol of 20% → should behave like GBM
        return LocalVolProcess(r=0.025, local_vol_fn=lambda S, t: 0.20)

    def test_output_shape(self, flat_lv):
        paths = flat_lv.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=100, seed=0)
        assert paths.shape == (100, 53)

    def test_initial_condition(self, flat_lv):
        paths = flat_lv.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=100, seed=0)
        np.testing.assert_allclose(paths[:, 0], 100.0)

    def test_positive_prices(self, flat_lv):
        paths = flat_lv.simulate(x0=100.0, T=5.0, n_steps=60, n_paths=200, seed=0)
        assert (paths > 0).all()

    def test_flat_lv_matches_gbm_distribution(self):
        # With constant σ, LocalVol terminal distribution should match GBM closely
        r, sigma, S0, T = 0.025, 0.20, 100.0, 1.0
        gbm  = GBMProcess(r=r, sigma=sigma)
        lv   = LocalVolProcess(r=r, local_vol_fn=lambda S, t: sigma)
        n    = 5000

        p_gbm = gbm.simulate(x0=S0, T=T, n_steps=52, n_paths=n, seed=0)
        p_lv  = lv.simulate(x0=S0,  T=T, n_steps=52, n_paths=n, seed=0)

        # Means should be close (within 1%)
        assert abs(p_gbm[:, -1].mean() - p_lv[:, -1].mean()) < 1.0

    def test_seed_reproducibility(self, flat_lv):
        p1 = flat_lv.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=50, seed=3)
        p2 = flat_lv.simulate(x0=100.0, T=1.0, n_steps=52, n_paths=50, seed=3)
        np.testing.assert_array_equal(p1, p2)

    def test_vol_surface_affects_distribution(self):
        # Skewed vol shifts the distribution relative to flat vol
        r, S0, T = 0.025, 100.0, 1.0
        flat_lv  = LocalVolProcess(r=r, local_vol_fn=lambda S, t: 0.20)
        # Higher vol for low S (skew): ITM calls more expensive
        skew_lv  = LocalVolProcess(
            r=r, local_vol_fn=lambda S, t: 0.20 + 0.05 * max(100.0 - S, 0) / 100.0
        )
        n = 3000
        p_flat = flat_lv.simulate(x0=S0, T=T, n_steps=52, n_paths=n, seed=0)
        p_skew = skew_lv.simulate(x0=S0, T=T, n_steps=52, n_paths=n, seed=0)
        # Skewed vol makes distribution different (higher std due to higher avg vol)
        assert p_skew[:, -1].std() != p_flat[:, -1].std()

    def test_describe_contains_name(self, flat_lv):
        assert "LocalVol" in flat_lv.describe()


# ---------------------------------------------------------------------------
# MCSimulator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CIRProcess
# ---------------------------------------------------------------------------


class TestCIRProcess:
    @pytest.fixture
    def cir(self):
        # κ=0.3, θ=0.03, σ=0.06: Feller = 2·0.3·0.03 = 0.018 > 0.06² = 0.0036 ✓
        return CIRProcess(kappa=0.3, theta=0.03, sigma=0.06)

    def test_output_shape(self, cir):
        paths = cir.simulate(x0=0.025, T=1.0, n_steps=252, n_paths=100, seed=0)
        assert paths.shape == (100, 253)

    def test_initial_condition(self, cir):
        paths = cir.simulate(x0=0.025, T=1.0, n_steps=252, n_paths=200, seed=0)
        np.testing.assert_allclose(paths[:, 0], 0.025)

    def test_positivity(self, cir):
        # Exact chi-squared simulation: rates are structurally non-negative
        paths = cir.simulate(x0=0.025, T=5.0, n_steps=250, n_paths=500, seed=0)
        assert (paths >= 0).all()

    def test_seed_reproducibility(self, cir):
        p1 = cir.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=7)
        p2 = cir.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, cir):
        p1 = cir.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=1)
        p2 = cir.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=50, seed=2)
        assert not np.array_equal(p1, p2)

    def test_long_run_mean_reversion(self):
        # E[r(T)] → θ as T → ∞ for large κ (same formula as Vasicek)
        proc = CIRProcess(kappa=1.0, theta=0.03, sigma=0.05)
        paths = proc.simulate(x0=0.005, T=20.0, n_steps=1000, n_paths=5000, seed=0)
        assert abs(paths[:, -1].mean() - 0.03) < 0.0005

    def test_antithetic_raises(self, cir):
        with pytest.raises(NotImplementedError, match="antithetic"):
            cir.simulate(x0=0.025, T=1.0, n_steps=52, n_paths=100, antithetic=True)

    def test_negative_x0_raises(self, cir):
        with pytest.raises(ValueError, match="x0"):
            cir.simulate(x0=-0.1, T=1.0, n_steps=52, n_paths=100)

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError):
            CIRProcess(kappa=0.0, theta=0.025, sigma=0.03)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            CIRProcess(kappa=0.1, theta=0.025, sigma=0.0)

    def test_feller_violation_warns(self):
        # κ=0.1, θ=1.0, σ=1.0: 2·0.1·1.0 = 0.2 ≤ 1.0² = 1.0
        with pytest.warns(UserWarning, match="Feller"):
            CIRProcess(kappa=0.1, theta=0.01, sigma=0.1)

    def test_describe_contains_name(self, cir):
        assert "CIR" in cir.describe()

    def test_describe_feller_met(self, cir):
        assert "met" in cir.describe()

    def test_describe_feller_violated(self):
        with pytest.warns(UserWarning):
            proc = CIRProcess(kappa=0.1, theta=1.0, sigma=1.0)
        assert "violated" in proc.describe()


from quant_risk.models.simulator import MCSimulator


class TestMCSimulator:
    @pytest.fixture
    def sim(self, vasicek):
        return MCSimulator(
            process=vasicek,
            x0=0.025,
            T=5.0,
            n_steps=60,
            n_paths=1000,
            antithetic=True,
            seed=42,
        )

    # -- construction / properties ------------------------------------------

    def test_paths_shape(self, sim):
        assert sim.paths.shape == (1000, 61)

    def test_dt(self, sim):
        assert sim.dt == pytest.approx(5.0 / 60)

    def test_T_property(self, sim):
        assert sim.T == 5.0

    def test_n_paths_property(self, sim):
        assert sim.n_paths == 1000

    def test_n_steps_property(self, sim):
        assert sim.n_steps == 60

    # -- sdf ----------------------------------------------------------------

    def test_sdf_shape(self, sim):
        assert sim.sdf(1.0).shape == (1000,)

    def test_sdf_positive(self, sim):
        # D(0,t) = exp(-∫r ds) > 0 for all paths
        assert (sim.sdf(2.5) > 0).all()

    def test_sdf_deterministic(self):
        # Near-zero sigma collapses variance: D(0,T) ≈ exp(-r * T)
        r, T = 0.03, 2.0
        proc = VasicekProcess(kappa=1.0, theta=r, sigma=0.00001)
        sim = MCSimulator(proc, x0=r, T=T, n_steps=200, n_paths=500, seed=0)
        expected = np.exp(-r * T)
        np.testing.assert_allclose(sim.sdf(T).mean(), expected, rtol=0.01)

    # -- sdf_between --------------------------------------------------------

    def test_sdf_between_shape(self, sim):
        assert sim.sdf_between(1.0, 3.0).shape == (1000,)

    def test_sdf_between_multiplicative(self, sim):
        # D(0,T) = D(0,t1) * D(t1,T) — exact equality via cumulative integral
        t1, T = 2.0, 5.0
        np.testing.assert_allclose(
            sim.sdf(t1) * sim.sdf_between(t1, T),
            sim.sdf(T),
            rtol=1e-10,
        )

    # -- price --------------------------------------------------------------

    def test_price_zero_payoff(self, sim):
        assert sim.price(lambda paths: np.zeros(paths.shape[0])) == 0.0

    def test_price_zcb_approximation(self):
        # E^Q[D(0,T) * 1] = P(0,T) ≈ exp(-r * T) under flat r
        r, T = 0.025, 3.0
        proc = VasicekProcess(kappa=1.0, theta=r, sigma=0.00001)
        sim = MCSimulator(proc, x0=r, T=T, n_steps=180, n_paths=500, seed=0)
        price = sim.price(lambda paths: np.ones(paths.shape[0]))
        np.testing.assert_allclose(price, np.exp(-r * T), rtol=0.01)

    # -- exposure_profile ---------------------------------------------------

    def test_exposure_profile_keys(self, sim):
        dates = np.array([1.0, 2.0, 3.0])
        result = sim.exposure_profile(
            lambda paths, t: np.zeros(paths.shape[0]), dates
        )
        assert set(result.keys()) == {"dates", "mtm", "EE", "NEE", "PFE", "EE_disc", "EPE"}

    def test_exposure_profile_shapes(self, sim):
        dates = np.array([1.0, 2.0, 3.0])
        result = sim.exposure_profile(
            lambda paths, t: np.zeros(paths.shape[0]), dates
        )
        assert result["mtm"].shape     == (1000, 3)
        assert result["EE"].shape      == (3,)
        assert result["NEE"].shape     == (3,)
        assert result["PFE"].shape     == (3,)
        assert result["EE_disc"].shape == (3,)
        assert isinstance(result["EPE"], float)

    def test_exposure_profile_ee_nonnegative(self, sim):
        # EE(t) = E[max(V(t),0)] ≥ 0 by definition
        dates = np.arange(1.0, 6.0, 1.0)
        result = sim.exposure_profile(
            lambda paths, t: paths[:, -1] - 2.5, dates
        )
        assert (result["EE"] >= 0).all()

    def test_exposure_profile_nee_nonpositive(self, sim):
        # NEE(t) = E[min(V(t),0)] ≤ 0 by definition
        dates = np.arange(1.0, 6.0, 1.0)
        result = sim.exposure_profile(
            lambda paths, t: paths[:, -1] - 2.5, dates
        )
        assert (result["NEE"] <= 0).all()

    def test_exposure_profile_zero_mtm(self, sim):
        dates = np.array([1.0, 2.0, 3.0])
        result = sim.exposure_profile(
            lambda paths, t: np.zeros(paths.shape[0]), dates
        )
        np.testing.assert_array_equal(result["EE"], 0.0)
        np.testing.assert_array_equal(result["NEE"], 0.0)
        assert result["EPE"] == 0.0

    def test_exposure_profile_positive_mtm(self, sim):
        # Always-positive MTM: EE = const, NEE = 0
        dates = np.array([1.0, 2.0])
        const_val = 5.0
        result = sim.exposure_profile(
            lambda paths, t: np.full(paths.shape[0], const_val), dates
        )
        np.testing.assert_allclose(result["EE"], const_val)
        np.testing.assert_allclose(result["NEE"], 0.0)

    def test_exposure_profile_ee_disc_lt_ee(self, sim):
        # EE_disc = E[D(0,t) * max(V,0)] < EE when positive rates → D(0,t) < 1
        dates = np.array([2.0, 4.0])
        result = sim.exposure_profile(
            lambda paths, t: np.ones(paths.shape[0]), dates
        )
        assert (result["EE_disc"] < result["EE"]).all()

    def test_exposure_profile_epe_is_ee_mean(self, sim):
        # EPE is defined as the time-average of EE
        dates = np.array([1.0, 2.0, 3.0])
        result = sim.exposure_profile(
            lambda paths, t: np.ones(paths.shape[0]), dates
        )
        assert result["EPE"] == pytest.approx(result["EE"].mean())

    # -- describe -----------------------------------------------------------

    def test_describe(self, sim):
        desc = sim.describe()
        assert "MCSimulator" in desc
        assert "Vasicek" in desc
