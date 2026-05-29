# tests/test_models.py

"""
Tests for src/quant_risk/models/base.py and models/rates.py.

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
from quant_risk.models.rates import HullWhiteProcess, VasicekProcess


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _synthetic_ois_data() -> pd.DataFrame:
    """Flat 2.5% OIS curve across all tenors, synthetic, no API calls."""
    data = pd.DataFrame(
        {
            "years":           [1/12, 2/12, 3/12, 6/12, 9/12,  1.0,  2.0,  3.0,  5.0, 10.0, 15.0],
            "zero_rate_pct":   [2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50, 2.50,  2.50,  2.50],
            "discount_factor": [
                np.exp(-0.025 * t)
                for t in [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
            ],
            "valuation_date": ["2026-03-24"] * 11,
        },
        index=["1M", "2M", "3M", "6M", "9M", "12M", "2Y", "3Y", "5Y", "10Y", "10Y+"],
    )
    data.index.name = "maturity"
    return data


@pytest.fixture
def flat_curve() -> OISCurve:
    """Flat 2.5% OIS curve for Hull-White calibration tests."""
    return OISCurve(_synthetic_ois_data())


@pytest.fixture
def vasicek() -> VasicekProcess:
    """Standard Vasicek instance for reuse across tests."""
    return VasicekProcess(kappa=0.1, theta=2.5, sigma=0.5)


@pytest.fixture
def hw(flat_curve) -> HullWhiteProcess:
    """Hull-White calibrated to the flat 2.5% curve."""
    return HullWhiteProcess(curve=flat_curve, kappa=0.1, sigma=0.5)


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
        paths = vasicek.simulate(x0=2.5, T=1.0, n_steps=252, n_paths=100, seed=0)
        assert paths.shape == (100, 253)

    def test_initial_condition(self, vasicek):
        paths = vasicek.simulate(x0=2.5, T=1.0, n_steps=252, n_paths=200, seed=0)
        # Every path starts at x0
        assert np.all(paths[:, 0] == 2.5)

    def test_seed_reproducibility(self, vasicek):
        p1 = vasicek.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=50, seed=99)
        p2 = vasicek.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=50, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self, vasicek):
        p1 = vasicek.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=50, seed=1)
        p2 = vasicek.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=50, seed=2)
        assert not np.array_equal(p1, p2)

    def test_antithetic_first_step_sum(self, vasicek):
        # At step 1, paths[i,1] + paths[n//2+i,1] = 2 * deterministic_drift
        # because the stochastic terms cancel: +Z and -Z.
        # deterministic_drift = x0 * exp(-κdt) + θ(1 - exp(-κdt))
        T, n_steps, n_paths = 1.0, 1, 100
        paths = vasicek.simulate(
            x0=2.5, T=T, n_steps=n_steps, n_paths=n_paths,
            antithetic=True, seed=0
        )
        dt = T / n_steps
        exp_kdt = np.exp(-vasicek.kappa * dt)
        deterministic = 2.5 * exp_kdt + vasicek.theta * (1.0 - exp_kdt)
        # sum of antithetic pair at step 1 == 2 * deterministic part
        pair_sums = paths[:50, 1] + paths[50:, 1]
        np.testing.assert_allclose(pair_sums, 2.0 * deterministic, rtol=1e-10)

    def test_long_run_mean_reversion(self):
        # For large T, E[r(T)] → θ regardless of r0
        proc = VasicekProcess(kappa=1.0, theta=3.0, sigma=0.3)
        paths = proc.simulate(x0=0.5, T=20.0, n_steps=1000, n_paths=5000, seed=0)
        mean_terminal = paths[:, -1].mean()
        # With κ=1.0 and T=20, exp(-κT) ≈ 2e-9, so E[r(T)] ≈ θ=3.0
        assert abs(mean_terminal - 3.0) < 0.05

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError):
            VasicekProcess(kappa=-0.1, theta=2.5, sigma=0.5)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError):
            VasicekProcess(kappa=0.1, theta=2.5, sigma=-0.5)

    def test_describe_contains_name(self, vasicek):
        assert "Vasicek" in vasicek.describe()


# ---------------------------------------------------------------------------
# HullWhiteProcess
# ---------------------------------------------------------------------------

class TestHullWhiteProcess:
    def test_output_shape(self, hw):
        paths = hw.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=100, seed=0)
        assert paths.shape == (100, 53)

    def test_initial_condition(self, hw):
        paths = hw.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=100, seed=0)
        assert np.all(paths[:, 0] == 2.5)

    def test_seed_reproducibility(self, hw):
        p1 = hw.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=50, seed=7)
        p2 = hw.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=50, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_antithetic_shape(self, hw):
        paths = hw.simulate(x0=2.5, T=1.0, n_steps=52, n_paths=100,
                            antithetic=True, seed=0)
        assert paths.shape == (100, 53)

    def test_antithetic_first_step_sum(self, hw):
        # Same logic as Vasicek: at step 1, noise cancels, leaving 2 * drift
        T, n_steps, n_paths = 1.0, 1, 100
        paths = hw.simulate(
            x0=2.5, T=T, n_steps=n_steps, n_paths=n_paths,
            antithetic=True, seed=0,
        )
        dt = T / n_steps
        theta_0 = hw._theta(0.0)
        # Euler drift: r0 + (θ(0) - κ r0) dt
        deterministic = 2.5 + (theta_0 - hw.kappa * 2.5) * dt
        pair_sums = paths[:50, 1] + paths[50:, 1]
        np.testing.assert_allclose(pair_sums, 2.0 * deterministic, rtol=1e-10)

    def test_calibration_flat_curve(self, hw):
        # With a flat curve at 2.5% and very small sigma, E[r(t)] ≈ 2.5% at all t.
        # Use low sigma to suppress stochastic noise.
        proc = HullWhiteProcess(
            curve=OISCurve(_synthetic_ois_data()),
            kappa=0.1,
            sigma=0.01,   # near-deterministic
        )
        paths = proc.simulate(x0=2.5, T=5.0, n_steps=60, n_paths=2000, seed=0)
        # Check mean at a few time points against the flat forward rate (2.5%)
        for col in [12, 24, 60]:   # 1Y, 2Y, 5Y
            mean_r = paths[:, col].mean()
            assert abs(mean_r - 2.5) < 0.10, (
                f"E[r] at column {col} = {mean_r:.4f}%, expected ≈ 2.5%"
            )

    def test_invalid_kappa_raises(self, flat_curve):
        with pytest.raises(ValueError):
            HullWhiteProcess(curve=flat_curve, kappa=0.0, sigma=0.5)

    def test_invalid_sigma_raises(self, flat_curve):
        with pytest.raises(ValueError):
            HullWhiteProcess(curve=flat_curve, kappa=0.1, sigma=0.0)

    def test_describe_contains_name(self, hw):
        assert "Hull-White" in hw.describe()
