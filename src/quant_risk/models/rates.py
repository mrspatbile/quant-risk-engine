# src/quant_risk/models/rates.py

"""
Short rate stochastic processes for Monte Carlo interest rate simulation.

Two models are implemented:

Vasicek (1977)
    dr(t) = κ(θ - r(t)) dt + σ dW(t)

    Constant parameters. Rates can go negative. Exact simulation is
    available because the conditional distribution r(t+dt)|r(t) is
    Gaussian -- no discretisation error.

Hull-White (1990)
    dr(t) = (θ(t) - κ r(t)) dt + σ dW(t)

    The time-dependent drift θ(t) is chosen so that E[r(t)] equals the
    market instantaneous forward rate f(0,t) at all t. Requires an
    OISCurve (or any DiscountCurve) at construction. Simulated via
    Euler-Maruyama because θ(t) makes the exact formula less tractable
    in the discrete-time setting.

Rate convention
---------------
All rates -- initial value x0, parameters θ, σ, simulated paths -- are
in percent throughout. σ = 1.0 means 1% per √year. This is consistent
with the project-wide convention in OISCurve and all instrument pricers.
"""

import warnings
from typing import Optional

import numpy as np

from quant_risk.curves.base import DiscountCurve
from quant_risk.models.base import StochasticProcess


class VasicekProcess(StochasticProcess):
    """
    Vasicek (1977) mean-reverting short rate model.

    SDE
    ---
    dr(t) = κ(θ - r(t)) dt + σ dW(t)

    where:
      κ  -- mean reversion speed (1/year). Higher κ → faster pull to θ.
      θ  -- long-run mean rate in percent (e.g. 2.5 for 2.5%).
      σ  -- instantaneous volatility in percent/√year.

    Simulation method
    -----------------
    Exact. The analytical solution to the SDE is:

      r(t) = r(0) e^{-κt} + θ(1 - e^{-κt}) + σ ∫₀ᵗ e^{-κ(t-s)} dW(s)

    The stochastic integral is Gaussian with mean 0 and variance
    σ²(1 - e^{-2κt}) / (2κ), so the one-step transition is:

      r(t+dt) | r(t)  ~  N( μ(t), v² )
      μ(t) = r(t) e^{-κ dt} + θ(1 - e^{-κ dt})
      v²   = σ²(1 - e^{-2κ dt}) / (2κ)

    Using the exact distribution means step size dt has no effect on
    accuracy -- useful for demonstrating convergence in the notebook.

    Parameters
    ----------
    kappa : float
        Mean reversion speed in 1/year. Typical range: 0.01 – 1.0.
    theta : float
        Long-run mean short rate, decimal. Example: 0.025 for 2.5%.
    sigma : float
        Volatility, decimal/√year. Example: 0.005 for 0.5%/√year.
    """

    def __init__(self, kappa: float, theta: float, sigma: float) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma

    @property
    def name(self) -> str:
        return "Vasicek"

    @property
    def kappa(self) -> float:
        """Mean reversion speed in 1/year."""
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-run mean short rate, decimal."""
        return self._theta

    @property
    def sigma(self) -> float:
        """Volatility, decimal/√year."""
        return self._sigma

    def simulate(
        self,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate Vasicek paths using exact conditional distribution.

        Parameters
        ----------
        x0 : float
            Initial short rate, decimal (e.g. 0.025 for 2.5%).
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps. Step size dt = T / n_steps.
        n_paths : int
            Number of paths. Must be even when antithetic=True.
        antithetic : bool
            Use antithetic variates.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Rates in decimal.
        """
        dt = T / n_steps

        # Draw standard normals (antithetic pairs if requested)
        Z = self._draw_normals(n_paths, n_steps, antithetic, seed)

        # Pre-compute time-invariant exact simulation constants
        # These depend only on κ, θ, σ and dt -- not on the path value
        exp_neg_kdt = np.exp(-self._kappa * dt)
        mean_adj = self._theta * (1.0 - exp_neg_kdt)   # θ(1 - e^{-κdt})
        cond_std = self._sigma * np.sqrt(
            (1.0 - np.exp(-2.0 * self._kappa * dt)) / (2.0 * self._kappa)
        )   # σ √((1 - e^{-2κdt}) / 2κ)

        # Allocate path array and set initial condition
        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = x0

        # Step forward: exact conditional normal at each step
        for i in range(n_steps):
            # r(t+dt) = r(t)*e^{-κdt}  +  θ*(1-e^{-κdt})  +  σ_cond * Z
            paths[:, i + 1] = (
                paths[:, i] * exp_neg_kdt + mean_adj + cond_std * Z[:, i]
            )

        return paths

    def describe(self) -> str:
        return (
            f"Vasicek | κ={self._kappa:.4f} | θ={self._theta:.2%} | "
            f"σ={self._sigma:.4%}/√yr"
        )


class HullWhiteProcess(StochasticProcess):
    """
    Hull-White (1990) one-factor mean-reverting short rate model.

    SDE
    ---
    dr(t) = (θ(t) - κ r(t)) dt + σ dW(t)

    where θ(t) is a deterministic function of time chosen so that the
    model reproduces the observed term structure exactly:

      θ(t) = ∂f(0,t)/∂t  +  κ f(0,t)  +  σ²/(2κ) (1 - e^{-2κt})

    and f(0,t) = -∂/∂t ln P(0,t) is the market instantaneous forward rate.

    The first two terms ensure E[r(t)] = f(0,t). The third term is a
    convexity adjustment that arises from Jensen's inequality when moving
    between the T-forward and risk-neutral measures.

    Simulation method
    -----------------
    Euler-Maruyama, because θ(t) is time-dependent:

      r(t+dt) ≈ r(t) + (θ(t) - κ r(t)) dt + σ √dt Z

    θ(t) is evaluated numerically at each time step using central finite
    differences on the OISCurve's instantaneous_forward() method.

    Parameters
    ----------
    curve : DiscountCurve
        Market term structure (OISCurve). Used to compute θ(t) at each step.
    kappa : float
        Mean reversion speed in 1/year.
    sigma : float
        Volatility in percent/√year.
    """

    # Step size for the numerical derivative of the instantaneous forward rate.
    # We use 1/12 (one month) rather than 1/365 (one day) for two reasons:
    #   1. OISCurve.forward_rate() converts time to QuantLib Dates via int(T*365).
    #      Fractional-day values like n/365 can land just below an integer due to
    #      floating-point precision (e.g. 6/365*365 = 5.9999... → int = 5).
    #      Monthly steps like k/12 produce int(k/12 * 365) ≈ 30k days, which are
    #      well within integer precision.
    #   2. Monthly windows give a smooth, stable estimate of the forward rate curve
    #      and its slope -- appropriate for XVA simulation with monthly time steps.
    _FD_STEP = 1.0 / 12.0

    def __init__(
        self,
        curve: DiscountCurve,
        kappa: float,
        sigma: float,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._curve = curve
        self._kappa = kappa
        self._sigma = sigma   # percent / sqrt(year)

    @property
    def name(self) -> str:
        return "Hull-White"

    @property
    def kappa(self) -> float:
        """Mean reversion speed in 1/year."""
        return self._kappa

    @property
    def sigma(self) -> float:
        """Volatility in percent/√year."""
        return self._sigma

    def _theta(self, t: float) -> float:
        """
        Compute the Hull-White drift function θ(t) at a given time.

        θ(t) = ∂f(0,t)/∂t  +  κ f(0,t)  +  σ²/(2κ)(1 - e^{-2κt})

        The derivative ∂f/∂t is evaluated numerically. At t=0 we use a
        adjacent monthly forward windows to avoid OISCurve integer arithmetic issues.

        Parameters
        ----------
        t : float
            Time in years. Must be >= 0.

        Returns
        -------
        float
            θ(t) in percent/year (consistent with rates in percent).
        """
        H = self._FD_STEP  # one month

        # Floor t at H so that the left endpoint of the central difference
        # (t_eval - H) is always >= 0, giving forward_rate a valid period.
        t_eval = max(t, H)

        # Approximate instantaneous forward rate at t_eval as the continuously
        # compounded forward rate over the 1-month window [t_eval, t_eval + H].
        # This avoids OISCurve.instantaneous_forward(), which uses int(T*365)
        # arithmetic that produces off-by-one day errors at sub-daily T values.
        f_t = self._curve.forward_rate(t_eval, t_eval + H)

        # df(0,t)/dt via central differences on adjacent 1-month forward windows
        f_minus = self._curve.forward_rate(t_eval - H, t_eval)
        f_plus  = self._curve.forward_rate(t_eval + H, t_eval + 2.0 * H)
        df_dt   = (f_plus - f_minus) / (2.0 * H)

        # Convexity adjustment: sigma^2/(2*kappa) * (1 - e^{-2*kappa*t})
        convexity = (self._sigma ** 2 / (2.0 * self._kappa)) * (
            1.0 - np.exp(-2.0 * self._kappa * t_eval)
        )

        return df_dt + self._kappa * f_t + convexity

    def simulate(
        self,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate Hull-White paths using Euler-Maruyama discretisation.

        Parameters
        ----------
        x0 : float
            Initial short rate, decimal (e.g. 0.025 for 2.5%). Should equal
            f(0,0) for a fully calibrated simulation.
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps. Smaller dt → more accurate Euler scheme.
        n_paths : int
            Number of paths. Must be even when antithetic=True.
        antithetic : bool
            Use antithetic variates.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Rates in decimal.
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        Z = self._draw_normals(n_paths, n_steps, antithetic, seed)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = x0

        for i in range(n_steps):
            t = i * dt
            # θ(t) encodes all market calibration information at this point in time
            theta_t = self._theta(t)

            # Euler-Maruyama step:
            # r(t+dt) = r(t) + (θ(t) - κ r(t)) dt + σ √dt Z
            drift = (theta_t - self._kappa * paths[:, i]) * dt
            diffusion = self._sigma * sqrt_dt * Z[:, i]
            paths[:, i + 1] = paths[:, i] + drift + diffusion

        return paths

    def describe(self) -> str:
        return (
            f"Hull-White | κ={self._kappa:.4f} | "
            f"σ={self._sigma:.4%}/√yr | "
            f"curve={self._curve.valuation_date}"
        )


class CIRProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross (1985) mean-reverting short rate model.

    SDE
    ---
    dr(t) = κ(θ - r(t)) dt + σ √r(t) dW(t)

    The √r diffusion term makes volatility proportional to the square root of
    the rate level. This, combined with the Feller condition, guarantees
    r(t) ≥ 0 at all times — the key property distinguishing CIR from Vasicek.

    Feller condition
    ----------------
    2κθ > σ² ensures the origin is inaccessible: r(t) stays strictly positive.
    When 0 < 2κθ ≤ σ², the process can reach zero but remains non-negative.
    A UserWarning is raised at construction when the condition is not met.

    Simulation method
    -----------------
    Exact. The conditional distribution r(t+dt)|r(t) is a scaled non-central
    chi-squared:

      r(t+dt)|r(t)  ~  c · χ²(d, λ)

      c = σ²(1 - e^{-κdt}) / (4κ)          [scale, in percent]
      d = 4κθ / σ²                           [degrees of freedom, constant]
      λ = r(t) · e^{-κdt} / c               [non-centrality, path-dependent]

    Sampled directly via NumPy's noncentral_chisquare. No discretisation error.
    Non-negativity is structurally guaranteed: χ²(d, λ) ≥ 0 for all d, λ > 0.

    Antithetic variates are not supported. The non-central chi-squared
    distribution has no simple symmetric pairing analogous to Gaussian ±Z
    (λ is path-dependent, so antithetic twins diverge after the first step).
    Passing antithetic=True raises NotImplementedError. Use more paths instead.

    Rate convention
    ---------------
    x0 and θ in decimal (e.g. 0.025 for 2.5%).
    σ in √decimal/√year — different units from Vasicek. At the long-run mean,
    the instantaneous vol is σ√θ decimal/√year. For σ = 0.032 and θ = 0.025,
    this equals 0.032·√0.025 ≈ 0.00506 decimal/√year ≈ 0.506%/√year.

    Parameters
    ----------
    kappa : float
        Mean reversion speed in 1/year. Must be positive.
    theta : float
        Long-run mean short rate, decimal. Example: 0.025 for 2.5%.
    sigma : float
        Volatility coefficient, √decimal/√year. Must be positive.
    """

    def __init__(self, kappa: float, theta: float, sigma: float) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if 2.0 * kappa * theta <= sigma ** 2:
            warnings.warn(
                f"Feller condition 2κθ > σ² not satisfied "
                f"(2·{kappa}·{theta} = {2*kappa*theta:.4f} ≤ σ² = {sigma**2:.4f}). "
                "r(t) may reach zero.",
                UserWarning,
                stacklevel=2,
            )
        self._kappa = kappa
        self._theta = theta
        self._sigma = sigma

    @property
    def name(self) -> str:
        return "CIR"

    @property
    def kappa(self) -> float:
        """Mean reversion speed in 1/year."""
        return self._kappa

    @property
    def theta(self) -> float:
        """Long-run mean short rate, decimal."""
        return self._theta

    @property
    def sigma(self) -> float:
        """Volatility coefficient, √decimal/√year."""
        return self._sigma

    def simulate(
        self,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate CIR paths using the exact non-central chi-squared distribution.

        Parameters
        ----------
        x0 : float
            Initial short rate, decimal (e.g. 0.025 for 2.5%). Must be >= 0.
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps. Step size dt = T / n_steps.
        n_paths : int
            Number of paths.
        antithetic : bool
            Must be False. Raises NotImplementedError if True.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Rates in decimal, guaranteed >= 0.
        """
        if antithetic:
            raise NotImplementedError(
                "CIRProcess does not support antithetic variates. "
                "The non-central chi-squared exact simulation has no simple "
                "symmetric pairing analogous to Gaussian ±Z (λ is path-dependent, "
                "so antithetic twins diverge immediately). "
                "Use antithetic=False and increase n_paths instead."
            )
        if x0 < 0:
            raise ValueError(f"Initial rate x0 must be >= 0, got {x0}")

        dt = T / n_steps
        rng = np.random.default_rng(seed)

        exp_neg_kdt = np.exp(-self._kappa * dt)
        # Scale factor c = σ²(1 - e^{-κdt}) / (4κ), in percent
        c = self._sigma ** 2 * (1.0 - exp_neg_kdt) / (4.0 * self._kappa)
        # Degrees of freedom d = 4κθ / σ² — constant across all steps
        d = 4.0 * self._kappa * self._theta / (self._sigma ** 2)

        paths = np.empty((n_paths, n_steps + 1))
        paths[:, 0] = x0

        for i in range(n_steps):
            # Non-centrality λ = r(t) · e^{-κdt} / c — path-dependent
            # Clipped to 0 to guard against floating-point negatives near zero
            lam = np.maximum(paths[:, i] * exp_neg_kdt / c, 0.0)
            paths[:, i + 1] = c * rng.noncentral_chisquare(df=d, nonc=lam)

        return paths

    def describe(self) -> str:
        feller_ok = 2.0 * self._kappa * self._theta > self._sigma ** 2
        return (
            f"CIR | κ={self._kappa:.4f} | θ={self._theta:.2%} | "
            f"σ={self._sigma:.4%}√/√yr | "
            f"Feller={'met' if feller_ok else 'violated'}"
        )
