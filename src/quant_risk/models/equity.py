# src/quant_risk/models/equity.py

"""
Equity and equity-like stochastic processes for Monte Carlo simulation.

GBMProcess
    dS/S = (r - q) dt + σ dW    (Geometric Brownian Motion)

    Constant volatility. Exact simulation via the log-normal transition density.
    This is the Black-Scholes model for equities, FX rates, and commodities.

LocalVolProcess
    dS/S = (r - q) dt + σ(S, t) dW    (Local Volatility)

    Time- and level-dependent volatility. Calibrated to the market implied-vol
    surface via Dupire's formula. Simulated via log-Euler because the exact
    transition density is not available when σ depends on S.

Rate convention
---------------
All rates, yields, and volatilities are in decimal throughout.
r = 0.025 means 2.5%/year. σ = 0.20 means 20%/year.
in exponential arguments — consistent with the project-wide convention.
"""

from typing import Callable, Optional

import numpy as np

from quant_risk.models.base import StochasticProcess


class GBMProcess(StochasticProcess):
    """
    Geometric Brownian Motion — the Black-Scholes equity model.

    SDE
    ---
    dS(t) = (r - q) S(t) dt + σ S(t) dW(t)

    Equivalently, applying Itô's lemma to X = ln S:

    d(ln S) = \\left(\\frac{r - q}{100} - \\frac{1}{2}\\left(\\frac{σ}{100}\\right)^2\\right) dt
              + \\frac{σ}{100} dW(t)

    Simulation method
    -----------------
    Exact. The log-return over one step is Gaussian:

    ln S(t+dt) | S(t)  ~  N\\!\\left(ln S(t) + μ_{dt},\\; v_{dt}^2\\right)

    where:
      μ_{dt} = \\left(\\frac{r-q}{100} - \\frac{σ^2}{20000}\\right) dt
      v_{dt} = \\frac{σ}{100} \\sqrt{dt}

    This is exact regardless of step size — no Euler discretisation error.

    Parameters
    ----------
    r : float
        Risk-free rate, decimal (e.g. 0.025 for 2.5%).
    sigma : float
        Volatility, decimal (e.g. 0.20 for 20%).
    q : float, optional
        Continuous dividend yield, decimal. Default 0.0.
    """

    def __init__(self, r: float, sigma: float, q: float = 0.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self._r     = r
        self._sigma = sigma
        self._q     = q

    @property
    def name(self) -> str:
        return "GBM"

    @property
    def r(self) -> float:
        """Risk-free rate, decimal."""
        return self._r

    @property
    def sigma(self) -> float:
        """Volatility, decimal."""
        return self._sigma

    @property
    def q(self) -> float:
        """Dividend yield, decimal."""
        return self._q

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
        Simulate GBM price paths using exact log-normal simulation.

        Parameters
        ----------
        x0 : float
            Initial stock price (e.g. 100.0).
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps. Step size dt = T / n_steps.
        n_paths : int
            Number of paths. Must be even when antithetic=True.
        antithetic : bool
            Use antithetic variates for variance reduction.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Price levels (not log-prices).
            paths[:, 0] == x0 for all paths.
        """
        dt = T / n_steps

        Z = self._draw_normals(n_paths, n_steps, antithetic, seed)

        # Pre-compute time-invariant exact simulation constants
        # log-return drift per step: (r-q) × dt - σ²/2 × dt
        drift_step = ((self._r - self._q) - self._sigma ** 2 / 2) * dt
        # log-return std per step
        vol_step   = self._sigma * np.sqrt(dt)

        # Simulate in log-space for numerical stability, then exponentiate
        log_S = np.empty((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(x0)

        for i in range(n_steps):
            # ln S(t+dt) = ln S(t) + drift_step + vol_step * Z
            log_S[:, i + 1] = log_S[:, i] + drift_step + vol_step * Z[:, i]

        return np.exp(log_S)

    def describe(self) -> str:
        return (
            f"GBM | r={self._r:.2%} | σ={self._sigma:.2%} | q={self._q:.2%}"
        )


class LocalVolProcess(StochasticProcess):
    """
    Local Volatility process — Dupire (1994).

    SDE
    ---
    dS(t) = (r - q) S(t) dt + σ(S(t), t) S(t) dW(t)

    where σ(S, t) is the local volatility surface, calibrated to reproduce
    the market's implied-vol smile at all strikes and maturities simultaneously.

    Simulation method
    -----------------
    Log-Euler (Euler applied to ln S). Because σ depends on S, there is no
    exact simulation formula. The log-Euler scheme:

    ln S(t+dt) ≈ ln S(t) + \\left(\\frac{r-q}{100} - \\frac{σ(S(t),t)^2}{20000}\\right) dt
                 + \\frac{σ(S(t),t)}{100} \\sqrt{dt}\\;Z

    Log-Euler preserves positivity (S > 0 for all paths) and has weak order 1.

    Parameters
    ----------
    r : float
        Risk-free rate in percent per year.
    local_vol_fn : Callable[[float, float], float]
        Local volatility function σ(S, t) → vol in percent per year.
        Called with current stock price S and time t.
    q : float, optional
        Continuous dividend yield in percent per year. Default 0.0.
    """

    def __init__(
        self,
        r: float,
        local_vol_fn: Callable[[float, float], float],
        q: float = 0.0,
    ) -> None:
        self._r           = r
        self._local_vol_fn = local_vol_fn
        self._q           = q

    @property
    def name(self) -> str:
        return "LocalVol"

    @property
    def r(self) -> float:
        """Risk-free rate in percent."""
        return self._r

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
        Simulate Local Volatility paths using log-Euler discretisation.

        Parameters
        ----------
        x0 : float
            Initial stock price.
        T : float
            Time horizon in years.
        n_steps : int
            Number of time steps. Smaller dt → less Euler bias.
        n_paths : int
            Number of paths. Must be even when antithetic=True.
        antithetic : bool
            Use antithetic variates.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). Price levels.
        """
        dt      = T / n_steps
        sqrt_dt = np.sqrt(dt)

        Z = self._draw_normals(n_paths, n_steps, antithetic, seed)

        log_S = np.empty((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(x0)

        S_curr = np.full(n_paths, x0)   # current price level for vol lookup

        for i in range(n_steps):
            t = i * dt
            # Evaluate local vol at each path's current price and time
            # local_vol_fn(S, t) returns decimal
            sigma_loc = np.array([
                self._local_vol_fn(s, t) for s in S_curr
            ])

            # Log-Euler step
            drift     = (self._r - self._q - sigma_loc ** 2 / 2) * dt
            diffusion = sigma_loc * sqrt_dt * Z[:, i]
            log_S[:, i + 1] = log_S[:, i] + drift + diffusion

            # Update current prices for next step's vol lookup
            S_curr = np.exp(log_S[:, i + 1])

        return np.exp(log_S)

    def describe(self) -> str:
        return f"LocalVol | r={self._r:.2%} | σ(S,t) user-defined"
