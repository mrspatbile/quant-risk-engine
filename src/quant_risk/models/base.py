# src/quant_risk/models/base.py

"""
Abstract base class for stochastic processes used in Monte Carlo simulation.

All implementations -- VasicekProcess, HullWhiteProcess, GBMProcess --
inherit from StochasticProcess and implement simulate().

Design principles
-----------------
- simulate() is the single public entry point. Callers never need to know
  which discretisation scheme (Euler, exact) is used internally.
- Antithetic variate logic lives here, not in each subclass. Concrete classes
  receive the full (n_paths, n_steps) normal draws and simulate without
  knowing whether those draws form antithetic pairs.
- No QuantLib dependency at the ABC level. Models that need to calibrate to
  a term structure receive a DiscountCurve in their own __init__.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class StochasticProcess(ABC):
    """
    Abstract stochastic process for Monte Carlo path generation.

    Defines the interface consumed by MCSimulator (QRE-51) and the
    XVA exposure engines (QRE-52). Every concrete process produces
    a paths array of shape (n_paths, n_steps + 1), where column 0
    holds the initial value and each subsequent column is one time step
    forward.

    Implementations
    ---------------
    VasicekProcess    -- mean-reverting short rate, constant parameters,
                         exact simulation via conditional normal distribution
    HullWhiteProcess  -- mean-reverting short rate, time-dependent drift
                         calibrated to the OIS term structure, Euler scheme
    GBMProcess        -- geometric Brownian motion for equity (QRE-50)
    """

    @abstractmethod
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
        Generate sample paths of the process.

        Parameters
        ----------
        x0 : float
            Initial value of the process. For short rate models this is
            the initial short rate in percent (e.g. 2.5 means 2.5%).
        T : float
            Time horizon in years.
        n_steps : int
            Number of discrete time steps. Step size dt = T / n_steps.
        n_paths : int
            Number of Monte Carlo paths. Must be even when antithetic=True.
        antithetic : bool
            Use antithetic variates for variance reduction. For each
            standard normal draw Z, also simulate with -Z. This halves
            the effective variance for a given path budget by ensuring
            that every upward shock has a paired downward shock.
            Paths [0 : n_paths//2] are the base draws; paths
            [n_paths//2 : n_paths] are the antithetic twins.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps + 1). paths[:, 0] == x0 for all paths.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'Vasicek' or 'Hull-White'."""
        ...

    def _draw_normals(
        self,
        n_paths: int,
        n_steps: int,
        antithetic: bool,
        seed: Optional[int],
    ) -> np.ndarray:
        """
        Draw standard normal variates, optionally as antithetic pairs.

        Antithetic construction: draw n_paths // 2 independent rows Z,
        then stack with -Z. This guarantees exact symmetry — the antithetic
        twin of path i is always path i + n_paths // 2.

        Parameters
        ----------
        n_paths : int
            Total number of paths. Must be even if antithetic=True.
        n_steps : int
            Number of time steps (one draw per step per path).
        antithetic : bool
            Whether to build antithetic pairs.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Shape (n_paths, n_steps). Each row is one path's noise sequence.
        """
        rng = np.random.default_rng(seed)

        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError(
                    f"n_paths must be even when antithetic=True, got {n_paths}."
                )
            # Draw the base half, then reflect: Z_anti = -Z_base
            half = n_paths // 2
            Z_base = rng.standard_normal((half, n_steps))
            return np.vstack([Z_base, -Z_base])

        return rng.standard_normal((n_paths, n_steps))

    def describe(self) -> str:
        """Human-readable parameter summary."""
        return f"{self.name} stochastic process"
