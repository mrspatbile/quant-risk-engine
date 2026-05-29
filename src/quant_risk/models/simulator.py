# src/quant_risk/models/simulator.py

"""
MCSimulator — Monte Carlo simulation and exposure profile engine.

Wraps a StochasticProcess and provides:

- Path simulation with configurable parameters
- Stochastic discount factors via path-integral: D(0,t) = exp(-∫₀ᵗ r(s)ds)
- Exposure profiles: EE, NEE, PFE, discounted EE — the raw inputs to CVA/FVA/MVA
- Instrument pricing via user-supplied payoff callables

Rate convention
---------------
Paths are assumed to carry SHORT RATES in percent throughout (consistent with
VasicekProcess and HullWhiteProcess). The path integral is divided by 100 before
exponentiation. For equity simulations (GBMProcess), discount the terminal payoff
with a deterministic exp(-r/100 * T) factor in the payoff function itself.

Design contract for payoff and MTM functions
--------------------------------------------
price() expects:
    payoff_fn(paths: np.ndarray) -> np.ndarray(n_paths,)
    Terminal payoff only (no discounting — the simulator discounts internally).

exposure_profile() expects:
    mtm_fn(paths: np.ndarray, t: float) -> np.ndarray(n_paths,)
    Present value at time t of all remaining cashflows from t to maturity.
    Must be computable from paths[:, :idx] — the path up to time t.
"""

from typing import Callable, Optional

import numpy as np

from quant_risk.models.base import StochasticProcess


class MCSimulator:
    """
    Monte Carlo simulation engine for XVA exposure calculation.

    Parameters
    ----------
    process : StochasticProcess
        The stochastic process to simulate (e.g. VasicekProcess, HullWhiteProcess).
    x0 : float
        Initial value (short rate in percent for rate processes; price for equity).
    T : float
        Simulation horizon in years.
    n_steps : int
        Number of time steps. Step size dt = T / n_steps.
    n_paths : int
        Number of Monte Carlo paths. Must be even when antithetic=True.
    antithetic : bool
        Use antithetic variates. Default True — recommended for XVA.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        process: StochasticProcess,
        x0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self._process   = process
        self._x0        = x0
        self._T         = T
        self._n_steps   = n_steps
        self._n_paths   = n_paths
        self._dt        = T / n_steps

        # Simulate all paths at construction time
        self._paths = process.simulate(
            x0        = x0,
            T         = T,
            n_steps   = n_steps,
            n_paths   = n_paths,
            antithetic= antithetic,
            seed      = seed,
        )

        # Pre-compute cumulative path integral ∫₀ᵗ r(s)ds at each grid point.
        # Divides by 100 to convert percent → decimal before integration.
        # Shape: (n_paths, n_steps) — cum_int[:, k] = ∫₀^{(k+1)·dt} r(s)ds
        self._cum_int = np.cumsum(
            self._paths[:, :-1] / 100.0 * self._dt, axis=1
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def paths(self) -> np.ndarray:
        """Simulated paths. Shape (n_paths, n_steps + 1)."""
        return self._paths

    @property
    def dt(self) -> float:
        """Simulation step size in years."""
        return self._dt

    @property
    def T(self) -> float:
        """Simulation horizon in years."""
        return self._T

    @property
    def n_paths(self) -> int:
        return self._n_paths

    @property
    def n_steps(self) -> int:
        return self._n_steps

    # ------------------------------------------------------------------
    # Discount factors
    # ------------------------------------------------------------------

    def sdf(self, t: float) -> np.ndarray:
        """
        Stochastic discount factor from t=0 to t along each path.

        D(0, t) = exp(-∫₀ᵗ r(s) ds)

        Parameters
        ----------
        t : float
            Time in years. Clamped to [dt, T].

        Returns
        -------
        np.ndarray
            Shape (n_paths,).
        """
        idx = self._t_to_idx(t)
        return np.exp(-self._cum_int[:, idx])

    def sdf_between(self, t1: float, t2: float) -> np.ndarray:
        """
        Stochastic discount factor for the period [t1, t2].

        D(t1, t2) = exp(-∫_{t1}^{t2} r(s) ds)

        Parameters
        ----------
        t1, t2 : float
            Period start and end in years. t2 > t1 >= 0.

        Returns
        -------
        np.ndarray
            Shape (n_paths,).
        """
        idx2 = self._t_to_idx(t2)
        c2   = self._cum_int[:, idx2]

        if t1 <= self._dt:
            # Period starts at (or before) the first grid point — c1 = 0
            return np.exp(-c2)

        idx1 = self._t_to_idx(t1)
        c1   = self._cum_int[:, idx1]
        return np.exp(-(c2 - c1))

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        payoff_fn: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        """
        Price an instrument as E^Q[D(0,T) × payoff(S(T))].

        The payoff function receives the full paths array and should return
        the undiscounted terminal payoff for each path. Discounting from T to 0
        is applied internally using the path-integral SDF.

        Parameters
        ----------
        payoff_fn : Callable[[np.ndarray], np.ndarray]
            payoff_fn(paths) → (n_paths,) array of terminal payoffs.

        Returns
        -------
        float
            Monte Carlo price estimate.
        """
        payoffs = np.asarray(payoff_fn(self._paths))
        disc    = self.sdf(self._T)
        return float((payoffs * disc).mean())

    # ------------------------------------------------------------------
    # Exposure profiles
    # ------------------------------------------------------------------

    def exposure_profile(
        self,
        mtm_fn: Callable[[np.ndarray, float], np.ndarray],
        dates: np.ndarray,
        pfe_quantile: float = 0.95,
    ) -> dict:
        """
        Compute the exposure profile of an instrument at specified dates.

        At each date t, mtm_fn returns V(t) — the present value of remaining
        cashflows from t to the instrument's maturity. The exposure profile
        summarises the distribution of V(t) across paths.

        Definitions
        -----------
        EE(t)      = E^Q[max(V(t), 0)]           Expected Exposure
        NEE(t)     = E^Q[min(V(t), 0)]           Negative Expected Exposure
        PFE(t)     = quantile(V(t), q)           Potential Future Exposure
        EE_disc(t) = E^Q[D(0,t) × max(V(t), 0)] Discounted EE  ← CVA integrand
        EPE        = mean(EE(t))                 Expected Positive Exposure

        Parameters
        ----------
        mtm_fn : Callable[[np.ndarray, float], np.ndarray]
            mtm_fn(paths, t) → (n_paths,) MTM of the instrument at time t.
        dates : np.ndarray
            Evaluation dates in years, e.g. np.arange(0.5, T + 0.5, 0.5).
        pfe_quantile : float
            Quantile for PFE. Default 0.95 (95th percentile).

        Returns
        -------
        dict with keys:
            dates   : np.ndarray  (n_dates,)         input dates
            mtm     : np.ndarray  (n_paths, n_dates)  full MTM surface
            EE      : np.ndarray  (n_dates,)
            NEE     : np.ndarray  (n_dates,)
            PFE     : np.ndarray  (n_dates,)
            EE_disc : np.ndarray  (n_dates,)          discounted EE
            EPE     : float                           time-average of EE
        """
        n_dates = len(dates)
        mtm     = np.zeros((self._n_paths, n_dates))

        for j, t in enumerate(dates):
            mtm[:, j] = mtm_fn(self._paths, t)

        # Undiscounted exposure metrics
        ee  = np.maximum(mtm, 0).mean(axis=0)
        nee = np.minimum(mtm, 0).mean(axis=0)
        pfe = np.percentile(mtm, pfe_quantile * 100, axis=0)

        # Discounted EE: E[D(0,t) × max(V(t), 0)] — the CVA integrand
        disc   = np.column_stack([self.sdf(t) for t in dates])  # (n_paths, n_dates)
        ee_disc = (disc * np.maximum(mtm, 0)).mean(axis=0)

        # EPE: time-average of EE — used in FVA approximations
        epe = float(ee.mean())

        return {
            "dates"   : dates,
            "mtm"     : mtm,
            "EE"      : ee,
            "NEE"     : nee,
            "PFE"     : pfe,
            "EE_disc" : ee_disc,
            "EPE"     : epe,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _t_to_idx(self, t: float) -> int:
        """Convert time t to the nearest cum_int column index (0-based, clamped)."""
        idx = int(round(t / self._dt)) - 1
        return max(0, min(idx, self._n_steps - 1))

    def describe(self) -> str:
        return (
            f"MCSimulator | {self._process.name} | "
            f"T={self._T}Y | n_steps={self._n_steps} | n_paths={self._n_paths:,}"
        )
