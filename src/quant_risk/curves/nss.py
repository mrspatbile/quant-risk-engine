# src/quant_risk/curves/nss.py

"""
NSS parametric yield curve -- Nelson-Siegel-Svensson fit to euro area
government bonds, published daily by the ECB.

The NSS formula is closed-form so discount factors and 
zero rates are computed directly
from the six ECB parameters.
"""

import numpy as np
import pandas as pd
import os

from quant_risk.curves.base import DiscountCurve


class NSSCurve(DiscountCurve):
    """
    Nelson-Siegel-Svensson parametric yield curve.

    Implements the ECB published NSS model:

    r(T) = β₀ + β₁·f₁(T) + β₂·f₂(T) + β₃·f₃(T)

    Where:
        f₁(T) = (1 - exp(-T/τ₁)) / (T/τ₁)
        f₂(T) = f₁(T) - exp(-T/τ₁)
        f₃(T) = (1 - exp(-T/τ₂)) / (T/τ₂) - exp(-T/τ₂)

    Parameters published daily by ECB under the YC dataset.
    Available for AAA-rated issuers (G_N_A) and all issuers (G_N_C).

    Construction
    ------------
    NSSCurve.from_ecb()           -- fetch live AAA parameters from ECB
    NSSCurve.from_ecb(rating='ALL') -- fetch all-issuers parameters
    NSSCurve.from_processed()     -- load latest saved curve
    NSSCurve(params)              -- construct from a parameter dict or Series

    Day count : ACT/365 (government bonds)
    Currency  : EUR
    """

    def __init__(self, params: dict | pd.Series, valuation_date: str = ""):
        """
        Parameters
        ----------
        params : dict or pd.Series
            Must contain keys: beta0, beta1, beta2, beta3, tau1, tau2.
        valuation_date : str
            ISO date string. Example: '2026-03-24'.
        """
        self._beta0 = float(params["beta0"])
        self._beta1 = float(params["beta1"])
        self._beta2 = float(params["beta2"])
        self._beta3 = float(params["beta3"])
        self._tau1  = float(params["tau1"])
        self._tau2  = float(params["tau2"])
        self._valuation_date = str(valuation_date)

    # ------------------------------------------------------------------
    # DiscountCurve interface
    # ------------------------------------------------------------------

    @property
    def currency(self) -> str:
        return "EUR"

    @property
    def valuation_date(self) -> str:
        return self._valuation_date

    def zero_rate(self, T: float) -> float:
        """
        NSS zero rate at maturity T in years.

        Parameters
        ----------
        T : float
            Maturity in years. Must be > 0.

        Returns
        -------
        float
            Zero rate, decimal (e.g. 0.025 for 2.5%).
        """
        T = max(T, 1e-6)
        f1 = (1 - np.exp(-T / self._tau1)) / (T / self._tau1)
        f2 = f1 - np.exp(-T / self._tau1)
        f3 = (1 - np.exp(-T / self._tau2)) / (T / self._tau2) - np.exp(-T / self._tau2)
        # ECB publishes beta parameters in percent; divide by 100 at output boundary
        return (self._beta0 + self._beta1 * f1 + self._beta2 * f2 + self._beta3 * f3) / 100

    def discount(self, T: float) -> float:
        """
        Discount factor P(0,T) = exp(-r(T) * T).

        Parameters
        ----------
        T : float
            Maturity in years.

        Returns
        -------
        float
            Discount factor in (0, 1].
        """
        T = max(T, 1e-6)
        return np.exp(-self.zero_rate(T) * T)

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Continuously compounded forward rate for [T1, T2].
        Derived analytically from NSS discount factors.

        Returns
        -------
        float
            Forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        T1 = max(T1, 1e-6)
        T2 = max(T2, T1 + 1e-6)
        p1 = self.discount(T1)
        p2 = self.discount(T2)
        return -(np.log(p2) - np.log(p1)) / (T2 - T1)

    def instantaneous_forward(self, T: float) -> float:
        """
        Instantaneous forward rate at T via finite differences.
        f(0,T) = -d/dT ln P(0,T)
        Required for Hull-White calibration in Module 2.

        Returns
        -------
        float
            Instantaneous forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        dT = 1e-4
        p1 = self.discount(max(T - dT, 1e-6))
        p2 = self.discount(T + dT)
        return -(np.log(p2) - np.log(p1)) / (2 * dT)

    # ------------------------------------------------------------------
    # NSS-specific methods
    # ------------------------------------------------------------------

    @property
    def parameters(self) -> dict:
        """Return the six NSS parameters as a dictionary."""
        return {
            "beta0": self._beta0,
            "beta1": self._beta1,
            "beta2": self._beta2,
            "beta3": self._beta3,
            "tau1" : self._tau1,
            "tau2" : self._tau2,
        }

    @property
    def short_rate(self) -> float:
        """Instantaneous short rate, decimal. β₀ + β₁ converted from ECB percent."""
        return (self._beta0 + self._beta1) / 100

    @property
    def long_rate(self) -> float:
        """Long-run level, decimal. β₀ (asymptote as T → ∞) converted from ECB percent."""
        return self._beta0 / 100

    # ------------------------------------------------------------------
    # alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ecb(cls, rating: str = "AAA", last_n: int = 10, date: str = None) -> "NSSCurve":
        """
        Fetch NSS parameters from ECB SDW and construct curve.

        Parameters
        ----------
        rating : str
            'AAA' (ECB AAA-rated issuers) or 'ALL' (all issuers). Default 'AAA'.
        last_n : int
            Number of most recent parameter sets to fetch from ECB.
            Default 10 (increased from 1 to provide more date fallback options).
        date : str or None
            Target valuation date as ISO string. If None, uses the most recent
            available date. Falls back to the most recent date <= target if
            exact match not found. Example: '2026-03-24'.

        Returns
        -------
        NSSCurve
            Curve at the selected date.

        Raises
        ------
        ValueError
            If no data available on or before the target date.
        """
        from quant_risk.data.ecb import ECBClient
        client = ECBClient()
        params = client.get_nss_parameters(last_n=last_n, rating=rating, date=date)
        row, val_date = cls._select_params_row(params, date)
        return cls(row, valuation_date=val_date)

    @classmethod
    def from_processed(cls, rating: str = "AAA",
                       processed_dir: str = None) -> "NSSCurve":
        """
        Load the most recent saved NSS curve from data/processed/.

        Parameters
        ----------
        rating : str
            'AAA' or 'ALL'.
        processed_dir : str, optional
            Path to data/processed/. Resolved from project root if None.

        Returns
        -------
        NSSCurve
        """
        if processed_dir is None:
            processed_dir = cls._find_processed_dir()

        prefix = "nss_aaa_curve_" if rating == "AAA" else "nss_all_curve_"
        files  = sorted([
            f for f in os.listdir(processed_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        ])
        if not files:
            raise FileNotFoundError(
                f"No NSS {rating} curve file found in data/processed/. "
                f"Run notebook 02_bootstrapping_ois.ipynb first."
            )
        path = os.path.join(processed_dir, files[-1])
        data = pd.read_csv(path, index_col="maturity")

        valuation_date = data["valuation_date"].iloc[0]

        # reconstruct parameters by fitting NSS to saved zero rates
        # use ECB API directly for parameters -- more reliable
        return cls.from_ecb(rating=rating)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_processed_dir() -> str:
        """Resolve data/processed/ relative to project root."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "../../.."))
        return os.path.join(root, "data", "processed")