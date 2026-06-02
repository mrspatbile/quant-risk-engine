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
from quant_risk.curves.models import NSSParameters


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
    NSSCurve(params)              -- construct from NSSParameters dataclass

    Day count : ACT/365 (government bonds)
    Currency  : EUR
    """

    def __init__(self, params: NSSParameters):
        """
        Construct NSS curve from validated parameters.

        Parameters
        ----------
        params : NSSParameters
            Frozen dataclass containing NSS parameters (beta0–3, tau1–2)
            and optional valuation_date. All validation happens in
            NSSParameters.__post_init__.
        """
        self._params = params
        self._beta0 = float(params.beta0)
        self._beta1 = float(params.beta1)
        self._beta2 = float(params.beta2)
        self._beta3 = float(params.beta3)
        self._tau1 = float(params.tau1)
        self._tau2 = float(params.tau2)
        self._valuation_date = str(params.valuation_date)

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
    def from_ecb(cls, rating: str = "AAA", date: str = None) -> "NSSCurve":
        """
        Fetch NSS parameters from ECB SDW and construct curve.

        Parameters
        ----------
        rating : str
            'AAA' (ECB AAA-rated issuers, default) or 'ALL' (all issuers).
        date : str, optional
            Target valuation date as ISO string (e.g. '2026-03-24').
            If None, uses the most recent available date.
            If exact date unavailable, falls back to most recent prior date
            with user notification.

        Returns
        -------
        NSSCurve
            Curve at the selected date.

        Raises
        ------
        ValueError
            If no data available for the requested date.
        """
        from quant_risk.data.ecb import ECBClient

        client = ECBClient()
        params_df = client.get_nss_parameters(last_n=60, rating=rating, date=date)
        row, val_date = cls._select_params_row(params_df, date)

        nss_params = NSSParameters(
            beta0=float(row["beta0"]),
            beta1=float(row["beta1"]),
            beta2=float(row["beta2"]),
            beta3=float(row["beta3"]),
            tau1=float(row["tau1"]),
            tau2=float(row["tau2"]),
            valuation_date=val_date,
        )
        return cls(nss_params)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_processed_dir() -> str:
        """Resolve data/processed/ relative to project root."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "../../.."))
        return os.path.join(root, "data", "processed")