# src/quant_risk/curves/base.py

"""
Abstract base class for discount curves.

All curve implementations -- static (OIS, NSS) and dynamic (Hull-White,
HJM) -- implement this interface. Instrument pricers and risk calculators
depend only on this abstraction, never on concrete implementations.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class DiscountCurve(ABC):
    """
    Abstract discount curve.

    Defines the minimal interface required by instrument pricers,
    EVE calculators, and dynamic model calibration routines.

    Implementations
    ---------------
    OISCurve   -- bootstrapped from ESTR OIS market instruments via QuantLib
    NSSCurve   -- parametric Nelson-Siegel-Svensson fit to government bonds
    """

    @staticmethod
    def _select_params_row(
        params: pd.DataFrame | pd.Series,
        date: str | pd.Timestamp | None = None,
    ) -> tuple:
        """
        Select a row from a time-series of parameters, with date fallback.

        Normalises the index to datetime, optionally falls back to the most
        recent date <= target date, and returns the row with its valuation
        date as an ISO string.

        Parameters
        ----------
        params : pd.DataFrame or pd.Series
            Time series of parameters, indexed by date (string or Timestamp).
        date : str, pd.Timestamp, or None
            Target date. If None, returns the last row. If a string, parsed
            as ISO 8601. If Timestamp, used directly.

        Returns
        -------
        tuple
            (row, val_date_str) where row is the selected Series and
            val_date_str is the date as YYYY-MM-DD.

        Raises
        ------
        ValueError
            If no data is available on or before the target date.
        """
        # Normalise index to datetime
        params.index = pd.to_datetime(params.index)

        # Convert date to Timestamp if provided
        if date is None:
            # Return last row
            row = params.iloc[-1]
            val_date = params.index[-1]
        else:
            target = pd.Timestamp(date)
            available = params.index[params.index <= target]
            if available.empty:
                raise ValueError(
                    f"No data available on or before {date}. "
                    f"Earliest available: {params.index[0].date()}"
                )
            row = params.loc[available[-1]]
            val_date = available[-1]

        # Return row and date as ISO string
        val_date_str = val_date.strftime("%Y-%m-%d")
        return row, val_date_str

    @property
    @abstractmethod
    def currency(self) -> str:
        """ISO 4217 currency code. Example: 'EUR'"""
        pass

    @property
    @abstractmethod
    def valuation_date(self) -> str:
        """Valuation date as ISO string. Example: '2026-03-24'"""
        pass

    @abstractmethod
    def discount(self, T: float) -> float:
        """
        Discount factor $P(0,T)$.

        Parameters
        ----------
        T : float
            Maturity in years. Must be > 0.

        Returns
        -------
        float
            Discount factor in (0, 1].
        """
        pass

    @abstractmethod
    def zero_rate(self, T: float) -> float:
        """
        Continuously compounded zero rate at maturity T.

        Parameters
        ----------
        T : float
            Maturity in years.

        Returns
        -------
        float
            Zero rate, decimal (e.g. 0.025 for 2.5%).
        """
        pass

    @abstractmethod
    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Continuously compounded forward rate for the period [T1, T2].

        Derived from discount factors:
        $f(T_1, T_2) = -\\frac{\\ln P(0,T_2) - \\ln P(0,T_1)}{T_2 - T_1}$

        Parameters
        ----------
        T1 : float
            Start of forward period in years.
        T2 : float
            End of forward period in years. Must be > T1.

        Returns
        -------
        float
            Forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        pass

    @abstractmethod
    def instantaneous_forward(self, T: float) -> float:
        """
        Instantaneous forward rate at maturity T.

        $f(0,T) = -\\frac{d}{dT} \\ln P(0,T)$

        Required for Hull-White and HJM model calibration.
        Computed numerically via finite differences in most implementations.

        Parameters
        ----------
        T : float
            Maturity in years.

        Returns
        -------
        float
            Instantaneous forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        pass

    def discount_vector(self, maturities: np.ndarray) -> np.ndarray:
        """
        Vectorised discount factors at multiple maturities.

        Default implementation calls discount() in a loop.
        Concrete classes may override for performance.

        Parameters
        ----------
        maturities : np.ndarray
            Array of maturities in years.

        Returns
        -------
        np.ndarray
            Array of discount factors.
        """
        return np.array([self.discount(T) for T in maturities])

    def zero_rate_vector(self, maturities: np.ndarray) -> np.ndarray:
        """
        Vectorised zero rates at multiple maturities.

        Parameters
        ----------
        maturities : np.ndarray
            Array of maturities in years.

        Returns
        -------
        np.ndarray
            Array of zero rates, decimal (e.g. 0.025 for 2.5%).
        """
        return np.array([self.zero_rate(T) for T in maturities])

    def describe(self) -> str:
        """Human readable summary."""
        return (
            f"{self.__class__.__name__} | "
            f"currency={self.currency} | "
            f"valuation_date={self.valuation_date}"
        )