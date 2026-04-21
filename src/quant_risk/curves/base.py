# src/quant_risk/curves/base.py

"""
Abstract base class for discount curves.

All curve implementations -- static (OIS, NSS) and dynamic (Hull-White,
HJM) -- implement this interface. Instrument pricers and risk calculators
depend only on this abstraction, never on concrete implementations.
"""

from abc import ABC, abstractmethod
import numpy as np


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
            Zero rate in percent.
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
            Forward rate in percent.
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
            Instantaneous forward rate in percent.
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
            Array of zero rates in percent.
        """
        return np.array([self.zero_rate(T) for T in maturities])

    def describe(self) -> str:
        """Human readable summary."""
        return (
            f"{self.__class__.__name__} | "
            f"currency={self.currency} | "
            f"valuation_date={self.valuation_date}"
        )