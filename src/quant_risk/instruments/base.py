# src/quant_risk/instruments/base.py

"""
Abstract base class for all instruments in the Quant Risk Engine.

All instrument implementations -- Bond, IRSwap, FXForward, VanillaOption --
implement this interface. Risk calculators and portfolio engines depend only
on this abstraction, never on concrete implementations.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from quant_risk.curves.base import DiscountCurve


class Instrument(ABC):
    """
    Abstract instrument.

    Defines the minimal interface required by pricing engines,
    risk calculators, and regulatory capital models (FRTB, IRRBB, XVA).

    All pricers receive a DiscountCurve -- they do not fetch market data
    themselves. This separation allows the same instrument to be priced
    against a live OIS curve, a shocked IRRBB curve, or a Monte Carlo
    simulated path without modifying the instrument code.

    Implementations
    ---------------
    Bond          -- fixed rate government or corporate bond
    IRSwap        -- interest rate swap, multi-curve OIS/EURIBOR
    FXForward     -- FX forward via covered interest rate parity
    VanillaOption -- European option via Black-Scholes-Merton
    """

    @property
    @abstractmethod
    def currency(self) -> str:
        """ISO 4217 currency code. Example: 'EUR'"""
        pass

    @property
    @abstractmethod
    def notional(self) -> float:
        """Instrument notional in currency units."""
        pass

    @abstractmethod
    def price(self, curve: DiscountCurve) -> float:
        """
        Full revaluation price given a discount curve.

        Parameters
        ----------
        curve : DiscountCurve
            Discount curve for present value calculation.

        Returns
        -------
        float
            Instrument price in currency units.
        """
        pass

    @abstractmethod
    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        Price change for a 1bp parallel shift in the discount curve.

        DV01 = Price(curve + 1bp) - Price(curve)

        Sign convention: positive DV01 means the instrument loses value
        when rates rise -- standard for fixed rate assets.

        Parameters
        ----------
        curve : DiscountCurve
            Baseline discount curve.
        bump : float
            Size of the rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        float
            DV01 in currency units per 1bp.
        """
        pass

    @abstractmethod
    def duration(self, curve: DiscountCurve) -> float:
        """
        Modified duration in years.

        $ModDuration = -\\frac{1}{P} \\frac{\\partial P}{\\partial y}$

        Parameters
        ----------
        curve : DiscountCurve
            Discount curve.

        Returns
        -------
        float
            Modified duration in years.
        """
        pass

    @abstractmethod
    def cash_flows(self) -> pd.DataFrame:
        """
        Scheduled cash flows.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [date, amount, type]
            where type is 'coupon' or 'principal'.
        """
        pass

    def key_rate_dv01(
        self,
        curve: DiscountCurve,
        tenors: list = None,
        bump: float = 0.0001,
    ) -> dict:
        """
        DV01 per tenor bucket -- FRTB SA delta risk sensitivity.

        Bumps each pillar rate by 1bp independently and revalues.
        The sum of key rate DV01s approximates the total DV01 subject
        to interpolation overlap between adjacent buckets.

        FRTB SA prescribed vertices (CRR3 Article 325):
        0.25Y, 0.5Y, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y

        Parameters
        ----------
        curve : DiscountCurve
            Baseline discount curve.
        tenors : list, optional
            Tenor buckets in years. Defaults to FRTB SA vertices.
        bump : float
            Size of the rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        dict
            {tenor_label: dv01_value} in currency units per 1bp.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement key_rate_dv01. "
            "Override this method in the concrete class."
        )

    def pv01(self, curve: DiscountCurve) -> float:
        """
        Alias for dv01 -- EUR market convention name.

        Parameters
        ----------
        curve : DiscountCurve

        Returns
        -------
        float
            PV01 in currency units per 1bp.
        """
        return self.dv01(curve)

    def describe(self) -> str:
        """Human readable summary."""
        return (
            f"{self.__class__.__name__} | "
            f"currency={self.currency} | "
            f"notional={self.notional:,.0f}"
        )