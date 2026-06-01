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
import QuantLib as ql
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
            where type is 'coupon', 'principal', or 'premium' (CDS).
        """
        pass

    @abstractmethod
    def npv(self, curve: DiscountCurve) -> float:
        """
        Net present value in domestic currency. Always a scalar float.

        Use this for portfolio aggregation, XVA exposure inputs, and any
        calculation that requires a consistent numeric type across instrument
        types. price() is left unchanged and may return a dict on some
        instruments.

        Parameters
        ----------
        curve : DiscountCurve
            Discount curve.

        Returns
        -------
        float
            NPV in currency units.
        """
        pass

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        """
        Sensitivity at each caller-specified tenor vertex.

        Bumps each pillar rate by `bump` independently and revalues.
        The sum approximates the total DV01 subject to interpolation
        overlap between adjacent vertices.

        The caller supplies the vertex list — this library has no opinion
        on which tenors regulators prescribe.

        Parameters
        ----------
        curve : DiscountCurve
            Baseline discount curve.
        tenors : list[float]
            Vertex maturities in years, e.g. [0.25, 0.5, 1.0, 2.0, 5.0, 10.0].
        bump : float
            Rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        dict[float, float]
            {tenor_years: sensitivity} in currency units per 1bp.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement rate_sensitivities. "
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

    def _disc_handle_from_curve(
        self, curve: "DiscountCurve"
    ) -> ql.YieldTermStructureHandle:
        """
        Extract a QuantLib discount handle from a DiscountCurve wrapper.

        Single access point for curve._ql_curve across all instrument pricers.
        Centralised here so a refactor of OISCurve internals has one breakage
        point instead of one per instrument.
        """
        return ql.YieldTermStructureHandle(curve._ql_curve)

    @staticmethod
    def _parse_date(date_str: str) -> ql.Date:
        """Parse ISO date string 'YYYY-MM-DD' to QuantLib Date."""
        parts = date_str.split("-")
        return ql.Date(int(parts[2]), int(parts[1]), int(parts[0]))
    
    def _validate(self) -> None:
        """Override in concrete classes to validate constructor arguments."""
        pass


    @staticmethod
    def _with_eval_date(date: ql.Date, fn):
        """Execute fn with the QuantLib global evaluation date set to date.

        Saves the prior value and restores it in a finally block, so the
        process-global clock is always left in the state it was found — even
        if fn raises. Every public pricing method must call this instead of
        setting ql.Settings.instance().evaluationDate directly.
        """
        prev = ql.Settings.instance().evaluationDate
        try:
            ql.Settings.instance().evaluationDate = date
            return fn()
        finally:
            ql.Settings.instance().evaluationDate = prev

