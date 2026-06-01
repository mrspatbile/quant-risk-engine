"""
EquityForward — vanilla equity forward under continuous dividend yield.

Pricing
-------
Forward price : F = S · exp((r - q) · T)
NPV           : sign · notional · (F - K) · df(T)

where sign = +1 for long (receive asset), -1 for short (deliver asset).

Rate convention : rates in decimal throughout (r = 0.035 means 3.5%).
Dates           : ql.Date at public boundaries; ISO string via _parse_date.
Notional        : currency units, no implicit scaling.
"""

import pandas as pd
import numpy as np
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


class EquityForward(Instrument):
    """
    Vanilla equity forward priced under continuous dividend yield.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Agreed delivery price K.
    maturity_date : ql.Date
        Settlement date of the forward.
    valuation_date : ql.Date
        Pricing date.
    div_yield : float
        Continuous dividend yield of the underlying, decimal. Default 0.
    notional_ : float
        Contract size in currency units. Default 1.
    long : bool
        True = long forward (receive asset, pay strike).
        False = short forward (deliver asset, receive strike). Default True.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        maturity_date: ql.Date,
        valuation_date: ql.Date,
        div_yield: float = 0.0,
        notional_: float = 1.0,
        long: bool = True,
        currency_: str = 'EUR',
    ):
        self._spot          = spot
        self._strike        = strike
        self._maturity_date = maturity_date
        self._valuation_date = valuation_date
        self._div_yield     = div_yield
        self._notional_     = notional_
        self._long          = long
        self._currency_     = currency_
        self._sign          = 1.0 if long else -1.0
        self._day_count     = ql.Actual365Fixed()
        self._T             = self._day_count.yearFraction(valuation_date, maturity_date)
        self._validate()

    def _validate(self) -> None:
        if self._spot <= 0:
            raise ValueError(f"spot must be positive, got {self._spot}")
        if self._strike <= 0:
            raise ValueError(f"strike must be positive, got {self._strike}")
        if self._T < 0:
            raise ValueError("maturity_date must be after valuation_date")
        if not (0.0 <= self._div_yield < 1.0):
            raise ValueError(f"div_yield must be in [0, 1), got {self._div_yield}")

    # ── Instrument ABC ────────────────────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """
        NPV of the equity forward in currency units.

        NPV = sign · notional · (F - K) · df(T)

        where F = S · exp((r - q) · T) and df(T) is the OIS discount factor.

        Parameters
        ----------
        curve : DiscountCurve
            OIS discount curve for rate and discount factor extraction.

        Returns
        -------
        float
            NPV in currency units. Positive = asset for long, liability for short.
        """
        df  = curve.discount(self._T)
        fwd = self.forward_price(curve)
        return self._sign * self._notional_ * (fwd - self._strike) * df

    def npv(self, curve: DiscountCurve) -> float:
        """NPV in currency units."""
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        Change in NPV for a 1bp parallel shift in the OIS curve.

        Computed analytically: extract r from the discount factor, apply bump,
        revalue forward price and discount factor at r + bump.

        Parameters
        ----------
        curve : DiscountCurve
            Baseline discount curve.
        bump : float
            Rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        float
            DV01 in currency units per 1bp.
        """
        df   = curve.discount(self._T)
        r    = -np.log(df) / self._T if self._T > 0 else 0.0
        r_up = r + bump
        fwd_up = self._spot * np.exp((r_up - self._div_yield) * self._T)
        df_up  = np.exp(-r_up * self._T)
        npv_up = self._sign * self._notional_ * (fwd_up - self._strike) * df_up
        return npv_up - self.price(curve)

    def duration(self, curve: DiscountCurve) -> float:
        """
        Time to maturity in years — natural duration analogue for forwards.
        Modified duration in the bond sense is not defined for equity forwards.
        """
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        """
        Single net settlement cash flow at maturity.
        Amount shown as the forward-implied net: sign · notional · (F - K).
        Actual settlement depends on S_T, which is unknown at pricing time.
        """
        return pd.DataFrame([{
            'date':   self._maturity_date.ISO(),
            'amount': self._sign * self._notional_ * (self._spot - self._strike),
            'type':   'net settlement (indicative at spot)',
        }])

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        """
        IR sensitivity assigned to the nearest tenor bucket to maturity.

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
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {
            t: (dv01_total if i == nearest_idx else 0.0)
            for i, t in enumerate(tenors)
        }

    # ── forward-specific methods ──────────────────────────────────────────────

    def forward_price(self, curve: DiscountCurve) -> float:
        """
        Fair forward price: F = S · exp((r - q) · T).

        r is extracted from the OIS curve as the continuously compounded
        zero rate to maturity. q is the continuous dividend yield.

        Returns
        -------
        float
            Forward price of the underlying.
        """
        r = curve.zero_rate(self._T)    # decimal (e.g. 0.025 for 2.5%)
        return self._spot * np.exp((r - self._div_yield) * self._T)

    def describe(self) -> str:
        side = 'Long' if self._long else 'Short'
        return (
            f"EquityForward | {side} | {self._currency_} | "
            f"Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"Maturity={self._maturity_date.ISO()} | "
            f"DivYield={self._div_yield*100:.2f}% | "
            f"Notional={self._notional_:,.0f}"
        )
