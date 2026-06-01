"""
Closed-form exotic options — DigitalOption, BarrierOption, ChooserOption, CompoundOption.

All priced analytically via QuantLib engines. No simulation required.

Rate convention : decimal throughout (0.025 = 2.5%).
Dates           : ql.Date at public boundaries.
Notional        : currency units, no implicit scaling.
"""

import numpy as np
import pandas as pd
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


# ── shared process builder ────────────────────────────────────────────────────

def _bsm_process(
    spot: float,
    sigma: float,
    div_yield: float,
    disc_handle: ql.YieldTermStructureHandle,
    valuation_date: ql.Date,
) -> ql.BlackScholesMertonProcess:
    day_count  = ql.Actual365Fixed()
    spot_h     = ql.QuoteHandle(ql.SimpleQuote(spot))
    vol_ts     = ql.BlackConstantVol(
        valuation_date, ql.NullCalendar(),
        ql.QuoteHandle(ql.SimpleQuote(sigma)), day_count,
    )
    vol_h      = ql.BlackVolTermStructureHandle(vol_ts)
    div_ts     = ql.FlatForward(valuation_date, div_yield, day_count)
    div_h      = ql.YieldTermStructureHandle(div_ts)
    return ql.BlackScholesMertonProcess(spot_h, div_h, disc_handle, vol_h)


def _flat_disc_handle(rate: float, valuation_date: ql.Date) -> ql.YieldTermStructureHandle:
    """Flat discount curve at rate (decimal)."""
    ts = ql.FlatForward(valuation_date, rate, ql.Actual365Fixed())
    return ql.YieldTermStructureHandle(ts)


# ── DigitalOption ─────────────────────────────────────────────────────────────

class DigitalOption(Instrument):
    """
    European digital option — Cash-or-Nothing or Asset-or-Nothing.

    Cash-or-Nothing: pays cash_amount if in-the-money at expiry.
    Asset-or-Nothing: delivers the underlying if in-the-money at expiry.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Barrier level for the binary payoff.
    expiry_date : ql.Date
        Option expiry date.
    valuation_date : ql.Date
        Pricing date. Must match QuantLib global evaluation date.
    sigma : float
        Implied volatility, decimal (e.g. 0.20 = 20%).
    payoff_type : str
        'cash' — pays cash_amount if ITM.
        'asset' — delivers the underlying if ITM.
    cash_amount : float
        Cash payment when payoff_type='cash'. Default 1.0.
    option_type : str
        'call' or 'put'. Default 'call'.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    _VALID_PAYOFFS = frozenset({'cash', 'asset'})

    def __init__(
        self,
        spot: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        payoff_type: str = 'cash',
        cash_amount: float = 1.0,
        option_type: str = 'call',
        notional_: float = 1.0,
        div_yield: float = 0.0,
        currency_: str = 'EUR',
    ):
        self._spot           = spot
        self._strike         = strike
        self._expiry_date    = expiry_date
        self._valuation_date = valuation_date
        self._sigma          = sigma
        self._payoff_type    = payoff_type.lower()
        self._cash_amount    = cash_amount
        self._option_type    = option_type.lower()
        self._notional_      = notional_
        self._div_yield      = div_yield
        self._currency_      = currency_
        self._day_count      = ql.Actual365Fixed()
        self._T              = self._day_count.yearFraction(valuation_date, expiry_date)
        self._ql_type        = ql.Option.Call if self._option_type == 'call' else ql.Option.Put
        self._validate()

    def _validate(self) -> None:
        if self._payoff_type not in self._VALID_PAYOFFS:
            raise ValueError(f"payoff_type must be 'cash' or 'asset', got '{self._payoff_type}'")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if self._spot <= 0:
            raise ValueError(f"spot must be positive, got {self._spot}")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        if self._payoff_type == 'cash':
            payoff = ql.CashOrNothingPayoff(self._ql_type, self._strike, self._cash_amount)
        else:
            payoff = ql.AssetOrNothingPayoff(self._ql_type, self._strike)
        exercise = ql.EuropeanExercise(self._expiry_date)
        option   = ql.VanillaOption(payoff, exercise)
        process  = _bsm_process(self._spot, self._sigma, self._div_yield,
                                 disc_handle, self._valuation_date)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Digital option NPV in currency units."""
        disc_handle = self._disc_handle_from_curve(curve)
        return self._with_eval_date(self._valuation_date,
                                    lambda: self._ql_npv(disc_handle))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        df   = curve.discount(self._T)
        r    = -np.log(df) / self._T if self._T > 0 else 0.0
        h_up = _flat_disc_handle(r + bump, self._valuation_date)
        h_base = self._disc_handle_from_curve(curve)
        return self._with_eval_date(
            self._valuation_date,
            lambda: self._ql_npv(h_up) - self._ql_npv(h_base),
        )

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        if self._payoff_type == 'cash':
            amount = self._cash_amount * self._notional_
        else:
            amount = self._spot * self._notional_
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': amount,
            'type':   f'digital {self._payoff_type}-or-nothing (contingent)',
        }])

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0)
                for i, t in enumerate(tenors)}

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"DigitalOption | {side} | {self._payoff_type}-or-nothing | {self._currency_} | "
            f"Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"Expiry={self._expiry_date.ISO()} | Vol={self._sigma*100:.1f}%"
        )


# ── BarrierOption ─────────────────────────────────────────────────────────────

_BARRIER_TYPE_MAP = {
    'DownOut': ql.Barrier.DownOut,
    'DownIn':  ql.Barrier.DownIn,
    'UpOut':   ql.Barrier.UpOut,
    'UpIn':    ql.Barrier.UpIn,
}


class BarrierOption(Instrument):
    """
    European single-barrier option — knock-out or knock-in.

    Priced via QuantLib AnalyticBarrierEngine (Haug / Reiner-Rubinstein).

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Option strike.
    barrier : float
        Barrier level. For Down barriers: barrier < spot.
                        For Up barriers: barrier > spot.
    rebate : float
        Cash paid if the barrier is hit (knock-out) or never touched (knock-in).
        Set to 0 for standard barrier options.
    expiry_date : ql.Date
        Option expiry.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    barrier_type : str
        'DownOut' | 'DownIn' | 'UpOut' | 'UpIn'.
    option_type : str
        'call' or 'put'. Default 'call'.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        barrier: float,
        rebate: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        barrier_type: str,
        option_type: str = 'call',
        notional_: float = 1.0,
        div_yield: float = 0.0,
        currency_: str = 'EUR',
    ):
        self._spot           = spot
        self._strike         = strike
        self._barrier        = barrier
        self._rebate         = rebate
        self._expiry_date    = expiry_date
        self._valuation_date = valuation_date
        self._sigma          = sigma
        self._barrier_type   = barrier_type
        self._option_type    = option_type.lower()
        self._notional_      = notional_
        self._div_yield      = div_yield
        self._currency_      = currency_
        self._day_count      = ql.Actual365Fixed()
        self._T              = self._day_count.yearFraction(valuation_date, expiry_date)
        self._ql_type        = ql.Option.Call if self._option_type == 'call' else ql.Option.Put
        self._ql_barrier     = _BARRIER_TYPE_MAP[barrier_type]
        self._validate()

    def _validate(self) -> None:
        if self._barrier_type not in _BARRIER_TYPE_MAP:
            raise ValueError(
                f"barrier_type must be one of {list(_BARRIER_TYPE_MAP)}, "
                f"got '{self._barrier_type}'"
            )
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if self._spot <= 0:
            raise ValueError(f"spot must be positive, got {self._spot}")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        payoff   = ql.PlainVanillaPayoff(self._ql_type, self._strike)
        exercise = ql.EuropeanExercise(self._expiry_date)
        option   = ql.BarrierOption(self._ql_barrier, self._barrier, self._rebate,
                                     payoff, exercise)
        process  = _bsm_process(self._spot, self._sigma, self._div_yield,
                                 disc_handle, self._valuation_date)
        option.setPricingEngine(ql.AnalyticBarrierEngine(process))
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Barrier option NPV in currency units."""
        disc_handle = self._disc_handle_from_curve(curve)
        return self._with_eval_date(self._valuation_date,
                                    lambda: self._ql_npv(disc_handle))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        df   = curve.discount(self._T)
        r    = -np.log(df) / self._T if self._T > 0 else 0.0
        h_up = _flat_disc_handle(r + bump, self._valuation_date)
        h_base = self._disc_handle_from_curve(curve)
        return self._with_eval_date(
            self._valuation_date,
            lambda: self._ql_npv(h_up) - self._ql_npv(h_base),
        )

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        intrinsic = max(self._spot - self._strike, 0) if self._option_type == 'call' \
                    else max(self._strike - self._spot, 0)
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': intrinsic * self._notional_,
            'type':   f'barrier {self._barrier_type} payoff (intrinsic, contingent)',
        }])

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0)
                for i, t in enumerate(tenors)}

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"BarrierOption | {self._barrier_type} {side} | {self._currency_} | "
            f"Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"Barrier={self._barrier:.2f} | Expiry={self._expiry_date.ISO()} | "
            f"Vol={self._sigma*100:.1f}%"
        )


# ── ChooserOption ─────────────────────────────────────────────────────────────

class ChooserOption(Instrument):
    """
    Simple chooser option — holder selects call or put at choice_date.

    Both the call and put share the same strike and expiry.
    Value >= max(call, put) at choice_date by put-call parity.
    Priced via QuantLib AnalyticSimpleChooserEngine.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Common strike for both the call and put.
    choice_date : ql.Date
        Date on which the holder chooses call or put. Must precede expiry_date.
    expiry_date : ql.Date
        Expiry of the underlying call or put.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        choice_date: ql.Date,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        notional_: float = 1.0,
        div_yield: float = 0.0,
        currency_: str = 'EUR',
    ):
        self._spot           = spot
        self._strike         = strike
        self._choice_date    = choice_date
        self._expiry_date    = expiry_date
        self._valuation_date = valuation_date
        self._sigma          = sigma
        self._notional_      = notional_
        self._div_yield      = div_yield
        self._currency_      = currency_
        self._day_count      = ql.Actual365Fixed()
        self._T              = self._day_count.yearFraction(valuation_date, expiry_date)
        self._validate()

    def _validate(self) -> None:
        if self._choice_date >= self._expiry_date:
            raise ValueError("choice_date must be before expiry_date")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if self._spot <= 0:
            raise ValueError(f"spot must be positive, got {self._spot}")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        exercise = ql.EuropeanExercise(self._expiry_date)
        option   = ql.SimpleChooserOption(self._choice_date, self._strike, exercise)
        process  = _bsm_process(self._spot, self._sigma, self._div_yield,
                                 disc_handle, self._valuation_date)
        option.setPricingEngine(ql.AnalyticSimpleChooserEngine(process))
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """
        Chooser option NPV in currency units.

        Uses a flat discount curve extracted from the OIS zero rate to expiry.
        AnalyticSimpleChooserEngine requires all term structures to share the
        same day count — a flat FlatForward(Actual365Fixed) satisfies this.
        """
        df = curve.discount(self._T)
        r  = -np.log(df) / self._T if self._T > 0 else 0.0
        disc_handle = _flat_disc_handle(r, self._valuation_date)
        return self._with_eval_date(self._valuation_date,
                                    lambda: self._ql_npv(disc_handle))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        df     = curve.discount(self._T)
        r      = -np.log(df) / self._T if self._T > 0 else 0.0
        h_base = _flat_disc_handle(r, self._valuation_date)
        h_up   = _flat_disc_handle(r + bump, self._valuation_date)
        return self._with_eval_date(
            self._valuation_date,
            lambda: self._ql_npv(h_up) - self._ql_npv(h_base),
        )

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': max(self._spot - self._strike, 0) * self._notional_,
            'type':   'chooser payoff (intrinsic call, contingent)',
        }])

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0)
                for i, t in enumerate(tenors)}

    def describe(self) -> str:
        return (
            f"ChooserOption | {self._currency_} | "
            f"Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"ChoiceDate={self._choice_date.ISO()} | Expiry={self._expiry_date.ISO()} | "
            f"Vol={self._sigma*100:.1f}%"
        )


# ── CompoundOption ────────────────────────────────────────────────────────────

_OPTION_TYPE_MAP = {'call': ql.Option.Call, 'put': ql.Option.Put}


class CompoundOption(Instrument):
    """
    Compound option — option on an option.

    The outer option gives the right to enter the inner (underlying) option
    at outer expiry by paying strike_outer.

    Priced via QuantLib AnalyticCompoundOptionEngine (Geske 1979).

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike_outer : float
        Premium paid at expiry_outer to enter the inner option.
    strike_inner : float
        Strike of the underlying vanilla option.
    expiry_outer : ql.Date
        Expiry of the compound option. Must precede expiry_inner.
    expiry_inner : ql.Date
        Expiry of the underlying option.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    outer_type : str
        Type of the compound option: 'call' | 'put'.
    inner_type : str
        Type of the underlying option: 'call' | 'put'.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        strike_outer: float,
        strike_inner: float,
        expiry_outer: ql.Date,
        expiry_inner: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        outer_type: str,
        inner_type: str,
        notional_: float = 1.0,
        div_yield: float = 0.0,
        currency_: str = 'EUR',
    ):
        self._spot           = spot
        self._strike_outer   = strike_outer
        self._strike_inner   = strike_inner
        self._expiry_outer   = expiry_outer
        self._expiry_inner   = expiry_inner
        self._valuation_date = valuation_date
        self._sigma          = sigma
        self._outer_type     = outer_type.lower()
        self._inner_type     = inner_type.lower()
        self._notional_      = notional_
        self._div_yield      = div_yield
        self._currency_      = currency_
        self._day_count      = ql.Actual365Fixed()
        self._T              = self._day_count.yearFraction(valuation_date, expiry_inner)
        self._validate()

    def _validate(self) -> None:
        if self._expiry_outer >= self._expiry_inner:
            raise ValueError("expiry_outer must be before expiry_inner")
        for name, val in [('outer_type', self._outer_type), ('inner_type', self._inner_type)]:
            if val not in _OPTION_TYPE_MAP:
                raise ValueError(f"{name} must be 'call' or 'put', got '{val}'")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if self._spot <= 0:
            raise ValueError(f"spot must be positive, got {self._spot}")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        outer_payoff = ql.PlainVanillaPayoff(
            _OPTION_TYPE_MAP[self._outer_type], self._strike_outer
        )
        inner_payoff = ql.PlainVanillaPayoff(
            _OPTION_TYPE_MAP[self._inner_type], self._strike_inner
        )
        outer_ex = ql.EuropeanExercise(self._expiry_outer)
        inner_ex = ql.EuropeanExercise(self._expiry_inner)
        option   = ql.CompoundOption(outer_payoff, outer_ex, inner_payoff, inner_ex)
        process  = _bsm_process(self._spot, self._sigma, self._div_yield,
                                 disc_handle, self._valuation_date)
        option.setPricingEngine(ql.AnalyticCompoundOptionEngine(process))
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """
        Compound option NPV in currency units.

        Uses a flat discount curve extracted from the OIS zero rate to inner expiry.
        AnalyticCompoundOptionEngine requires all term structures to share the
        same day count — a flat FlatForward(Actual365Fixed) satisfies this.
        """
        df = curve.discount(self._T)
        r  = -np.log(df) / self._T if self._T > 0 else 0.0
        disc_handle = _flat_disc_handle(r, self._valuation_date)
        return self._with_eval_date(self._valuation_date,
                                    lambda: self._ql_npv(disc_handle))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        df     = curve.discount(self._T)
        r      = -np.log(df) / self._T if self._T > 0 else 0.0
        h_base = _flat_disc_handle(r, self._valuation_date)
        h_up   = _flat_disc_handle(r + bump, self._valuation_date)
        return self._with_eval_date(
            self._valuation_date,
            lambda: self._ql_npv(h_up) - self._ql_npv(h_base),
        )

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                'date':   self._expiry_outer.ISO(),
                'amount': -self._strike_outer * self._notional_,
                'type':   'outer premium (contingent)',
            },
            {
                'date':   self._expiry_inner.ISO(),
                'amount': max(self._spot - self._strike_inner, 0) * self._notional_,
                'type':   'inner payoff (intrinsic, contingent)',
            },
        ])

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0)
                for i, t in enumerate(tenors)}

    def describe(self) -> str:
        return (
            f"CompoundOption | {self._outer_type.capitalize()} on "
            f"{self._inner_type.capitalize()} | {self._currency_} | "
            f"Spot={self._spot:.2f} | K_inner={self._strike_inner:.2f} | "
            f"K_outer={self._strike_outer:.2f} | "
            f"Expiry_outer={self._expiry_outer.ISO()} | "
            f"Expiry_inner={self._expiry_inner.ISO()} | "
            f"Vol={self._sigma*100:.1f}%"
        )
