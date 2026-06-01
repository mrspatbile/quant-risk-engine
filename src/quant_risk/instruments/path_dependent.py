"""
Path-dependent single-asset exotic options.

QuantLib-backed (analytic or built-in MC engines):
    AsianOption     — discrete averaging, geometric (analytic) or arithmetic (MC)
    LookbackOption  — continuous monitoring, fixed or floating strike (analytic)
    CliquetOption   — sum of forward-start options (analytic)

Custom GBM Monte Carlo (no QuantLib native pricer):
    ShoutOption     — holder locks in intrinsic value once
    NapoleonOption  — sum of worst periodic returns
    Accumulator     — forced buy at discount with knock-out
    Decumulator     — forced sell at premium with knock-out

Rate convention : decimal throughout (0.025 = 2.5%).
Dates           : ql.Date at public boundaries.
Notional        : currency units, no implicit scaling.
"""

import numpy as np
import pandas as pd
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


# ── shared helpers ────────────────────────────────────────────────────────────

def _bsm_process(
    spot: float,
    sigma: float,
    div_yield: float,
    disc_handle: ql.YieldTermStructureHandle,
    valuation_date: ql.Date,
) -> ql.BlackScholesMertonProcess:
    day_count = ql.Actual365Fixed()
    spot_h    = ql.QuoteHandle(ql.SimpleQuote(spot))
    vol_ts    = ql.BlackConstantVol(
        valuation_date, ql.NullCalendar(),
        ql.QuoteHandle(ql.SimpleQuote(sigma)), day_count,
    )
    vol_h  = ql.BlackVolTermStructureHandle(vol_ts)
    div_ts = ql.FlatForward(valuation_date, div_yield, day_count)
    div_h  = ql.YieldTermStructureHandle(div_ts)
    return ql.BlackScholesMertonProcess(spot_h, div_h, disc_handle, vol_h)


def _flat_disc_handle(rate: float, valuation_date: ql.Date) -> ql.YieldTermStructureHandle:
    ts = ql.FlatForward(valuation_date, rate, ql.Actual365Fixed())
    return ql.YieldTermStructureHandle(ts)


def _simulate_gbm(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    GBM paths under risk-neutral measure.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1)
        paths[:, 0] = S0; paths[:, -1] = S_T.
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    Z   = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def _extract_r(curve: DiscountCurve, T: float) -> float:
    """Continuously compounded zero rate from discount factor."""
    df = curve.discount(T)
    return -np.log(df) / T if T > 0 else 0.0


# ── AsianOption ───────────────────────────────────────────────────────────────

class AsianOption(Instrument):
    """
    Discrete averaging Asian option.

    Average Price (fixed strike): payoff on average of spot vs strike.
    Average Strike (floating strike): payoff on final spot vs average of spot.

    Geometric average → QuantLib analytic engine.
    Arithmetic average → QuantLib MC engine.

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Option strike. For average strike, this parameter is not used in payoff
        computation — the engine substitutes the realised average.
    expiry_date : ql.Date
        Option expiry.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    average_type : str
        'arithmetic' or 'geometric'. Default 'arithmetic'.
    strike_type : str
        'fixed' (average of spot vs K) or 'floating' (spot vs average). Default 'fixed'.
    fixing_dates : list[ql.Date] or None
        Observation dates. Defaults to monthly from valuation to expiry.
    option_type : str
        'call' or 'put'. Default 'call'.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    n_paths : int
        MC paths for arithmetic average. Default 10_000.
    seed : int
        MC seed passed to QuantLib's pseudo-random engine.
        seed=0 means non-deterministic (time-based). Use a positive integer
        for reproducible results.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        average_type: str = 'arithmetic',
        strike_type: str = 'fixed',
        fixing_dates: list | None = None,
        option_type: str = 'call',
        notional_: float = 1.0,
        div_yield: float = 0.0,
        n_paths: int = 10_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot          = spot
        self._strike        = strike
        self._expiry_date   = expiry_date
        self._valuation_date = valuation_date
        self._sigma         = sigma
        self._average_type  = average_type.lower()
        self._strike_type   = strike_type.lower()
        self._option_type   = option_type.lower()
        self._notional_     = notional_
        self._div_yield     = div_yield
        self._n_paths       = n_paths
        self._seed          = seed
        self._currency_     = currency_
        self._day_count     = ql.Actual365Fixed()
        self._T             = self._day_count.yearFraction(valuation_date, expiry_date)
        self._ql_type       = ql.Option.Call if self._option_type == 'call' else ql.Option.Put
        self._fixing_dates  = fixing_dates or self._default_fixings()
        self._validate()

    def _default_fixings(self) -> list:
        T_days = int(self._T * 365)
        n_months = max(1, T_days // 30)
        return [self._valuation_date + ql.Period(i, ql.Months)
                for i in range(1, n_months + 1)
                if self._valuation_date + ql.Period(i, ql.Months) <= self._expiry_date]

    def _validate(self) -> None:
        if self._average_type not in ('arithmetic', 'geometric'):
            raise ValueError(f"average_type must be 'arithmetic' or 'geometric', got '{self._average_type}'")
        if self._strike_type not in ('fixed', 'floating'):
            raise ValueError(f"strike_type must be 'fixed' or 'floating', got '{self._strike_type}'")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        payoff   = ql.PlainVanillaPayoff(self._ql_type, self._strike)
        exercise = ql.EuropeanExercise(self._expiry_date)
        avg_type = ql.Average.Geometric if self._average_type == 'geometric' else ql.Average.Arithmetic
        option   = ql.DiscreteAveragingAsianOption(avg_type, 0.0, 0, self._fixing_dates, payoff, exercise)
        process  = _bsm_process(self._spot, self._sigma, self._div_yield,
                                 disc_handle, self._valuation_date)
        if self._average_type == 'geometric' and self._strike_type == 'fixed':
            option.setPricingEngine(
                ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(process)
            )
        elif self._strike_type == 'floating':
            option.setPricingEngine(
                ql.MCDiscreteArithmeticASEngine(
                    process, 'pseudorandom', requiredSamples=self._n_paths, seed=self._seed
                )
            )
        else:
            option.setPricingEngine(
                ql.MCDiscreteArithmeticAPEngine(
                    process, 'pseudorandom', requiredSamples=self._n_paths, seed=self._seed
                )
            )
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Asian option NPV in currency units."""
        r  = _extract_r(curve, self._T)
        dh = _flat_disc_handle(r, self._valuation_date)
        return self._with_eval_date(self._valuation_date, lambda: self._ql_npv(dh))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r      = _extract_r(curve, self._T)
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
            'type':   f'asian {self._average_type} {self._strike_type}-strike payoff (indicative)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"AsianOption | {side} | {self._average_type} {self._strike_type}-strike | "
            f"{self._currency_} | Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"Expiry={self._expiry_date.ISO()} | Vol={self._sigma*100:.1f}% | "
            f"Fixings={len(self._fixing_dates)}"
        )


# ── LookbackOption ────────────────────────────────────────────────────────────

class LookbackOption(Instrument):
    """
    Continuous monitoring lookback option.

    Fixed strike: payoff on realised extremum vs strike.
        Call: max(S_max - K, 0). Put: max(K - S_min, 0).
    Floating strike: payoff on final spot vs realised extremum.
        Call: S_T - S_min. Put: S_max - S_T.

    Priced via QuantLib analytic engines (Goldman-Sosin-Gatto).

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float or None
        Strike for fixed-strike lookback. Pass None for floating strike.
    expiry_date : ql.Date
        Option expiry.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    strike_type : str
        'fixed' or 'floating'. Default 'fixed'.
    option_type : str
        'call' or 'put'. Default 'call'.
    min_max : float or None
        Running minimum (for call) or maximum (for put) observed so far.
        At inception pass None — defaults to spot (no historical observations).
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
        strike: float | None,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        strike_type: str = 'fixed',
        option_type: str = 'call',
        min_max: float | None = None,
        notional_: float = 1.0,
        div_yield: float = 0.0,
        currency_: str = 'EUR',
    ):
        self._spot          = spot
        self._strike        = strike
        self._expiry_date   = expiry_date
        self._valuation_date = valuation_date
        self._sigma         = sigma
        self._strike_type   = strike_type.lower()
        self._option_type   = option_type.lower()
        self._min_max       = min_max if min_max is not None else spot
        self._notional_     = notional_
        self._div_yield     = div_yield
        self._currency_     = currency_
        self._day_count     = ql.Actual365Fixed()
        self._T             = self._day_count.yearFraction(valuation_date, expiry_date)
        self._ql_type       = ql.Option.Call if self._option_type == 'call' else ql.Option.Put
        self._validate()

    def _validate(self) -> None:
        if self._strike_type not in ('fixed', 'floating'):
            raise ValueError(f"strike_type must be 'fixed' or 'floating', got '{self._strike_type}'")
        if self._strike_type == 'fixed' and self._strike is None:
            raise ValueError("strike is required for fixed-strike lookback")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        process = _bsm_process(self._spot, self._sigma, self._div_yield,
                                disc_handle, self._valuation_date)
        exercise = ql.EuropeanExercise(self._expiry_date)
        if self._strike_type == 'fixed':
            payoff = ql.PlainVanillaPayoff(self._ql_type, self._strike)
            option = ql.ContinuousFixedLookbackOption(self._min_max, payoff, exercise)
            option.setPricingEngine(ql.AnalyticContinuousFixedLookbackEngine(process))
        else:
            payoff = ql.FloatingTypePayoff(self._ql_type)
            option = ql.ContinuousFloatingLookbackOption(self._min_max, payoff, exercise)
            option.setPricingEngine(ql.AnalyticContinuousFloatingLookbackEngine(process))
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Lookback option NPV in currency units."""
        r  = _extract_r(curve, self._T)
        dh = _flat_disc_handle(r, self._valuation_date)
        return self._with_eval_date(self._valuation_date, lambda: self._ql_npv(dh))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r      = _extract_r(curve, self._T)
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
            'amount': self._spot * self._notional_,
            'type':   f'lookback {self._strike_type}-strike payoff (indicative at spot)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"LookbackOption | {side} | {self._strike_type}-strike | {self._currency_} | "
            f"Spot={self._spot:.2f} | Expiry={self._expiry_date.ISO()} | "
            f"Vol={self._sigma*100:.1f}%"
        )


# ── CliquetOption ─────────────────────────────────────────────────────────────

class CliquetOption(Instrument):
    """
    Cliquet option (ratchet) — sum of periodic forward-start returns.

    Each period's payoff is max(S_ti/S_{ti-1} - 1, 0) for a call.
    The QuantLib AnalyticCliquetEngine prices this as the sum of forward-start
    options using Black's formula for each period.

    Parameters
    ----------
    spot : float
        Current underlying price.
    reset_dates : list[ql.Date]
        Period reset dates (observation schedule). Must all precede expiry_date.
    expiry_date : ql.Date
        Final settlement date.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
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
        reset_dates: list,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        option_type: str = 'call',
        notional_: float = 1.0,
        div_yield: float = 0.0,
        currency_: str = 'EUR',
    ):
        self._spot          = spot
        self._reset_dates   = reset_dates
        self._expiry_date   = expiry_date
        self._valuation_date = valuation_date
        self._sigma         = sigma
        self._option_type   = option_type.lower()
        self._notional_     = notional_
        self._div_yield     = div_yield
        self._currency_     = currency_
        self._day_count     = ql.Actual365Fixed()
        self._T             = self._day_count.yearFraction(valuation_date, expiry_date)
        self._ql_type       = ql.Option.Call if self._option_type == 'call' else ql.Option.Put
        self._validate()

    def _validate(self) -> None:
        if any(d >= self._expiry_date for d in self._reset_dates):
            raise ValueError("All reset_dates must be strictly before expiry_date")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if not self._reset_dates:
            raise ValueError("reset_dates must not be empty")

    def _ql_npv(self, disc_handle: ql.YieldTermStructureHandle) -> float:
        payoff   = ql.PercentageStrikePayoff(self._ql_type, 1.0)
        exercise = ql.EuropeanExercise(self._expiry_date)
        option   = ql.CliquetOption(payoff, exercise, self._reset_dates)
        process  = _bsm_process(self._spot, self._sigma, self._div_yield,
                                 disc_handle, self._valuation_date)
        option.setPricingEngine(ql.AnalyticCliquetEngine(process))
        return option.NPV() * self._notional_

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Cliquet option NPV in currency units."""
        r  = _extract_r(curve, self._T)
        dh = _flat_disc_handle(r, self._valuation_date)
        return self._with_eval_date(self._valuation_date, lambda: self._ql_npv(dh))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r      = _extract_r(curve, self._T)
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
            'amount': 0.0,
            'type':   'cliquet aggregate payoff (contingent on period returns)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"CliquetOption | {side} | {self._currency_} | "
            f"Spot={self._spot:.2f} | Periods={len(self._reset_dates)} | "
            f"Expiry={self._expiry_date.ISO()} | Vol={self._sigma*100:.1f}%"
        )


# ── ShoutOption ───────────────────────────────────────────────────────────────

class ShoutOption(Instrument):
    """
    Shout option — holder may lock in intrinsic value once during the option's life.

    At shout time t*: floor = S_{t*} - K is locked in.
    At expiry: payoff = max(S_T - K, S_{t*} - K, 0).
    The holder shouts optimally (when current intrinsic is maximised in expectation).

    Priced via GBM Monte Carlo with optimal shout policy:
    shout when current continuation value < intrinsic value (Longstaff-Schwartz
    approximation using a simple boundary: shout at the first time the intrinsic
    value exceeds the continuation value estimated as a European call from that node).

    Parameters
    ----------
    spot : float
        Current underlying price.
    strike : float
        Option strike.
    expiry_date : ql.Date
        Option expiry.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    option_type : str
        'call' or 'put'. Default 'call'.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    n_paths : int
        Number of MC paths. Default 10_000.
    n_steps : int
        Time steps per path. Default 252.
    seed : int
        Random seed. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        option_type: str = 'call',
        notional_: float = 1.0,
        div_yield: float = 0.0,
        n_paths: int = 10_000,
        n_steps: int = 252,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot          = spot
        self._strike        = strike
        self._expiry_date   = expiry_date
        self._valuation_date = valuation_date
        self._sigma         = sigma
        self._option_type   = option_type.lower()
        self._notional_     = notional_
        self._div_yield     = div_yield
        self._n_paths       = n_paths
        self._n_steps       = n_steps
        self._seed          = seed
        self._currency_     = currency_
        self._day_count     = ql.Actual365Fixed()
        self._T             = self._day_count.yearFraction(valuation_date, expiry_date)
        self._validate()

    def _validate(self) -> None:
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if self._spot <= 0:
            raise ValueError(f"spot must be positive, got {self._spot}")

    def _mc_price(self, r: float) -> float:
        from scipy.stats import norm
        paths = _simulate_gbm(
            self._spot, r, self._div_yield, self._sigma,
            self._T, self._n_steps, self._n_paths, self._seed,
        )
        K   = self._strike
        dt  = self._T / self._n_steps
        sign = 1 if self._option_type == 'call' else -1

        # For each path, find the optimal shout time by dynamic programming.
        # Continuation value at each step approximated as BSM call/put value
        # from that node to expiry. Shout when intrinsic >= continuation value.
        payoffs = np.zeros(self._n_paths)
        for path_idx in range(self._n_paths):
            shout_floor = 0.0
            for step in range(self._n_steps):
                S_t     = paths[path_idx, step]
                intrinsic = max(sign * (S_t - K), 0.0)
                t_remain  = self._T - step * dt
                if t_remain <= 0:
                    break
                # BSM continuation value from current node
                d1 = (np.log(S_t / K) + (r - self._div_yield + 0.5 * self._sigma**2) * t_remain) \
                     / (self._sigma * np.sqrt(t_remain))
                d2 = d1 - self._sigma * np.sqrt(t_remain)
                if self._option_type == 'call':
                    cont = (S_t * np.exp(-self._div_yield * t_remain) * norm.cdf(d1)
                            - K * np.exp(-r * t_remain) * norm.cdf(d2))
                else:
                    cont = (K * np.exp(-r * t_remain) * norm.cdf(-d2)
                            - S_t * np.exp(-self._div_yield * t_remain) * norm.cdf(-d1))
                if intrinsic >= cont and shout_floor == 0.0 and intrinsic > 0:
                    shout_floor = intrinsic
            S_T     = paths[path_idx, -1]
            final   = max(sign * (S_T - K), 0.0)
            payoffs[path_idx] = max(final, shout_floor)

        return float(np.exp(-r * self._T) * payoffs.mean() * self._notional_)

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Shout option NPV in currency units via MC."""
        return self._mc_price(_extract_r(curve, self._T))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r = _extract_r(curve, self._T)
        return self._mc_price(r + bump) - self._mc_price(r)

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': max(self._spot - self._strike, 0) * self._notional_,
            'type':   'shout payoff (indicative intrinsic)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"ShoutOption | {side} | {self._currency_} | "
            f"Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"Expiry={self._expiry_date.ISO()} | Vol={self._sigma*100:.1f}% | "
            f"Paths={self._n_paths}"
        )


# ── NapoleonOption ────────────────────────────────────────────────────────────

class NapoleonOption(Instrument):
    """
    Napoleon option — payoff is the sum of worst periodic log-returns.

    At each observation date, the period return r_i = S_ti/S_{ti-1} - 1.
    Payoff = notional × sum(min(r_i)) across all periods.
    Typically used as a yield enhancement note component.

    Priced via GBM Monte Carlo.

    Parameters
    ----------
    spot : float
        Current underlying price.
    observation_dates : list[ql.Date]
        Period end dates (observation schedule).
    expiry_date : ql.Date
        Final settlement date (typically last observation date).
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    notional_ : float
        Contract multiplier. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    n_paths : int
        Number of MC paths. Default 10_000.
    seed : int
        Random seed. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        observation_dates: list,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        notional_: float = 1.0,
        div_yield: float = 0.0,
        n_paths: int = 10_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot              = spot
        self._observation_dates = observation_dates
        self._expiry_date       = expiry_date
        self._valuation_date    = valuation_date
        self._sigma             = sigma
        self._notional_         = notional_
        self._div_yield         = div_yield
        self._n_paths           = n_paths
        self._seed              = seed
        self._currency_         = currency_
        self._day_count         = ql.Actual365Fixed()
        self._T                 = self._day_count.yearFraction(valuation_date, expiry_date)
        self._obs_times         = [
            self._day_count.yearFraction(valuation_date, d)
            for d in observation_dates
        ]
        self._validate()

    def _validate(self) -> None:
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")
        if not self._observation_dates:
            raise ValueError("observation_dates must not be empty")

    def _mc_price(self, r: float) -> float:
        n_periods = len(self._obs_times)
        obs_steps = [max(1, int(t * self._n_steps_per_year)) for t in self._obs_times]
        n_steps   = max(obs_steps)
        paths = _simulate_gbm(
            self._spot, r, self._div_yield, self._sigma,
            self._T, n_steps, self._n_paths, self._seed,
        )
        # Extract spot at each observation time
        obs_spots = np.column_stack([
            paths[:, min(s, paths.shape[1] - 1)] for s in obs_steps
        ])
        # Period returns
        prev = np.column_stack([np.full(self._n_paths, self._spot), obs_spots[:, :-1]])
        returns = obs_spots / prev - 1.0
        worst_returns = returns.min(axis=1)
        payoff = np.maximum(worst_returns, 0.0)
        return float(np.exp(-r * self._T) * payoff.mean() * self._notional_)

    _n_steps_per_year = 252

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Napoleon option NPV in currency units via MC."""
        return self._mc_price(_extract_r(curve, self._T))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r = _extract_r(curve, self._T)
        return self._mc_price(r + bump) - self._mc_price(r)

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': 0.0,
            'type':   'napoleon payoff (sum of worst period returns, contingent)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        return (
            f"NapoleonOption | {self._currency_} | "
            f"Spot={self._spot:.2f} | Periods={len(self._observation_dates)} | "
            f"Expiry={self._expiry_date.ISO()} | Vol={self._sigma*100:.1f}%"
        )


# ── Accumulator ───────────────────────────────────────────────────────────────

class Accumulator(Instrument):
    """
    Accumulator — holder is obligated to buy shares at forward_price each period
    while spot stays below barrier. Knock-out terminates the contract early.

    This is a leveraged structured product with significant tail risk:
    if spot falls far below forward_price the holder is forced to accumulate
    at a loss with no exit until knock-out or expiry.

    Priced via GBM Monte Carlo: at each observation date, if spot > barrier
    the contract knocks out (no more purchases). Otherwise the holder buys
    shares_per_period at forward_price, accumulating gain/loss vs spot.

    Parameters
    ----------
    spot : float
        Current underlying price.
    forward_price : float
        Contracted purchase price, typically set below spot at inception.
    barrier : float
        Knock-out level above spot. Contract terminates if spot > barrier.
    observation_dates : list[ql.Date]
        Periodic observation dates for purchases and knock-out check.
    expiry_date : ql.Date
        Final settlement date.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    shares_per_period : float
        Number of shares bought at each observation. Default 1.0.
    notional_ : float
        Total notional scaling. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    n_paths : int
        MC paths. Default 10_000.
    n_steps : int
        Steps per path. Default 252.
    seed : int
        Random seed. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        forward_price: float,
        barrier: float,
        observation_dates: list,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        shares_per_period: float = 1.0,
        notional_: float = 1.0,
        div_yield: float = 0.0,
        n_paths: int = 10_000,
        n_steps: int = 252,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot              = spot
        self._forward_price     = forward_price
        self._barrier           = barrier
        self._observation_dates = observation_dates
        self._expiry_date       = expiry_date
        self._valuation_date    = valuation_date
        self._sigma             = sigma
        self._shares_per_period = shares_per_period
        self._notional_         = notional_
        self._div_yield         = div_yield
        self._n_paths           = n_paths
        self._n_steps           = n_steps
        self._seed              = seed
        self._currency_         = currency_
        self._day_count         = ql.Actual365Fixed()
        self._T                 = self._day_count.yearFraction(valuation_date, expiry_date)
        self._obs_times         = [
            self._day_count.yearFraction(valuation_date, d) for d in observation_dates
        ]
        self._validate()

    def _validate(self) -> None:
        if self._barrier <= self._spot:
            raise ValueError("barrier must be above spot for Accumulator")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")

    def _mc_price(self, r: float) -> float:
        paths = _simulate_gbm(
            self._spot, r, self._div_yield, self._sigma,
            self._T, self._n_steps, self._n_paths, self._seed,
        )
        dt = self._T / self._n_steps
        payoffs = np.zeros(self._n_paths)

        for path_idx in range(self._n_paths):
            knocked_out = False
            for obs_t in self._obs_times:
                if knocked_out:
                    break
                step = min(int(obs_t / dt), self._n_steps)
                S_t  = paths[path_idx, step]
                if S_t > self._barrier:
                    knocked_out = True
                else:
                    # Buy at forward_price, mark to market at S_t
                    payoffs[path_idx] += (S_t - self._forward_price) * self._shares_per_period

        disc_payoffs = np.exp(-r * self._T) * payoffs
        return float(disc_payoffs.mean() * self._notional_)

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Accumulator NPV in currency units via MC."""
        return self._mc_price(_extract_r(curve, self._T))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r = _extract_r(curve, self._T)
        return self._mc_price(r + bump) - self._mc_price(r)

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': (self._spot - self._forward_price) * self._shares_per_period
                      * len(self._observation_dates) * self._notional_,
            'type':   'accumulator net settlement (indicative at current spot)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        return (
            f"Accumulator | {self._currency_} | "
            f"Spot={self._spot:.2f} | ForwardPrice={self._forward_price:.2f} | "
            f"Barrier={self._barrier:.2f} | Periods={len(self._observation_dates)} | "
            f"Vol={self._sigma*100:.1f}%"
        )


# ── Decumulator ───────────────────────────────────────────────────────────────

class Decumulator(Instrument):
    """
    Decumulator — holder is obligated to sell shares at forward_price each period
    while spot stays above barrier. Knock-out terminates the contract early.

    Mirror of Accumulator: the holder sells at a premium to spot at inception
    but misses upside if spot rallies sharply, and knocks out if spot falls
    below barrier.

    Parameters
    ----------
    spot : float
        Current underlying price.
    forward_price : float
        Contracted sale price, typically set above spot at inception.
    barrier : float
        Knock-out level below spot. Contract terminates if spot < barrier.
    observation_dates : list[ql.Date]
        Periodic observation dates.
    expiry_date : ql.Date
        Final settlement date.
    valuation_date : ql.Date
        Pricing date.
    sigma : float
        Implied volatility, decimal.
    shares_per_period : float
        Number of shares sold at each observation. Default 1.0.
    notional_ : float
        Total notional scaling. Default 1.0.
    div_yield : float
        Continuous dividend yield, decimal. Default 0.
    n_paths : int
        MC paths. Default 10_000.
    n_steps : int
        Steps per path. Default 252.
    seed : int
        Random seed. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    def __init__(
        self,
        spot: float,
        forward_price: float,
        barrier: float,
        observation_dates: list,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma: float,
        shares_per_period: float = 1.0,
        notional_: float = 1.0,
        div_yield: float = 0.0,
        n_paths: int = 10_000,
        n_steps: int = 252,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot              = spot
        self._forward_price     = forward_price
        self._barrier           = barrier
        self._observation_dates = observation_dates
        self._expiry_date       = expiry_date
        self._valuation_date    = valuation_date
        self._sigma             = sigma
        self._shares_per_period = shares_per_period
        self._notional_         = notional_
        self._div_yield         = div_yield
        self._n_paths           = n_paths
        self._n_steps           = n_steps
        self._seed              = seed
        self._currency_         = currency_
        self._day_count         = ql.Actual365Fixed()
        self._T                 = self._day_count.yearFraction(valuation_date, expiry_date)
        self._obs_times         = [
            self._day_count.yearFraction(valuation_date, d) for d in observation_dates
        ]
        self._validate()

    def _validate(self) -> None:
        if self._barrier >= self._spot:
            raise ValueError("barrier must be below spot for Decumulator")
        if self._sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self._sigma}")

    def _mc_price(self, r: float) -> float:
        paths = _simulate_gbm(
            self._spot, r, self._div_yield, self._sigma,
            self._T, self._n_steps, self._n_paths, self._seed,
        )
        dt = self._T / self._n_steps
        payoffs = np.zeros(self._n_paths)

        for path_idx in range(self._n_paths):
            knocked_out = False
            for obs_t in self._obs_times:
                if knocked_out:
                    break
                step = min(int(obs_t / dt), self._n_steps)
                S_t  = paths[path_idx, step]
                if S_t < self._barrier:
                    knocked_out = True
                else:
                    # Sell at forward_price, mark to market vs S_t
                    payoffs[path_idx] += (self._forward_price - S_t) * self._shares_per_period

        disc_payoffs = np.exp(-r * self._T) * payoffs
        return float(disc_payoffs.mean() * self._notional_)

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """Decumulator NPV in currency units via MC."""
        return self._mc_price(_extract_r(curve, self._T))

    def npv(self, curve: DiscountCurve) -> float:
        return self.price(curve)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        r = _extract_r(curve, self._T)
        return self._mc_price(r + bump) - self._mc_price(r)

    def duration(self, curve: DiscountCurve) -> float:
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': (self._forward_price - self._spot) * self._shares_per_period
                      * len(self._observation_dates) * self._notional_,
            'type':   'decumulator net settlement (indicative at current spot)',
        }])

    def rate_sensitivities(
        self, curve: DiscountCurve, tenors: list[float], bump: float = 0.0001,
    ) -> dict[float, float]:
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    def describe(self) -> str:
        return (
            f"Decumulator | {self._currency_} | "
            f"Spot={self._spot:.2f} | ForwardPrice={self._forward_price:.2f} | "
            f"Barrier={self._barrier:.2f} | Periods={len(self._observation_dates)} | "
            f"Vol={self._sigma*100:.1f}%"
        )
