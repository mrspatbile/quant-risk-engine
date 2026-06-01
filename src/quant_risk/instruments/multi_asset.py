"""
Multi-asset exotic options — WorstOfOption, BestOfOption.

Priced via bivariate GBM Monte Carlo with Cholesky decomposition of the
correlation matrix.

Payoff convention : performance-based (returns), common in structured notes.
    WorstOfOption Call : max(min(S1_T/S1_0, S2_T/S2_0) - K, 0) × notional
    WorstOfOption Put  : max(K - min(S1_T/S1_0, S2_T/S2_0), 0) × notional
    BestOfOption Call  : max(max(S1_T/S1_0, S2_T/S2_0) - K, 0) × notional
    BestOfOption Put   : max(K - max(S1_T/S1_0, S2_T/S2_0), 0) × notional

K = 1.0 → at-the-money (receive the worst/best return if positive/negative).

Construction
------------
Both classes expose two factory methods:

    from_rho(...)    — rho already known, pass directly
    from_prices(...) — estimate rho from historical price series via
                       realized_correlation() in utils/calibration.py

Rate convention : decimal throughout (0.025 = 2.5%).
Dates           : ql.Date at public boundaries.
Notional        : currency units, no implicit scaling.
"""

import numpy as np
import pandas as pd
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument
from quant_risk.utils.calibration import realized_correlation


# ── shared bivariate GBM simulation ──────────────────────────────────────────

def _simulate_2asset(
    S1_0: float, S2_0: float,
    r: float,
    q1: float, q2: float,
    sigma1: float, sigma2: float,
    rho: float,
    T: float,
    n_paths: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate terminal prices for two correlated GBM assets under risk-neutral measure.

    Uses Cholesky decomposition: W2 = rho·W1 + sqrt(1-rho²)·Z where W1, Z iid N(0,1).

    Returns
    -------
    (S1_T, S2_T) : tuple of np.ndarray, shape (n_paths,)
    """
    rng  = np.random.default_rng(seed)
    Z1   = rng.standard_normal(n_paths)
    Z2   = rng.standard_normal(n_paths)
    W1   = Z1
    W2   = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    S1_T = S1_0 * np.exp((r - q1 - 0.5 * sigma1**2) * T + sigma1 * np.sqrt(T) * W1)
    S2_T = S2_0 * np.exp((r - q2 - 0.5 * sigma2**2) * T + sigma2 * np.sqrt(T) * W2)
    return S1_T, S2_T


def _extract_r(curve: DiscountCurve, T: float) -> float:
    df = curve.discount(T)
    return -np.log(df) / T if T > 0 else 0.0


# ── WorstOfOption ─────────────────────────────────────────────────────────────

class WorstOfOption(Instrument):
    """
    Worst-of option on two underlyings — pays based on the worst performer.

    Payoff (call): max(min(S1_T/S1_0, S2_T/S2_0) − K, 0) × notional
    Payoff (put) : max(K − min(S1_T/S1_0, S2_T/S2_0), 0) × notional

    K = 1.0 → at-the-money (the instrument settles at the worst return).
    Higher correlation reduces the value of the worst-of (less diversification benefit).

    Priced via bivariate GBM Monte Carlo.

    Do not construct directly — use the factory methods:
        WorstOfOption.from_rho(...)    — pass rho directly
        WorstOfOption.from_prices(...) — estimate rho from historical prices
    """

    def __init__(
        self,
        spot1: float,
        spot2: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma1: float,
        sigma2: float,
        rho: float,
        option_type: str = 'put',
        div_yield1: float = 0.0,
        div_yield2: float = 0.0,
        notional_: float = 1.0,
        n_paths: int = 50_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot1          = spot1
        self._spot2          = spot2
        self._strike         = strike
        self._expiry_date    = expiry_date
        self._valuation_date = valuation_date
        self._sigma1         = sigma1
        self._sigma2         = sigma2
        self._rho            = rho
        self._option_type    = option_type.lower()
        self._div_yield1     = div_yield1
        self._div_yield2     = div_yield2
        self._notional_      = notional_
        self._n_paths        = n_paths
        self._seed           = seed
        self._currency_      = currency_
        self._day_count      = ql.Actual365Fixed()
        self._T              = self._day_count.yearFraction(valuation_date, expiry_date)
        self._validate()

    def _validate(self) -> None:
        if not -1.0 <= self._rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self._rho}")
        if self._sigma1 <= 0 or self._sigma2 <= 0:
            raise ValueError("sigma1 and sigma2 must be positive")
        if self._option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self._option_type}'")

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_rho(
        cls,
        spot1: float,
        spot2: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma1: float,
        sigma2: float,
        rho: float,
        option_type: str = 'put',
        div_yield1: float = 0.0,
        div_yield2: float = 0.0,
        notional_: float = 1.0,
        n_paths: int = 50_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ) -> 'WorstOfOption':
        """
        Construct with a known correlation.

        Parameters
        ----------
        rho : float
            Correlation between the two underlyings, in (-1, 1).
            Can be estimated from historical data via realized_correlation().
        """
        return cls(
            spot1, spot2, strike, expiry_date, valuation_date,
            sigma1, sigma2, rho, option_type,
            div_yield1, div_yield2, notional_, n_paths, seed, currency_,
        )

    @classmethod
    def from_prices(
        cls,
        spot1: float,
        spot2: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma1: float,
        sigma2: float,
        prices1: pd.Series,
        prices2: pd.Series,
        window: int | None = None,
        option_type: str = 'put',
        div_yield1: float = 0.0,
        div_yield2: float = 0.0,
        notional_: float = 1.0,
        n_paths: int = 50_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ) -> 'WorstOfOption':
        """
        Construct by estimating correlation from historical price series.

        Computes log-return correlation via realized_correlation() and
        delegates to from_rho(). The estimated rho is stored on the instance
        as self.rho for inspection.

        Parameters
        ----------
        prices1 : pd.Series
            Historical price series for asset 1 (e.g. from ExternalStore).
        prices2 : pd.Series
            Historical price series for asset 2.
        window : int or None
            Rolling window in trading days. None = full common history.
        """
        rho = realized_correlation(prices1, prices2, window=window)
        return cls(
            spot1, spot2, strike, expiry_date, valuation_date,
            sigma1, sigma2, rho, option_type,
            div_yield1, div_yield2, notional_, n_paths, seed, currency_,
        )

    # ── pricing ───────────────────────────────────────────────────────────────

    def _mc_price(self, r: float) -> float:
        S1_T, S2_T = _simulate_2asset(
            self._spot1, self._spot2, r,
            self._div_yield1, self._div_yield2,
            self._sigma1, self._sigma2, self._rho,
            self._T, self._n_paths, self._seed,
        )
        perf1 = S1_T / self._spot1
        perf2 = S2_T / self._spot2
        worst = np.minimum(perf1, perf2)

        if self._option_type == 'call':
            payoff = np.maximum(worst - self._strike, 0.0)
        else:
            payoff = np.maximum(self._strike - worst, 0.0)

        return float(np.exp(-r * self._T) * payoff.mean() * self._notional_)

    # ── Instrument ABC ────────────────────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    @property
    def rho(self) -> float:
        """Correlation used in pricing (estimated or supplied directly)."""
        return self._rho

    def price(self, curve: DiscountCurve) -> float:
        """Worst-of option NPV in currency units via bivariate GBM MC."""
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
            'type':   f'worst-of {self._option_type} payoff (contingent on performance)',
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
            f"WorstOfOption | {side} | {self._currency_} | "
            f"Spot1={self._spot1:.2f} | Spot2={self._spot2:.2f} | "
            f"Strike={self._strike:.2f} | ρ={self._rho:.2f} | "
            f"σ1={self._sigma1*100:.1f}% | σ2={self._sigma2*100:.1f}% | "
            f"Expiry={self._expiry_date.ISO()}"
        )


# ── BestOfOption ──────────────────────────────────────────────────────────────

class BestOfOption(Instrument):
    """
    Best-of option on two underlyings — pays based on the best performer.

    Payoff (call): max(max(S1_T/S1_0, S2_T/S2_0) − K, 0) × notional
    Payoff (put) : max(K − max(S1_T/S1_0, S2_T/S2_0), 0) × notional

    Higher correlation reduces diversification and lowers the best-of value.

    Priced via bivariate GBM Monte Carlo.

    Do not construct directly — use the factory methods:
        BestOfOption.from_rho(...)    — pass rho directly
        BestOfOption.from_prices(...) — estimate rho from historical prices
    """

    def __init__(
        self,
        spot1: float,
        spot2: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma1: float,
        sigma2: float,
        rho: float,
        option_type: str = 'call',
        div_yield1: float = 0.0,
        div_yield2: float = 0.0,
        notional_: float = 1.0,
        n_paths: int = 50_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ):
        self._spot1          = spot1
        self._spot2          = spot2
        self._strike         = strike
        self._expiry_date    = expiry_date
        self._valuation_date = valuation_date
        self._sigma1         = sigma1
        self._sigma2         = sigma2
        self._rho            = rho
        self._option_type    = option_type.lower()
        self._div_yield1     = div_yield1
        self._div_yield2     = div_yield2
        self._notional_      = notional_
        self._n_paths        = n_paths
        self._seed           = seed
        self._currency_      = currency_
        self._day_count      = ql.Actual365Fixed()
        self._T              = self._day_count.yearFraction(valuation_date, expiry_date)
        self._validate()

    def _validate(self) -> None:
        if not -1.0 <= self._rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self._rho}")
        if self._sigma1 <= 0 or self._sigma2 <= 0:
            raise ValueError("sigma1 and sigma2 must be positive")
        if self._option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put', got '{self._option_type}'")

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def from_rho(
        cls,
        spot1: float,
        spot2: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma1: float,
        sigma2: float,
        rho: float,
        option_type: str = 'call',
        div_yield1: float = 0.0,
        div_yield2: float = 0.0,
        notional_: float = 1.0,
        n_paths: int = 50_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ) -> 'BestOfOption':
        """Construct with a known correlation."""
        return cls(
            spot1, spot2, strike, expiry_date, valuation_date,
            sigma1, sigma2, rho, option_type,
            div_yield1, div_yield2, notional_, n_paths, seed, currency_,
        )

    @classmethod
    def from_prices(
        cls,
        spot1: float,
        spot2: float,
        strike: float,
        expiry_date: ql.Date,
        valuation_date: ql.Date,
        sigma1: float,
        sigma2: float,
        prices1: pd.Series,
        prices2: pd.Series,
        window: int | None = None,
        option_type: str = 'call',
        div_yield1: float = 0.0,
        div_yield2: float = 0.0,
        notional_: float = 1.0,
        n_paths: int = 50_000,
        seed: int = 0,
        currency_: str = 'EUR',
    ) -> 'BestOfOption':
        """Construct by estimating correlation from historical price series."""
        rho = realized_correlation(prices1, prices2, window=window)
        return cls(
            spot1, spot2, strike, expiry_date, valuation_date,
            sigma1, sigma2, rho, option_type,
            div_yield1, div_yield2, notional_, n_paths, seed, currency_,
        )

    # ── pricing ───────────────────────────────────────────────────────────────

    def _mc_price(self, r: float) -> float:
        S1_T, S2_T = _simulate_2asset(
            self._spot1, self._spot2, r,
            self._div_yield1, self._div_yield2,
            self._sigma1, self._sigma2, self._rho,
            self._T, self._n_paths, self._seed,
        )
        perf1 = S1_T / self._spot1
        perf2 = S2_T / self._spot2
        best  = np.maximum(perf1, perf2)

        if self._option_type == 'call':
            payoff = np.maximum(best - self._strike, 0.0)
        else:
            payoff = np.maximum(self._strike - best, 0.0)

        return float(np.exp(-r * self._T) * payoff.mean() * self._notional_)

    # ── Instrument ABC ────────────────────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    @property
    def rho(self) -> float:
        """Correlation used in pricing (estimated or supplied directly)."""
        return self._rho

    def price(self, curve: DiscountCurve) -> float:
        """Best-of option NPV in currency units via bivariate GBM MC."""
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
            'type':   f'best-of {self._option_type} payoff (contingent on performance)',
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
            f"BestOfOption | {side} | {self._currency_} | "
            f"Spot1={self._spot1:.2f} | Spot2={self._spot2:.2f} | "
            f"Strike={self._strike:.2f} | ρ={self._rho:.2f} | "
            f"σ1={self._sigma1*100:.1f}% | σ2={self._sigma2*100:.1f}% | "
            f"Expiry={self._expiry_date.ISO()}"
        )
