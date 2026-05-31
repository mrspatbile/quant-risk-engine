"""
VanillaOption -- European option pricer via Black-Scholes-Merton.

Uses QuantLib AnalyticEuropeanEngine for pricing and Greeks.
Implements the Instrument ABC -- price(), dv01(), duration(), cash_flows(),
key_rate_dv01().

Regulatory context:
    FRTB SA -- delta (equity/FX/GIRR), vega, curvature (CRR3 Article 325)
    IFRS 13 -- Level 2 (vanilla with observable vol), Level 3 (exotic)
    EMIR -- OTC options reporting and margining
"""

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


class VanillaOption(Instrument):
    """
    European vanilla option pricer -- Black-Scholes-Merton via QuantLib.

    Parameters
    ----------
    spot : float
        Underlying spot price.
    strike : float
        Option strike price.
    expiry_date : ql.Date
        Option expiry date.
    valuation_date : ql.Date
        QuantLib valuation date -- must match global clock.
    sigma : float
        Implied volatility (annualised, decimal). E.g. 0.165 for 16.5%.
    option_type : str
        'call' or 'put'.
    notional : float
        Number of options / contract multiplier. Default 1.
    div_yield : float
        Continuous dividend yield of the underlying. Default 0.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    underlying : str
        Underlying identifier for display. Default 'equity'.
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
        currency_: str = 'EUR',
        underlying: str = 'equity',
    ):
        self._spot          = spot
        self._strike        = strike
        self._expiry_date   = expiry_date
        self._valuation_date = valuation_date
        self._sigma         = sigma
        self._option_type   = option_type.lower()
        self._notional_     = notional_
        self._div_yield     = div_yield
        self._currency_     = currency_
        self._underlying    = underlying
        self._day_count     = ql.Actual365Fixed()
        self._calendar      = ql.TARGET()

        self._T = self._day_count.yearFraction(valuation_date, expiry_date)
        self._ql_type = ql.Option.Call if self._option_type == 'call' else ql.Option.Put

    # ── internal BSM analytics ────────────────────────────────────────────────

    def _d1_d2(self, r: float) -> tuple:
        S, K, T, sig, q = self._spot, self._strike, self._T, self._sigma, self._div_yield
        if T <= 0:
            return np.nan, np.nan
        d1 = (np.log(S / K) + (r - q + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        return d1, d2

    def _bsm_price(self, r: float) -> float:
        S, K, T, sig, q = self._spot, self._strike, self._T, self._sigma, self._div_yield
        if T <= 0:
            if self._option_type == 'call':
                return max(S - K, 0.0)
            return max(K - S, 0.0)
        d1, d2 = self._d1_d2(r)
        if self._option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def _ql_engine(self, disc_handle: ql.YieldTermStructureHandle) -> ql.AnalyticEuropeanEngine:
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self._spot))
        flat_vol    = ql.BlackConstantVol(
            self._valuation_date, self._calendar,
            ql.QuoteHandle(ql.SimpleQuote(self._sigma)),
            self._day_count
        )
        vol_handle  = ql.BlackVolTermStructureHandle(flat_vol)
        div_ts      = ql.FlatForward(self._valuation_date, self._div_yield, self._day_count)
        div_handle  = ql.YieldTermStructureHandle(div_ts)
        process     = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, disc_handle, vol_handle
        )
        return ql.AnalyticEuropeanEngine(process)

    def _ql_option(self, disc_handle: ql.YieldTermStructureHandle) -> ql.VanillaOption:
        payoff   = ql.PlainVanillaPayoff(self._ql_type, self._strike)
        exercise = ql.EuropeanExercise(self._expiry_date)
        option   = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(self._ql_engine(disc_handle))
        return option

    # ── Instrument ABC implementation ─────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def price(self, curve: DiscountCurve) -> float:
        """
        Option NPV via QuantLib AnalyticEuropeanEngine.

        Returns option price per unit × notional.
        Positive for long option positions.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        r           = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0
        return self._bsm_price(r) * self._notional_

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        Rho -- change in option price for 1bp shift in risk-free rate.

        For calls: positive rho (higher rates increase call value).
        For puts: negative rho.
        FRTB GIRR sensitivity for rate-sensitive options.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        r_base = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0
        npv_base = self._bsm_price(r_base) * self._notional_
        npv_up   = self._bsm_price(r_base + bump) * self._notional_
        return npv_up - npv_base

    def duration(self, curve: DiscountCurve) -> float:
        """
        Time to expiry in years -- natural duration equivalent for options.
        Options have no modified duration in the bond sense.
        """
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        """
        Single contingent cash flow at expiry -- payoff if in the money.
        Intrinsic value at current spot shown as indicative amount.
        """
        intrinsic = max(self._spot - self._strike, 0) if self._option_type == 'call' \
                    else max(self._strike - self._spot, 0)
        return pd.DataFrame([{
            'date':   self._expiry_date.ISO(),
            'amount': intrinsic * self._notional_,
            'type':   'payoff (intrinsic)',
        }])

    def key_rate_dv01(
        self,
        curve: DiscountCurve,
        tenors: list,
        bump: float = 0.0001,
    ) -> dict:
        """
        Rho assigned to the nearest of the caller-supplied vertices.

        Option IR sensitivity concentrates at the expiry tenor.
        All key rate DV01s are zero except at the nearest bucket.

        Parameters
        ----------
        tenors : list
            Vertex maturities in years, e.g. [0.25, 0.5, 1, 2, 5, 10].
        """
        rho_total   = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {
            (f"{int(t*12)}M" if t < 1 else f"{int(t)}Y"): (rho_total if i == nearest_idx else 0.0)
            for i, t in enumerate(tenors)
        }


    # ── option-specific methods ───────────────────────────────────────────────

    def delta(self, curve: DiscountCurve) -> float:
        """
        Delta -- change in option price for 1 unit move in underlying.

        Call delta: [0, 1]. Put delta: [-1, 0].
        ATM: approximately ±0.5.
        FRTB SA equity/FX delta sensitivity.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        r           = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0
        d1, _       = self._d1_d2(r)
        if self._option_type == 'call':
            return np.exp(-self._div_yield * self._T) * norm.cdf(d1) * self._notional_
        return -np.exp(-self._div_yield * self._T) * norm.cdf(-d1) * self._notional_

    def gamma(self, curve: DiscountCurve) -> float:
        """
        Gamma -- rate of change of delta per unit move in underlying.

        Always positive for long options.
        Key input for FRTB curvature risk.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        r           = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0
        d1, _       = self._d1_d2(r)
        return (np.exp(-self._div_yield * self._T) * norm.pdf(d1)
                / (self._spot * self._sigma * np.sqrt(self._T)) * self._notional_)

    def vega(self, curve: DiscountCurve) -> float:
        """
        Vega -- change in option price per 1% move in implied vol.

        Always positive for long options.
        FRTB SA vega risk sensitivity.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        r           = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0
        d1, _       = self._d1_d2(r)
        return (self._spot * np.exp(-self._div_yield * self._T)
                * norm.pdf(d1) * np.sqrt(self._T) / 100 * self._notional_)

    def theta(self, curve: DiscountCurve) -> float:
        """
        Theta -- change in option price per calendar day.

        Negative for long options -- time decay.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        r           = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0
        d1, d2      = self._d1_d2(r)
        S, K, T, sig, q = self._spot, self._strike, self._T, self._sigma, self._div_yield

        common = -S * np.exp(-q * T) * norm.pdf(d1) * sig / (2 * np.sqrt(T))
        if self._option_type == 'call':
            return (common - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365 * self._notional_
        return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)
                - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365 * self._notional_

    def implied_vol(self, market_price: float, r: float = 0.025) -> float:
        def objective(sigma):
            S, K, T, q = self._spot, self._strike, self._T, self._div_yield
            d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            if self._option_type == 'call':
                return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - market_price
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1) - market_price

        try:
            return brentq(objective, 0.001, 5.0, xtol=1e-8)
        except ValueError:
            return np.nan

    def frtb_curvature(self, curve: DiscountCurve) -> dict:
        """
        FRTB SA curvature risk -- scenario-based gamma capture.

        CVR = -min(V(S+RW) - V - RW*delta, V(S-RW) - V + RW*delta, 0)

        Risk weight for equity: 15%.
        """
        rw = 0.15   # equity delta risk weight
        disc_handle = self._disc_handle_from_curve(curve)
        r = -np.log(disc_handle.discount(self._expiry_date)) / self._T if self._T > 0 else 0

        V      = self._bsm_price(r) * self._notional_
        delta_ = self.delta(curve)

        # shocked spot prices
        S_up   = self._spot * (1 + rw)
        S_down = self._spot * (1 - rw)

        # reprice at shocked spots
        orig_spot    = self._spot
        self._spot   = S_up
        V_up         = self._bsm_price(r) * self._notional_
        self._spot   = S_down
        V_down       = self._bsm_price(r) * self._notional_
        self._spot   = orig_spot   # restore

        cvr_up   = V_up   - V - rw * orig_spot * delta_
        cvr_down = V_down - V + rw * orig_spot * delta_
        cvr      = -min(cvr_up, cvr_down, 0)

        return {
            'V':        V,
            'V_up':     V_up,
            'V_down':   V_down,
            'cvr_up':   cvr_up,
            'cvr_down': cvr_down,
            'cvr':      cvr,
        }

    def greeks_summary(self, curve: DiscountCurve) -> dict:
        """All Greeks in one call."""
        return {
            'price': self.price(curve),
            'delta': self.delta(curve),
            'gamma': self.gamma(curve),
            'vega':  self.vega(curve),
            'theta': self.theta(curve),
            'rho':   self.dv01(curve),
        }

    def describe(self) -> str:
        side = 'Call' if self._option_type == 'call' else 'Put'
        return (
            f"VanillaOption | {side} | {self._currency_} | "
            f"Spot={self._spot:.2f} | Strike={self._strike:.2f} | "
            f"Expiry={self._expiry_date.ISO()} | "
            f"Vol={self._sigma*100:.1f}% | Notional={self._notional_:,.0f}"
        )
