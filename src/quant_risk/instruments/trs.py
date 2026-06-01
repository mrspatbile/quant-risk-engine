"""
TotalReturnSwap — equity total return swap.

The asset receiver:
    — receives the total return on the reference asset (price + dividends)
    — pays OIS + funding spread on the notional

Pricing model
-------------
Return leg NPV:
    PV of receiving the asset at maturity, in units of notional:
        notional × (S_current × exp(-q × T) / S_inception − df(T))

    where:
        S_current   — reference asset price today
        S_inception — reference asset price at trade inception (the "strike")
        q           — continuous dividend yield
        T           — remaining life in years
        df(T)       — OIS discount factor to maturity

    At inception (S_current = S_inception, q = 0):
        return_leg_npv ≈ notional × (1 − df(T))

Funding leg NPV:
    Sum of OIS + spread coupons at each payment date, discounted:
        notional × Σᵢ (r_i + spread) × Δtᵢ × df(tᵢ)

    where r_i is the continuously compounded zero rate to payment date tᵢ.

NPV (to asset receiver):
    return_leg_npv − funding_leg_npv

Rate convention : decimal throughout. funding_spread_bps is converted to
                  decimal internally (÷ 10_000).
Dates           : ISO strings at public boundaries; ql.Date internally.
Notional        : currency units, no implicit scaling.
"""

import numpy as np
import pandas as pd
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


class TotalReturnSwap(Instrument):
    """
    Equity total return swap (TRS).

    The asset receiver receives the total return on the reference asset
    (price appreciation + continuous dividend yield) and pays OIS + spread.

    Parameters
    ----------
    notional_ : float
        Notional in currency units.
    spot_inception : float
        Reference asset price at trade inception — acts as the "strike".
    spot_current : float
        Current reference asset price used for mark-to-market. At inception,
        set equal to spot_inception (NPV ≈ 0 with zero spread).
    valuation_date : str
        Pricing date as ISO string 'YYYY-MM-DD'.
    maturity_date : str
        Trade maturity as ISO string 'YYYY-MM-DD'.
    funding_spread_bps : float
        Spread over OIS paid by the asset receiver on the funding leg, in bps.
        Positive = receiver pays above OIS. Negative = receiver funded below OIS.
    div_yield : float
        Continuous dividend yield of the reference asset, decimal. Default 0.
    long_asset : bool
        True  = asset receiver (receive return, pay funding). Default True.
        False = asset payer (pay return, receive funding).
    payment_freq : int
        Funding leg payment frequency per year: 1=annual, 2=semi, 4=quarterly.
        Default 4.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    """

    _day_count = ql.Actual365Fixed()

    def __init__(
        self,
        notional_:          float,
        spot_inception:     float,
        spot_current:       float,
        valuation_date:     str,
        maturity_date:      str,
        funding_spread_bps: float,
        div_yield:          float = 0.0,
        long_asset:         bool  = True,
        payment_freq:       int   = 4,
        currency_:          str   = 'EUR',
    ):
        self._notional_          = notional_
        self._spot_inception     = spot_inception
        self._spot_current       = spot_current
        self._valuation_date_str = valuation_date
        self._maturity_date_str  = maturity_date
        self._funding_spread     = funding_spread_bps / 10_000
        self._div_yield          = div_yield
        self._long_asset         = long_asset
        self._payment_freq       = payment_freq
        self._currency_          = currency_
        self._sign               = 1.0 if long_asset else -1.0

        self._val_date = self._parse_date(valuation_date)
        self._mat_date = self._parse_date(maturity_date)
        self._T = self._day_count.yearFraction(self._val_date, self._mat_date)

        self._validate()

    def _validate(self) -> None:
        if self._T <= 0:
            raise ValueError("maturity_date must be after valuation_date")
        if self._spot_inception <= 0:
            raise ValueError(f"spot_inception must be positive, got {self._spot_inception}")
        if self._spot_current <= 0:
            raise ValueError(f"spot_current must be positive, got {self._spot_current}")
        if self._payment_freq not in (1, 2, 4, 12):
            raise ValueError(f"payment_freq must be 1, 2, 4 or 12, got {self._payment_freq}")

    # ── internal schedule ─────────────────────────────────────────────────────

    def _payment_times(self) -> list[float]:
        """
        List of time-in-years from valuation date to each funding coupon date.
        The final payment coincides with maturity.
        """
        months_per_period = 12 // self._payment_freq
        times  = []
        cursor = self._val_date
        while True:
            cursor = cursor + ql.Period(months_per_period, ql.Months)
            t = self._day_count.yearFraction(self._val_date, cursor)
            if t >= self._T - 1e-6:
                times.append(self._T)
                break
            times.append(t)
        return times

    # ── pricing ───────────────────────────────────────────────────────────────

    def _compute(self, curve: DiscountCurve) -> dict:
        df_T = curve.discount(self._T)

        # Return leg: PV of receiving asset total return to maturity
        return_leg_npv = self._notional_ * (
            self._spot_current * np.exp(-self._div_yield * self._T) / self._spot_inception
            - df_T
        )

        # Funding leg: periodic OIS + spread coupons
        payment_times  = self._payment_times()
        funding_leg_npv = 0.0
        t_prev = 0.0
        for t_i in payment_times:
            dt   = t_i - t_prev
            df_i = curve.discount(t_i)
            r_i  = -np.log(df_i) / t_i if t_i > 0 else 0.0
            coupon = self._notional_ * (r_i + self._funding_spread) * dt
            funding_leg_npv += coupon * df_i
            t_prev = t_i

        npv = self._sign * (return_leg_npv - funding_leg_npv)
        return {
            'return_leg_npv':  return_leg_npv,
            'funding_leg_npv': funding_leg_npv,
            'npv':             npv,
        }

    def price(self, curve: DiscountCurve) -> dict:
        """
        Full TRS valuation.

        Parameters
        ----------
        curve : DiscountCurve
            OIS discount curve.

        Returns
        -------
        dict with keys:
            return_leg_npv  : PV of asset total return leg (always from asset receiver view)
            funding_leg_npv : PV of OIS + spread funding payments (always from asset receiver view)
            npv             : net NPV to the holder (positive = asset receiver profits)
        """
        return self._compute(curve)

    def npv(self, curve: DiscountCurve) -> float:
        """Net NPV in currency units."""
        return self.price(curve)['npv']

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        Change in NPV for a 1bp parallel shift in the OIS curve.

        Asset receivers have negative DV01 on the funding leg (higher rates
        increase the floating coupon cost) and positive DV01 on the return
        leg (higher rates raise the forward asset price). Net sign depends
        on the relative sizes.

        Parameters
        ----------
        curve : DiscountCurve
            Baseline discount curve.
        bump : float
            Rate bump in decimal. Default 0.0001 (1bp).
        """
        # Compute analytically: extract flat rate, bump, recompute both legs
        df_T = curve.discount(self._T)
        r    = -np.log(df_T) / self._T if self._T > 0 else 0.0

        # Build a shifted version of each discount factor using a flat bump
        class _BumpedCurve:
            """Thin wrapper that shifts all discount factors by bump."""
            def __init__(self, base, bump):
                self._base = base
                self._bump = bump
            def discount(self, T):
                df = self._base.discount(T)
                r  = -np.log(df) / T if T > 0 else 0.0
                return np.exp(-(r + self._bump) * T)

        bumped = _BumpedCurve(curve, bump)
        return self._compute(bumped)['npv'] - self._compute(curve)['npv']

    def duration(self, curve: DiscountCurve) -> float:
        """
        Weighted average maturity of the funding leg cash flows.
        Approximates the interest rate sensitivity duration.
        """
        payment_times = self._payment_times()
        t_prev = 0.0
        total_pv = 0.0
        weighted = 0.0
        for t_i in payment_times:
            dt     = t_i - t_prev
            df_i   = curve.discount(t_i)
            r_i    = -np.log(df_i) / t_i if t_i > 0 else 0.0
            pv     = (r_i + self._funding_spread) * dt * df_i
            weighted += t_i * pv
            total_pv += pv
            t_prev = t_i
        return weighted / total_pv if total_pv > 0 else self._T

    def cash_flows(self) -> pd.DataFrame:
        """
        Expected funding leg coupon schedule based on current OIS rates.
        Return leg cash flow shown as a single indicative amount at maturity.
        """
        rows = []
        # Return leg — single indicative at maturity
        indicative_return = (
            self._spot_current / self._spot_inception - 1.0 + self._div_yield * self._T
        ) * self._notional_
        rows.append({
            'date':   self._mat_date.ISO(),
            'amount': indicative_return,
            'type':   'return leg (indicative at current spot)',
        })
        # Funding leg — schedule without curve dependency for simplicity
        payment_times = self._payment_times()
        t_prev = 0.0
        for t_i in payment_times:
            dt = t_i - t_prev
            rows.append({
                'date':   (self._val_date + ql.Period(int(t_i * 365), ql.Days)).ISO(),
                'amount': -self._notional_ * self._funding_spread * dt,
                'type':   'funding spread coupon (OIS spread component)',
            })
            t_prev = t_i
        return pd.DataFrame(rows)

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        """
        IR sensitivity concentrated at the nearest tenor to maturity.

        TRS rate sensitivity arises primarily from the funding leg discount
        and the forward asset price, both dominated by the maturity tenor.

        Parameters
        ----------
        curve : DiscountCurve
            Baseline discount curve.
        tenors : list[float]
            Vertex maturities in years.
        bump : float
            Rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        dict[float, float]
            {tenor_years: sensitivity} in currency units per 1bp.
        """
        dv01_total  = self.dv01(curve, bump)
        nearest_idx = min(range(len(tenors)), key=lambda i: abs(tenors[i] - self._T))
        return {t: (dv01_total if i == nearest_idx else 0.0) for i, t in enumerate(tenors)}

    # ── Instrument ABC ────────────────────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._currency_

    @property
    def notional(self) -> float:
        return self._notional_

    def describe(self) -> str:
        side = 'Asset Receiver' if self._long_asset else 'Asset Payer'
        return (
            f"TotalReturnSwap | {side} | {self._currency_} | "
            f"Notional={self._notional_:,.0f} | "
            f"SpotInception={self._spot_inception:.2f} | "
            f"SpotCurrent={self._spot_current:.2f} | "
            f"Spread={self._funding_spread * 10_000:.0f}bps | "
            f"Maturity={self._maturity_date_str} | "
            f"DivYield={self._div_yield * 100:.2f}%"
        )
