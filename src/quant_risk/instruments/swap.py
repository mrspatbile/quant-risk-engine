# src/quant_risk/instruments/swap.py

"""
Interest rate swap pricer -- multi-curve OIS/EURIBOR via QuantLib.

Implements the post-2008 multi-curve framework:
- OIS curve (ESTR) -- discounting
- EURIBOR curve    -- floating rate projection

Regulatory use:
- FRTB SA (CRR3 Article 325) -- delta GIRR sensitivity per tenor vertex
- IRRBB (EBA/RTS/2022/09)    -- EVE and NII sensitivity
- EMIR                        -- OIS discounting mandatory for cleared swaps
- IFRS 9                      -- Level 2 fair value
"""

import numpy as np
import pandas as pd
import QuantLib as ql

from quant_risk.instruments.base import Instrument
from quant_risk.curves.base import DiscountCurve


class IRSwap(Instrument):
    """
    Vanilla fixed-for-floating interest rate swap.

    Pay fixed annually (30/360), receive 3M EURIBOR quarterly (ACT/360).
    Discounted at OIS. Floating leg projected from EURIBOR curve built
    as OIS + constant basis spread.

    Parameters
    ----------
    notional : float
        Swap notional in currency units. Example: 10_000_000
    maturity_years : int
        Swap tenor in years. Example: 5
    fixed_rate : float
        Contractual fixed rate in percent. Example: 2.50
    valuation_date : str
        ISO date string. Example: '2026-03-24'
    pay_fixed : bool
        True = pay fixed receive floating (payer swap).
        False = receive fixed pay floating (receiver swap).
    currency : str
        ISO 4217 currency code. Default 'EUR'
    euribor_spread_bps : float
        OIS-EURIBOR basis spread in bps. Default 12.
    settlement_days : int
        Days between trade and settlement. Default 2.

    Example
    -------
    >>> from quant_risk.instruments.swap import IRSwap
    >>> from quant_risk.curves.ois import OISCurve
    >>> curve = OISCurve.from_processed()
    >>> swap  = IRSwap(
    ...     notional       = 10_000_000,
    ...     maturity_years = 5,
    ...     fixed_rate     = 2.50,
    ...     valuation_date = '2026-03-24',
    ... )
    >>> swap.par_rate(curve)
    >>> swap.price(curve)
    >>> swap.dv01(curve)
    """

    _calendar         = ql.TARGET()
    _fixed_dc         = ql.Thirty360(ql.Thirty360.BondBasis)
    _float_dc         = ql.Actual360()
    _fixed_freq       = ql.Annual
    _float_freq       = ql.Quarterly
    _FRTB_VERTICES    = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]

    def __init__(
        self,
        notional          : float,
        maturity_years    : int,
        fixed_rate        : float,
        valuation_date    : str,
        pay_fixed         : bool  = True,
        currency          : str   = "EUR",
        euribor_spread_bps: float = 12.0,
        settlement_days   : int   = 2,
    ):
        self._notional           = notional
        self._maturity_years     = maturity_years
        self._fixed_rate         = fixed_rate / 100
        self._valuation_date_str = valuation_date
        self._pay_fixed          = pay_fixed
        self._currency           = currency
        self._euribor_spread     = euribor_spread_bps / 10000
        self._settlement_days    = settlement_days
        self._ql_valuation_date  = self._parse_date(valuation_date)

    # ------------------------------------------------------------------
    # Instrument interface
    # ------------------------------------------------------------------

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def notional(self) -> float:
        return self._notional

    def price(self, curve: DiscountCurve) -> dict:
        """
        Full revaluation of the swap using OIS discounting.

        Returns
        -------
        dict with keys: npv, fixed_leg_npv, float_leg_npv,
                        par_rate, fixed_rate
        """
        swap, _, _ = self._build_swap(curve)
        npv           = swap.NPV()
        par_rate      = swap.fairRate() * 100
        fixed_leg_npv = swap.fixedLegNPV()
        float_leg_npv = swap.floatingLegNPV()

        return {
            "npv"           : round(npv, 2),
            "fixed_leg_npv" : round(fixed_leg_npv, 2),
            "float_leg_npv" : round(float_leg_npv, 2),
            "par_rate"      : round(par_rate, 4),
            "fixed_rate"    : round(self._fixed_rate * 100, 4),
        }

    def par_rate(self, curve: DiscountCurve) -> float:
        """
        Par swap rate -- fixed rate that sets NPV to zero at inception.

        Parameters
        ----------
        curve : DiscountCurve
            OIS discount curve.

        Returns
        -------
        float
            Par rate in percent.
        """
        swap, _, _ = self._build_swap(curve)
        return round(swap.fairRate() * 100, 4)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        DV01 via parallel 1bp bump of OIS and EURIBOR curves.

        Sign convention: positive = gain when rates rise 1bp (receiver swap).
        Negative = loss when rates rise 1bp (payer swap).

        Parameters
        ----------
        curve : DiscountCurve
        bump : float
            Rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        float
            DV01 in currency units per 1bp.
        """
        npv_base = self.price(curve)["npv"]
        npv_up   = self._npv_bumped(curve, bump)
        return round(npv_base - npv_up, 2)

    def duration(self, curve: DiscountCurve) -> float:
        """
        Approximate modified duration of the fixed leg in years.
        """
        p        = self.price(curve)
        if abs(p["fixed_leg_npv"]) < 1e-6:
            return 0.0
        dv01     = abs(self.dv01(curve))
        return round(dv01 / (abs(p["fixed_leg_npv"]) * 0.0001), 4)

    def cash_flows(self) -> pd.DataFrame:
        """
        Fixed and floating leg cash flows at inception.
        """
        swap, _, _ = self._build_swap_at_par_placeholder()
        records = []
        for cf in swap.fixedLeg():
            records.append({
                "date"   : str(cf.date()),
                "amount" : round(cf.amount(), 2),
                "leg"    : "fixed",
            })
        for cf in swap.floatingLeg():
            records.append({
                "date"   : str(cf.date()),
                "amount" : round(cf.amount(), 2),
                "leg"    : "floating",
            })
        return pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    def key_rate_dv01(
        self,
        curve  : DiscountCurve,
        tenors : list = None,
        bump   : float = 0.0001,
        ) -> dict:
        if tenors is None:
            # use our actual pillar tenors, not FRTB vertices
            tenors = [0.0, 1/12, 2/12, 3/12, 6/12, 9/12, 1, 2, 3, 5, 10, 15]

        ql.Settings.instance().evaluationDate = self._ql_valuation_date
        dates, ois_rates, euribor_rates = self._pillar_rates(curve)
        npv_base = self._npv_from_pillars(dates, ois_rates, euribor_rates, curve)

        result = {}
        for i, tenor in enumerate(tenors):
            if i >= len(ois_rates):
                break
            ois_up    = ois_rates.copy()
            eur_up    = euribor_rates.copy()
            ois_up[i] += bump
            eur_up[i] += bump
            npv_up     = self._npv_from_pillars(dates, ois_up, eur_up, curve)
            label      = "ON" if tenor == 0.0 else (
                f"{int(tenor*12)}M" if tenor < 1 else f"{int(tenor)}Y"
                )
            result[label] = round(npv_base - npv_up, 2)

        return result

    def describe(self) -> str:
        direction = "Payer" if self._pay_fixed else "Receiver"
        return (
            f"IRSwap | {direction} | "
            f"fixed={self._fixed_rate*100:.4f}% | "
            f"maturity={self._maturity_years}Y | "
            f"currency={self._currency} | "
            f"notional={self._notional:,.0f}"
        )

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(date_str: str) -> ql.Date:
        parts = date_str.split("-")
        return ql.Date(int(parts[2]), int(parts[1]), int(parts[0]))

    def _pillar_rates(self, curve: DiscountCurve) -> tuple:
        """Extract OIS and EURIBOR pillar dates and rates."""
        tenors = [0.0, 1/12, 2/12, 3/12, 6/12, 9/12, 1, 2, 3, 5, 10, 15]
        val    = self._ql_valuation_date
        dates  = [
            val if t == 0
            else val + ql.Period(max(1, int(t * 365)), ql.Days)
            for t in tenors
        ]
        ois_rates = [
            curve.zero_rate(1/365) / 100 if t == 0
            else curve.zero_rate(t) / 100
            for t in tenors
        ]
        euribor_rates = [r + self._euribor_spread for r in ois_rates]
        return dates, ois_rates, euribor_rates

    def _build_curves(self, dates, ois_rates, euribor_rates):
        """Build QuantLib ZeroCurve objects from pillar dates and rates."""
        ois_zc = ql.ZeroCurve(
            dates, ois_rates, self._float_dc,
            self._calendar, ql.Linear(), ql.Continuous
        )
        ois_zc.enableExtrapolation()

        euribor_zc = ql.ZeroCurve(
            dates, euribor_rates, self._float_dc,
            self._calendar, ql.Linear(), ql.Continuous
        )
        euribor_zc.enableExtrapolation()

        return (
            ql.YieldTermStructureHandle(ois_zc),
            ql.YieldTermStructureHandle(euribor_zc),
        )

    def _build_schedules(self):
        """Build fixed and floating leg schedules."""
        val        = self._ql_valuation_date
        start      = self._calendar.advance(
            val, self._settlement_days, ql.Days
        )
        end        = self._calendar.advance(
            start, ql.Period(self._maturity_years, ql.Years)
        )
        fixed_sch  = ql.Schedule(
            start, end, ql.Period(self._fixed_freq),
            self._calendar,
            ql.ModifiedFollowing, ql.ModifiedFollowing,
            ql.DateGeneration.Forward, False
        )
        float_sch  = ql.Schedule(
            start, end, ql.Period(self._float_freq),
            self._calendar,
            ql.ModifiedFollowing, ql.ModifiedFollowing,
            ql.DateGeneration.Forward, False
        )
        return fixed_sch, float_sch

    def _add_past_fixings(self, euribor_index, fixed_sch, float_sch):
        """Add historical EURIBOR fixings for past reset dates."""
        temp = ql.VanillaSwap(
            ql.VanillaSwap.Payer, self._notional,
            fixed_sch, self._fixed_rate, self._fixed_dc,
            float_sch, euribor_index, 0.0, self._float_dc,
        )
        val = ql.Settings.instance().evaluationDate
        for cf in temp.floatingLeg():
            coupon = ql.as_floating_rate_coupon(cf)
            if coupon:
                fd = coupon.fixingDate()
                if fd < val and euribor_index.isValidFixingDate(fd):
                    euribor_index.addFixing(
                        fd, self._fixed_rate + self._euribor_spread
                    )

    def _build_swap(self, curve: DiscountCurve):
        """Build and price the swap from the given curve."""
        ql.Settings.instance().evaluationDate = self._ql_valuation_date
        dates, ois_rates, euribor_rates = self._pillar_rates(curve)
        ois_handle, euribor_handle      = self._build_curves(
            dates, ois_rates, euribor_rates
        )
        fixed_sch, float_sch = self._build_schedules()

        euribor_index = ql.Euribor3M(euribor_handle)
        self._add_past_fixings(euribor_index, fixed_sch, float_sch)

        direction = ql.VanillaSwap.Payer if self._pay_fixed else ql.VanillaSwap.Receiver
        swap = ql.VanillaSwap(
            direction, self._notional,
            fixed_sch, self._fixed_rate, self._fixed_dc,
            float_sch, euribor_index, 0.0, self._float_dc,
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(ois_handle))
        return swap, fixed_sch, float_sch

    def _build_swap_at_par_placeholder(self):
        """Build swap with placeholder rate for cash flow inspection."""
        return self._build_swap_internal(0.0)

    def _build_swap_internal(self, rate):
        ql.Settings.instance().evaluationDate = self._ql_valuation_date
        fixed_sch, float_sch = self._build_schedules()
        euribor_index = ql.Euribor3M()
        direction = ql.VanillaSwap.Payer if self._pay_fixed else ql.VanillaSwap.Receiver
        swap = ql.VanillaSwap(
            direction, self._notional,
            fixed_sch, rate, self._fixed_dc,
            float_sch, euribor_index, 0.0, self._float_dc,
        )
        return swap, fixed_sch, float_sch

    def _npv_bumped(self, curve: DiscountCurve, bump: float) -> float:
        """NPV after parallel bump of all pillar rates."""
        dates, ois_rates, euribor_rates = self._pillar_rates(curve)
        ois_up  = [r + bump for r in ois_rates]
        eur_up  = [r + bump for r in euribor_rates]
        return self._npv_from_pillars(dates, ois_up, eur_up, curve)

    def _npv_from_pillars(
        self, dates, ois_rates, euribor_rates, curve
    ) -> float:
        """NPV from explicit pillar dates and rates."""
        ois_handle, euribor_handle = self._build_curves(
            dates, ois_rates, euribor_rates
        )
        fixed_sch, float_sch = self._build_schedules()
        euribor_index = ql.Euribor3M(euribor_handle)
        self._add_past_fixings(euribor_index, fixed_sch, float_sch)

        direction = ql.VanillaSwap.Payer if self._pay_fixed else ql.VanillaSwap.Receiver
        swap = ql.VanillaSwap(
            direction, self._notional,
            fixed_sch, self._fixed_rate, self._fixed_dc,
            float_sch, euribor_index, 0.0, self._float_dc,
        )
        swap.setPricingEngine(ql.DiscountingSwapEngine(ois_handle))
        return swap.NPV()