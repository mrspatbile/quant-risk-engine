# src/quant_risk/instruments/bond.py

"""
Fixed rate bond pricer -- OIS discounting via QuantLib.

Prices government and corporate fixed rate bonds using the multi-curve
framework: coupon rate set at issuance referencing the government benchmark,
discounted at the OIS risk-free curve.

Regulatory use:
- FRTB SA (CRR3 Article 325) -- delta GIRR sensitivity per tenor vertex
- IRRBB (EBA/RTS/2022/09)    -- key rate DV01 feeds EVE bucket calculation
- IFRS 9                      -- Level 2 fair value, OIS discounting
"""

import numpy as np
import pandas as pd
import QuantLib as ql
from datetime import datetime

from quant_risk.instruments.base import Instrument
from quant_risk.curves.base import DiscountCurve


class Bond(Instrument):
    """
    Fixed rate bond priced via OIS discounting.

    Wraps QuantLib FixedRateBond with a clean interface for pricing,
    risk sensitivity, and cash flow analysis.

    Parameters
    ----------
    isin : str
        ISIN identifier. Example: 'DE0001102580'
    face_value : float
        Notional in currency units. Example: 1_000_000
    coupon_rate : float
        Annual coupon rate in percent. Example: 2.60
    issue_date : str
        Issue date as ISO string. Example: '2023-08-15'
    maturity_date : str
        Maturity date as ISO string. Example: '2034-08-15'
    currency : str
        ISO 4217 currency code. Default 'EUR'
    settlement_days : int
        Days between trade and settlement. Default 2 (T+2)
    frequency : ql.Frequency
        Coupon frequency. Default ql.Annual (Bund convention)

    Example
    -------
    >>> from quant_risk.instruments.bond import Bond
    >>> from quant_risk.curves.ois import OISCurve
    >>> curve = OISCurve.from_processed()
    >>> bond  = Bond(
    ...     isin          = 'DE0001102580',
    ...     face_value    = 1_000_000,
    ...     coupon_rate   = 2.60,
    ...     issue_date    = '2023-08-15',
    ...     maturity_date = '2034-08-15',
    ... )
    >>> bond.price(curve)
    >>> bond.dv01(curve)
    >>> bond.key_rate_dv01(curve)
    """

    _calendar         = ql.TARGET()
    _day_count        = ql.ActualActual(ql.ActualActual.Bond)

    def __init__(
        self,
        isin          : str,
        face_value    : float,
        coupon_rate   : float,
        issue_date    : str,
        maturity_date : str,
        currency      : str = "EUR",
        settlement_days: int = 2,
        frequency     : int = ql.Annual,
    ):
        self._isin            = isin
        self._face_value      = face_value
        self._coupon_rate     = coupon_rate
        self._issue_date      = self._parse_date(issue_date)
        self._maturity_date   = self._parse_date(maturity_date)
        self._currency        = currency
        self._settlement_days = settlement_days
        self._frequency       = frequency
        self._ql_bond         = self._build_ql_bond()

    # ------------------------------------------------------------------
    # Instrument interface
    # ------------------------------------------------------------------

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def notional(self) -> float:
        return self._face_value

    def price(self, curve: DiscountCurve) -> dict:
        """
        Full revaluation price using OIS discount curve.

        Returns
        -------
        dict with keys: clean_price, dirty_price, accrued,
                        ytm, duration, convexity, dv01
        """
        val_date = self._valuation_date(curve)

        def _impl():
            engine = self._build_engine(curve)
            self._ql_bond.setPricingEngine(engine)
            dirty_price = self._ql_bond.dirtyPrice()
            clean_price = self._ql_bond.cleanPrice()
            accrued     = self._ql_bond.accruedAmount()
            ytm         = self._ql_bond.bondYield(
                self._day_count, ql.Compounded, ql.Annual
            ) * 100
            ir          = ql.InterestRate(
                ytm / 100, self._day_count, ql.Compounded, ql.Annual
            )
            duration    = ql.BondFunctions.duration(
                self._ql_bond, ir, ql.Duration.Modified
            )
            convexity   = ql.BondFunctions.convexity(self._ql_bond, ir)
            dv01        = dirty_price * duration / 10000
            return {
                "clean_price" : round(clean_price, 4),
                "dirty_price" : round(dirty_price, 4),
                "accrued"     : round(accrued, 4),
                "ytm"         : round(ytm, 4),
                "duration"    : round(duration, 4),
                "convexity"   : round(convexity, 4),
                "dv01"        : round(dv01, 4),
            }

        return self._with_eval_date(val_date, _impl)

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        DV01 via parallel bump of OIS curve by 1bp.
        """
        val_date = self._valuation_date(curve)

        def _impl():
            p_base = self.price(curve)["dirty_price"]
            p_up   = self._price_bumped(curve, bump)
            return round((p_base - p_up) * self._face_value / 100, 2)

        return self._with_eval_date(val_date, _impl)

    def duration(self, curve: DiscountCurve) -> float:
        """Modified duration in years."""
        return self.price(curve)["duration"]

    def cash_flows(self) -> pd.DataFrame:
        """
        Scheduled cash flows with dates, amounts, and type.
        """
        records = []
        settlement = self._ql_bond.settlementDate()
        for cf in self._ql_bond.cashflows():
            if not cf.hasOccurred(settlement):
                amount = cf.amount()
                cf_type = "principal" if amount > self._face_value * 0.5 else "coupon"
                records.append({
                    "date"   : str(cf.date()),
                    "amount" : round(amount, 2),
                    "type"   : cf_type,
                })
        return pd.DataFrame(records)

    def key_rate_dv01(
        self,
        curve  : DiscountCurve,
        tenors : list = None,
        bump   : float = 0.0001,
    ) -> dict:
        """
        Key rate DV01 per tenor bucket -- FRTB SA delta GIRR sensitivity.

        Parameters
        ----------
        curve : DiscountCurve
            Baseline OIS curve.
        tenors : list, optional
            Tenor buckets in years. Defaults to FRTB SA vertices.
        bump : float
            Rate bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        dict {tenor_label: dv01_eur}
        """
        if tenors is None:
            tenors = self._FRTB_VERTICES

        val_date = self._valuation_date(curve)

        def _impl():
            dates, rates = self._curve_pillars(curve, val_date)
            base_price   = self._price_from_pillars(dates, rates)
            result = {}
            for i, tenor in enumerate(tenors):
                rates_up    = rates.copy()
                rates_up[i + 1] += bump
                price_up    = self._price_from_pillars(dates, rates_up)
                dv01_kr     = (base_price - price_up) * self._face_value / 100
                label       = f"{int(tenor*12)}M" if tenor < 1 else f"{int(tenor)}Y"
                result[label] = round(dv01_kr, 2)
            return result

        return self._with_eval_date(val_date, _impl)

    def z_spread(
        self,
        market_clean_price: float,
        curve: DiscountCurve,
    ) -> float:
        """
        Z-spread over OIS curve in basis points.

        The constant spread added to the OIS zero curve at every maturity
        such that the discounted cash flows equal the market price.

        Parameters
        ----------
        market_clean_price : float
            Observed market clean price per 100 face value.
        curve : DiscountCurve
            OIS discount curve -- the risk-free benchmark.

        Returns
        -------
        float
            Z-spread in basis points.
        """
        val_date = self._valuation_date(curve)
        ql.Settings.instance().evaluationDate = val_date

        dates, rates = self._curve_pillars(curve, val_date)
        zc           = self._build_zc(dates, rates)

        engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(zc))
        self._ql_bond.setPricingEngine(engine)

        bond_price = ql.BondPrice(market_clean_price, ql.BondPrice.Clean)

        z = ql.BondFunctions.zSpread(
            self._ql_bond,
            bond_price,
            zc,
            self._day_count,
            ql.Compounded,
            ql.Annual,
            self._ql_bond.settlementDate(),
        )
        return round(z * 10000, 4)

    def describe(self) -> str:
        return (
            f"Bond | {self._isin} | "
            f"coupon={self._coupon_rate}% | "
            f"maturity={self._maturity_date} | "
            f"currency={self._currency} | "
            f"notional={self._face_value:,.0f}"
        )

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _build_ql_bond(self) -> ql.FixedRateBond:
        schedule = ql.Schedule(
        self._issue_date,          # effectiveDate
        self._maturity_date,       # terminationDate
        ql.Period(self._frequency),# tenor
        self._calendar,            # calendar
        ql.ModifiedFollowing,      # convention
        ql.ModifiedFollowing,      # terminationDateConvention
        ql.DateGeneration.Backward,# rule
        False,                     # endOfMonth
        )
        return ql.FixedRateBond(
            self._settlement_days,
            self._face_value,
            schedule,
            [self._coupon_rate / 100.0],
            self._day_count,
        )

    def _valuation_date(self, curve: DiscountCurve) -> ql.Date:
        """Parse valuation date from curve."""
        parts = curve.valuation_date.split("-")
        return ql.Date(int(parts[2]), int(parts[1]), int(parts[0]))

    def _curve_pillars(
        self, curve: DiscountCurve, val_date: ql.Date
    ) -> tuple:
        """Extract OIS pillar dates and rates."""
        tenors = [0.0, 1/12, 2/12, 3/12, 6/12, 9/12, 1, 2, 3, 5, 10, 15]
        dates  = [
            val_date if t == 0
            else val_date + ql.Period(max(1, int(t * 365)), ql.Days)
            for t in tenors
        ]
        rates  = [
            curve.zero_rate(1/365) / 100 if t == 0
            else curve.zero_rate(t) / 100
            for t in tenors
        ]
        return dates, rates

    def _build_zc(self, dates, rates) -> ql.ZeroCurve:
        """Build QuantLib ZeroCurve from pillar dates and rates."""
        zc = ql.ZeroCurve(
            dates, rates, self._day_count,
            self._calendar, ql.Linear(), ql.Continuous
        )
        zc.enableExtrapolation()
        return zc

    def _build_engine(
        self, curve: DiscountCurve
    ) -> ql.DiscountingBondEngine:
        """Build pricing engine from discount curve."""
        val_date = self._valuation_date(curve)
        ql.Settings.instance().evaluationDate = val_date
        dates, rates = self._curve_pillars(curve, val_date)
        zc           = self._build_zc(dates, rates)
        return ql.DiscountingBondEngine(ql.YieldTermStructureHandle(zc))

    def _price_bumped(self, curve: DiscountCurve, bump: float) -> float:
        """Price bond after parallel bump of all pillar rates."""
        val_date     = self._valuation_date(curve)
        dates, rates = self._curve_pillars(curve, val_date)
        rates_up     = [r + bump for r in rates]
        zc_up        = self._build_zc(dates, rates_up)
        engine_up    = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(zc_up))
        self._ql_bond.setPricingEngine(engine_up)
        return self._ql_bond.dirtyPrice()

    def _price_from_pillars(self, dates, rates) -> float:
        """Price bond from explicit pillar dates and rates."""
        zc     = self._build_zc(dates, rates)
        engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(zc))
        self._ql_bond.setPricingEngine(engine)
        return self._ql_bond.dirtyPrice()