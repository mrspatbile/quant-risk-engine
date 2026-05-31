# src/quant_risk/instruments/cds.py

"""
Credit Default Swap (CDS) pricing -- QuantLib implementation.

Follows the Instrument ABC pattern with dependency injection:
- OIS discount curve passed to price(), dv01(), duration()
- Hazard rate curve injected at construction

Three construction paths:
- CreditDefaultSwap(...)              -- direct constructor, pre-built hazard handle
- CreditDefaultSwap.from_flat_spread  -- flat hazard rate from single spread
- CreditDefaultSwap.from_market_quotes -- bootstrapped PiecewiseFlatHazardRate

ISDA Big Bang (2009) conventions:
- Standardised coupons: 100bps (IG), 500bps (HY)
- Quarterly payments on 20th IMM dates (TwentiethIMM)
- Day count: Actual/360
- Recovery: 40% senior unsecured convention
- Upfront payment settles (par spread - coupon) x RPV01 x notional

Regulatory context:
- FRTB SA CSR-NS: CS01 per tenor vertex (CRR3 Article 325)
- FRTB Default Risk Charge: jump-to-default
- EMIR: reporting and clearing obligations
- IFRS 9/13: Level 2 fair value, OIS discounting
"""

import numpy as np
import pandas as pd
import QuantLib as ql
from typing import Optional

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


# ISDA standard coupons
STANDARD_COUPON_IG = 0.0100   # 100bps
STANDARD_COUPON_HY = 0.0500   # 500bps

# ISDA senior unsecured recovery convention
STANDARD_RECOVERY = 0.40

# ── helper: build IMM maturity date ──────────────────────────────────────────
def _imm_maturity(valuation_date: ql.Date, tenor_years: float) -> ql.Date:
    """Nearest TwentiethIMM date at or after valuation + tenor."""
    months = int(round(tenor_years * 12))
    raw    = ql.TARGET().advance(
        valuation_date, ql.Period(months, ql.Months)
    )
    # roll to 20th of the IMM month (Mar/Jun/Sep/Dec)
    imm_months = [3, 6, 9, 12]
    m = raw.month()
    y = raw.year()
    # find next IMM month
    for im in imm_months:
        if m <= im:
            return ql.Date(20, im, y)
    return ql.Date(20, 3, y + 1)


class CreditDefaultSwap(Instrument):
    """
    Credit Default Swap pricer -- QuantLib MidPointCdsEngine.

    Parameters
    ----------
    valuation_date : ql.Date
        QuantLib valuation date -- must match global clock.
    maturity : ql.Date
        CDS maturity date -- typically a TwentiethIMM date.
    notional_ : float
        Notional in currency units.
    coupon : float
        Running coupon in decimal. Standard: 0.01 (IG) or 0.05 (HY).
    hazard_handle : ql.DefaultProbabilityTermStructureHandle
        Pre-built hazard rate curve handle.
    recovery : float
        Recovery rate assumption. Default 0.40.
    currency_ : str
        ISO 4217 currency code. Default 'EUR'.
    protection_buyer : bool
        True = protection buyer perspective. Default True.
    """

    def __init__(
        self,
        valuation_date: ql.Date,
        maturity: ql.Date,
        notional_: float,
        coupon: float,
        hazard_handle: ql.DefaultProbabilityTermStructureHandle,
        recovery: float = STANDARD_RECOVERY,
        currency_: str = 'EUR',
        protection_buyer: bool = True,
    ):
        self._valuation_date  = valuation_date
        self._maturity        = maturity
        self._notional        = notional_
        self._coupon          = coupon
        self._hazard_handle   = hazard_handle
        self._recovery        = recovery
        self._currency        = currency_
        self._protection_buyer = protection_buyer
        self._calendar        = ql.TARGET()
        self._day_count       = ql.Actual360()   # ISDA CDS convention

        self._side = (
            ql.Protection.Buyer if protection_buyer else ql.Protection.Seller
        )
        self._schedule = ql.Schedule(
            valuation_date, maturity,
            ql.Period(ql.Quarterly),
            self._calendar,
            ql.Following, ql.Unadjusted,
            ql.DateGeneration.TwentiethIMM, False
        )
        self._cds = ql.CreditDefaultSwap(
            self._side,
            notional_,
            coupon,
            self._schedule,
            ql.Following,
            self._day_count,
        )

    # ── classmethods -- alternative constructors ──────────────────────────────

    @classmethod
    def from_flat_spread(
        cls,
        valuation_date: ql.Date,
        maturity: ql.Date,
        notional_: float,
        par_spread: float,
        coupon: float = STANDARD_COUPON_IG,
        recovery: float = STANDARD_RECOVERY,
        currency_: str = 'EUR',
        protection_buyer: bool = True,
    ) -> 'CreditDefaultSwap':
        """
        Construct CDS with flat hazard rate implied from par spread.

        Uses the credit triangle: h = spread / (1 - R).
        Suitable for single-tenor pricing and intuition.
        Not for production curve risk -- use from_market_quotes.

        Parameters
        ----------
        par_spread : float
            Market CDS spread in decimal (e.g. 0.0150 for 150bps).
        """
        day_count    = ql.Actual365Fixed()
        hazard_rate  = par_spread / (1 - recovery)
        hazard_curve = ql.FlatHazardRate(
            valuation_date,
            ql.QuoteHandle(ql.SimpleQuote(hazard_rate)),
            day_count,
        )
        hazard_handle = ql.DefaultProbabilityTermStructureHandle(hazard_curve)
        return cls(
            valuation_date, maturity, notional_, coupon,
            hazard_handle, recovery, currency_, protection_buyer
        )

    @classmethod
    def from_market_quotes(
        cls,
        valuation_date: ql.Date,
        maturity: ql.Date,
        notional_: float,
        market_tenors: list,
        market_spreads: list,
        coupon: float = STANDARD_COUPON_IG,
        recovery: float = STANDARD_RECOVERY,
        currency_: str = 'EUR',
        protection_buyer: bool = True,
        disc_handle: Optional[ql.YieldTermStructureHandle] = None,
    ) -> 'CreditDefaultSwap':
        """
        Construct CDS with bootstrapped PiecewiseFlatHazardRate curve.

        Production-grade -- fits the full credit curve term structure
        from market CDS quotes at standard tenors.

        Parameters
        ----------
        market_tenors : list of int
            Tenors in years, e.g. [1, 2, 3, 5, 7, 10].
        market_spreads : list of float
            Par spreads in decimal at each tenor.
        disc_handle : ql.YieldTermStructureHandle, optional
            OIS discount handle for bootstrapping. If None uses flat 2.5%.
        """
        day_count = ql.Actual365Fixed()
        calendar  = ql.TARGET()

        if disc_handle is None:
            flat_ois   = ql.FlatForward(valuation_date, 0.025, day_count)
            disc_handle = ql.YieldTermStructureHandle(flat_ois)

        helpers = []
        for t, s in zip(market_tenors, market_spreads):
            helper = ql.SpreadCdsHelper(
                ql.QuoteHandle(ql.SimpleQuote(s)),
                ql.Period(t, ql.Years),
                0,
                calendar,
                ql.Quarterly,
                ql.Following,
                ql.DateGeneration.TwentiethIMM,
                day_count,
                recovery,
                disc_handle,
            )
            helpers.append(helper)

        bootstrapped = ql.PiecewiseFlatHazardRate(
            valuation_date, helpers, day_count
        )
        bootstrapped.enableExtrapolation()
        hazard_handle = ql.DefaultProbabilityTermStructureHandle(bootstrapped)

        return cls(
            valuation_date, maturity, notional_, coupon,
            hazard_handle, recovery, currency_, protection_buyer
        )

    # ── internal engine setup ─────────────────────────────────────────────────

    def _get_engine(
        self, disc_handle: ql.YieldTermStructureHandle
    ) -> ql.MidPointCdsEngine:
        """MidPointCdsEngine -- ISDA standard for CDS pricing."""
        return ql.MidPointCdsEngine(
            self._hazard_handle, self._recovery, disc_handle
        )

    def _reprice(
        self, disc_handle: ql.YieldTermStructureHandle
    ) -> ql.CreditDefaultSwap:
        """Attach engine and return priced CDS."""
        self._cds.setPricingEngine(self._get_engine(disc_handle))
        return self._cds

    def _build_cds_with_hazard(
        self,
        hazard_handle: ql.DefaultProbabilityTermStructureHandle,
        disc_handle: ql.YieldTermStructureHandle,
        maturity: Optional[ql.Date] = None,
        coupon: Optional[float] = None,
    ) -> ql.CreditDefaultSwap:
        """Build and price a fresh CDS -- used for bumped scenarios."""
        mat   = maturity or self._maturity
        coup  = coupon or self._coupon
        sched = ql.Schedule(
            self._valuation_date, mat,
            ql.Period(ql.Quarterly),
            self._calendar,
            ql.Following, ql.Unadjusted,
            ql.DateGeneration.TwentiethIMM, False
        )
        cds = ql.CreditDefaultSwap(
            self._side, self._notional, coup,
            sched, ql.Following, self._day_count
        )
        cds.setPricingEngine(
            ql.MidPointCdsEngine(hazard_handle, self._recovery, disc_handle)
        )
        return cds


    # ── Instrument ABC implementation ─────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def notional(self) -> float:
        return self._notional

    def price(self, curve: DiscountCurve) -> float:
        """
        CDS NPV -- protection buyer perspective by default.

        Positive NPV: position has gained value (spreads widened for buyer).
        Negative NPV: position has lost value (spreads tightened for buyer).
        """
        disc_handle = self._disc_handle_from_curve(curve)
        return self._reprice(disc_handle).NPV()

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        IR PV01 -- change in NPV for 1bp parallel shift in OIS curve.

        Note: for CDS the dominant sensitivity is CS01 (credit spread),
        not IR PV01. Use cs01() for credit risk measurement.
        IR PV01 is implemented here to satisfy the Instrument ABC.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        npv_base    = self._reprice(disc_handle).NPV()

        # bump OIS curve by 1bp -- rebuild handle
        bumped_curve  = ql.ZeroSpreadedTermStructure(
            disc_handle, ql.QuoteHandle(ql.SimpleQuote(bump))
        )
        bumped_handle = ql.YieldTermStructureHandle(bumped_curve)
        npv_up        = self._build_cds_with_hazard(
            self._hazard_handle, bumped_handle
        ).NPV()
        return npv_up - npv_base

    def duration(self, curve: DiscountCurve) -> float:
        """
        Risky annuity (RPV01) -- natural duration equivalent for CDS.

        RPV01 = sum_i delta_i * P(0,ti) * S(0,ti)

        Measures the present value of receiving 1 unit per year,
        weighted by survival probability. Stops on default -- hence risky.
        Units: years.
        """
        disc_handle   = self._disc_handle_from_curve(curve)
        priced        = self._reprice(disc_handle)
        pv01_bps      = priced.couponLegBPS()   # value of 1bp on premium leg
        risky_annuity = -pv01_bps / 0.0001 / self._notional
        return risky_annuity

    def cash_flows(self) -> pd.DataFrame:
        """
        Premium leg scheduled cash flows.

        Returns date, amount (undiscounted), and type for each payment.
        Does not include the contingent protection payment -- that is
        path-dependent on default time.
        """
        dates   = list(self._schedule)[1:]   # exclude start date
        amount  = self._coupon * self._notional
        records = []
        for i, d in enumerate(dates):
            prev = list(self._schedule)[i]
            dcf  = ql.Actual360().yearFraction(prev, d)
            records.append({
                'date':   d.ISO(),
                'amount': amount * dcf,
                'type':   'premium',
            })
        return pd.DataFrame(records)

    def rate_sensitivities(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        """
        CS01 at caller-specified tenor vertices.

        For each tenor, prices a CDS maturing at that tenor and measures
        its sensitivity to a 1bp parallel hazard rate bump. The caller
        supplies the vertex list.

        Parameters
        ----------
        curve : DiscountCurve
            OIS discount curve.
        tenors : list[float]
            Vertex maturities in years, e.g. [0.5, 1.0, 2.0, 3.0, 5.0, 10.0].
        bump : float
            Spread bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        dict[float, float]
            {tenor_years: cs01} in currency units per 1bp.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        hazard_bump = bump / (1 - self._recovery)

        result = {}
        for t in tenors:
            mat = _imm_maturity(self._valuation_date, t)
            h_up_val = self._hazard_handle.currentLink().hazardRate(
                self._valuation_date + ql.Period(max(1, int(t * 365)), ql.Days)
            ) + hazard_bump
            hc_up = ql.FlatHazardRate(
                self._valuation_date,
                ql.QuoteHandle(ql.SimpleQuote(h_up_val)),
                ql.Actual365Fixed()
            )
            hh_up      = ql.DefaultProbabilityTermStructureHandle(hc_up)
            npv_up     = self._build_cds_with_hazard(hh_up, disc_handle, maturity=mat).NPV()
            npv_base_t = self._build_cds_with_hazard(self._hazard_handle, disc_handle, maturity=mat).NPV()
            result[t]  = npv_up - npv_base_t

        return result

    # ── CDS-specific methods (beyond ABC) ────────────────────────────────────

    def cs01(
        self,
        curve: DiscountCurve,
        tenors: list[float],
        bump: float = 0.0001,
    ) -> dict[float, float]:
        """
        CS01 at each caller-specified tenor vertex.

        For each tenor t, samples the hazard rate at that point, applies a
        1bp flat bump, and measures the NPV change of this CDS. The caller
        supplies the vertex list.

        Parameters
        ----------
        curve : DiscountCurve
            OIS discount curve.
        tenors : list[float]
            Credit spread tenor vertices in years, e.g. [1.0, 3.0, 5.0].
        bump : float
            Spread bump in decimal. Default 0.0001 (1bp).

        Returns
        -------
        dict[float, float]
            {tenor_years: cs01} in currency units per 1bp.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        npv_base    = self._reprice(disc_handle).NPV()
        hazard_bump = bump / (1 - self._recovery)

        result = {}
        for t in tenors:
            tenor_date = self._valuation_date + ql.Period(max(1, int(t * 365)), ql.Days)
            h_val    = self._hazard_handle.currentLink().hazardRate(tenor_date)
            h_bumped = h_val + hazard_bump
            hc_up = ql.FlatHazardRate(
                self._valuation_date,
                ql.QuoteHandle(ql.SimpleQuote(h_bumped)),
                ql.Actual365Fixed()
            )
            hh_up  = ql.DefaultProbabilityTermStructureHandle(hc_up)
            npv_up = self._build_cds_with_hazard(hh_up, disc_handle).NPV()
            result[t] = npv_up - npv_base
        return result

    def par_spread(self, curve: DiscountCurve) -> float:
        """Current par spread -- the coupon that makes NPV zero."""
        disc_handle = self._disc_handle_from_curve(curve)
        return self._reprice(disc_handle).fairSpread()

    def risky_annuity(self, curve: DiscountCurve) -> float:
        """RPV01 -- alias for duration(). See duration() docstring."""
        return self.duration(curve)

    def upfront(self, curve: DiscountCurve) -> float:
        """
        Upfront payment at inception.

        Positive: buyer pays upfront (par spread > coupon).
        Negative: buyer receives upfront (par spread < coupon).

        Upfront = (par_spread - coupon) x RPV01 x notional
        """
        s_star = self.par_spread(curve)
        rpv01  = self.duration(curve)
        return (s_star - self._coupon) * rpv01 * self._notional

    def jump_to_default(self, curve: DiscountCurve) -> dict:
        """
        Jump-to-default P&L -- discontinuous loss/gain on immediate default.

        For protection buyer:
            JTD = LGD - PV(remaining premium)

        Positive JTD for buyer -- default is a gain event.
        Feeds into FRTB Default Risk Charge (DRC).
        """
        disc_handle = self._disc_handle_from_curve(curve)
        priced      = self._reprice(disc_handle)
        lgd         = (1 - self._recovery) * self._notional
        pv_premium  = abs(priced.couponLegNPV())

        jtd_buyer  =  lgd - pv_premium
        jtd_seller = -lgd + pv_premium

        return {
            'lgd':        lgd,
            'pv_premium': pv_premium,
            'jtd_buyer':  jtd_buyer,
            'jtd_seller': jtd_seller,
        }

    def puf(self, curve: DiscountCurve) -> float:
        """
        Points upfront -- upfront as percentage of notional.

        PUF = NPV / notional * 100
        Standard market quoting convention post-ISDA Big Bang 2009.
        """
        disc_handle = self._disc_handle_from_curve(curve)
        return self._reprice(disc_handle).NPV() / self._notional * 100

    def describe(self) -> str:
        side = 'Buyer' if self._protection_buyer else 'Seller'
        return (
            f"CreditDefaultSwap | {side} | {self._currency} | "
            f"Notional={self._notional:,.0f} | "
            f"Coupon={self._coupon*10000:.0f}bps | "
            f"Maturity={self._maturity.ISO()} | "
            f"Recovery={self._recovery*100:.0f}%"
        )