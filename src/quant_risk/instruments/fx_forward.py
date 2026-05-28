# src/quant_risk/instruments/fx_forward.py

"""
FX Forward pricer -- CIP-based valuation.

Forward rate formula (EUR base, USD terms):
    F = S * P_EUR(0,T) / P_USD(0,T)

NPV (exporter/hedger perspective -- deliver foreign, receive domestic):
    NPV = N_f * (F_0 - F_t) * P_EUR(0,T)

Where:
    F_0 = contractual forward rate at inception
    F_t = current market forward rate
    N_f = foreign currency notional
    P_EUR = EUR OIS discount factor
    P_USD = USD OIS/Treasury discount factor

Sign convention:
    Positive NPV: domestic has weakened since inception (F_t < F_0) -- exporter gains
    Negative NPV: domestic has strengthened since inception (F_t > F_0) -- exporter loses

Regulatory context:
    FRTB SA -- delta FX (FX risk bucket), delta IR (GIRR bucket)
    EMIR -- FX forwards > 3 days are OTC derivatives, reporting mandatory
    IFRS 9/13 -- Level 2 fair value, CIP observable inputs
    AIFMD II -- hedge effectiveness monitoring, monthly rebalancing
"""

import numpy as np
import pandas as pd
import QuantLib as ql
from typing import Optional

from quant_risk.curves.base import DiscountCurve
from quant_risk.instruments.base import Instrument


class FXForward(Instrument):
    """
    FX Forward pricer -- exporter/hedger perspective by default.

    Prices a forward contract to deliver foreign currency and receive
    domestic (EUR) currency at the contractual forward rate F_0.

    Parameters
    ----------
    notional_foreign : float
        Notional in foreign currency units (e.g. USD).
    spot_rate : float
        Spot rate at valuation -- domestic per foreign (EUR/USD).
    f0 : float or None
        Contractual forward rate at inception -- domestic per foreign.
        If None, resolved to current market forward on first price() call.
    maturity_date : ql.Date
        Forward settlement date.
    valuation_date : ql.Date
        QuantLib valuation date -- must match global clock.
    usd_disc_handle : ql.YieldTermStructureHandle
        Foreign currency (USD) discount curve handle.
    domestic_currency : str
        ISO 4217 domestic currency code. Default 'EUR'.
    foreign_currency : str
        ISO 4217 foreign currency code. Default 'USD'.
    basis_bps : float
        Cross-currency basis in bps. Default 0 (pure CIP).
        Negative for EUR/USD (USD scarce vs EUR).
    exporter : bool
        True = deliver foreign, receive domestic (exporter hedger).
        False = deliver domestic, receive foreign (importer hedger).
    """

    def __init__(
        self,
        notional_foreign: float,
        spot_rate: float,
        f0: Optional[float],
        maturity_date: ql.Date,
        valuation_date: ql.Date,
        usd_disc_handle: ql.YieldTermStructureHandle,
        domestic_currency: str = 'EUR',
        foreign_currency: str = 'USD',
        basis_bps: float = 0.0,
        exporter: bool = True,
    ):
        self._notional_foreign  = notional_foreign
        self._spot_rate         = spot_rate
        self._f0                = f0
        self._maturity_date     = maturity_date
        self._valuation_date    = valuation_date
        self._usd_disc_handle   = usd_disc_handle
        self._domestic_currency = domestic_currency
        self._foreign_currency  = foreign_currency
        self._basis             = basis_bps / 10000
        self._exporter          = exporter
        self._day_count         = ql.Actual365Fixed()
        self._sign              = 1.0 if exporter else -1.0
        self._T                 = self._day_count.yearFraction(
            valuation_date, maturity_date
        )

    # ── classmethods -- alternative constructors ──────────────────────────────

    @classmethod
    def at_inception(
        cls,
        notional_foreign: float,
        spot_rate: float,
        maturity_date: ql.Date,
        valuation_date: ql.Date,
        usd_disc_handle: ql.YieldTermStructureHandle,
        domestic_currency: str = 'EUR',
        foreign_currency: str = 'USD',
        basis_bps: float = 0.0,
        exporter: bool = True,
    ) -> 'FXForward':
        """
        Construct forward at inception -- F_0 set to current market forward.
        NPV = 0 at construction.
        """
        return cls(
            notional_foreign  = notional_foreign,
            spot_rate         = spot_rate,
            f0                = None,
            maturity_date     = maturity_date,
            valuation_date    = valuation_date,
            usd_disc_handle   = usd_disc_handle,
            domestic_currency = domestic_currency,
            foreign_currency  = foreign_currency,
            basis_bps         = basis_bps,
            exporter          = exporter,
        )

    @classmethod
    def seasoned(
        cls,
        notional_foreign: float,
        spot_rate: float,
        f0: float,
        maturity_date: ql.Date,
        valuation_date: ql.Date,
        usd_disc_handle: ql.YieldTermStructureHandle,
        domestic_currency: str = 'EUR',
        foreign_currency: str = 'USD',
        basis_bps: float = 0.0,
        exporter: bool = True,
    ) -> 'FXForward':
        """
        Construct seasoned forward -- F_0 locked at inception rate.
        NPV reflects market move since inception.
        """
        return cls(
            notional_foreign  = notional_foreign,
            spot_rate         = spot_rate,
            f0                = f0,
            maturity_date     = maturity_date,
            valuation_date    = valuation_date,
            usd_disc_handle   = usd_disc_handle,
            domestic_currency = domestic_currency,
            foreign_currency  = foreign_currency,
            basis_bps         = basis_bps,
            exporter          = exporter,
        )

    # ── internal helpers ──────────────────────────────────────────────────────


    def _df_eur(self, eur_handle: ql.YieldTermStructureHandle) -> float:
        """EUR OIS discount factor to maturity."""
        return eur_handle.discount(self._maturity_date)

    def _df_usd(self) -> float:
        """USD discount factor to maturity."""
        return self._usd_disc_handle.discount(self._maturity_date)

    def _market_forward(
        self, eur_handle: ql.YieldTermStructureHandle
    ) -> float:
        """
        CIP forward rate with optional cross-currency basis.

        F = S * P_EUR(0,T) / P_USD(0,T) * exp(basis * T)
        """
        df_eur = self._df_eur(eur_handle)
        df_usd = self._df_usd()
        f_cip  = self._spot_rate * df_eur / df_usd
        return f_cip * np.exp(self._basis * self._T)

    def _resolve_f0(self, eur_handle: ql.YieldTermStructureHandle) -> float:
        """Return F_0 -- resolve to market forward if set at inception."""
        if self._f0 is None:
            return self._market_forward(eur_handle)
        return self._f0

    # ── Instrument ABC implementation ─────────────────────────────────────────

    @property
    def currency(self) -> str:
        return self._domestic_currency

    @property
    def notional(self) -> float:
        return self._notional_foreign

    def price(self, curve: DiscountCurve) -> float:
        """
        FX Forward NPV in domestic currency (EUR).

        NPV = sign * N_f * (F_0 - F_t) * P_EUR(0,T)

        Positive NPV: domestic weakened -- exporter gains.
        Negative NPV: domestic strengthened -- exporter loses.
        """
        eur_handle = self._disc_handle_from_curve(curve)
        f0         = self._resolve_f0(eur_handle)
        ft         = self._market_forward(eur_handle)
        df_eur     = self._df_eur(eur_handle)
        return self._sign * self._notional_foreign * (f0 - ft) * df_eur

    def dv01(self, curve: DiscountCurve, bump: float = 0.0001) -> float:
        """
        Delta IR EUR -- change in NPV for 1bp parallel shift in EUR OIS curve.

        Analytical: sign * N_f * F_0 * T * P_EUR(0,T) * bump

        When EUR rates rise, F_t falls, (F_0 - F_t) rises -- exporter gains.
        Positive for exporter. FRTB GIRR bucket.
        """
        eur_handle = self._disc_handle_from_curve(curve)
        f0         = self._resolve_f0(eur_handle)
        df_eur     = self._df_eur(eur_handle)
        return self._sign * self._notional_foreign * f0 * self._T * df_eur * bump

    def duration(self, curve: DiscountCurve) -> float:
        """
        Tenor in years -- natural duration equivalent for FX forward.
        FX forwards have no modified duration in the bond sense.
        """
        return self._T

    def cash_flows(self) -> pd.DataFrame:
        """
        Single cash flow at maturity -- net settlement in foreign currency units.
        """
        return pd.DataFrame([{
            'date':   self._maturity_date.ISO(),
            'amount': self._sign * self._notional_foreign,
            'type':   'settlement',
        }])

    def key_rate_dv01(
        self,
        curve: DiscountCurve,
        tenors: list = None,
        bump: float = 0.0001,
    ) -> dict:
        """
        Delta IR per FRTB GIRR tenor vertex.

        FX forward IR sensitivity concentrates at the maturity tenor.
        All key rate DV01s are zero except at the maturity bucket.
        """
        vertices      = tenors or self._FRTB_VERTEX_LABELS
        ir_dv01       = self.dv01(curve, bump)

        return self._nearest_frtb_vertex(ir_dv01, self._T, vertices)


    # ── FXForward-specific methods ────────────────────────────────────────────

    def forward_rate(self, curve: DiscountCurve) -> float:
        """Current market CIP forward rate."""
        return self._market_forward(self._disc_handle_from_curve(curve))

    def delta_fx(
        self,
        curve: DiscountCurve,
        bump_pct: float = 0.01,
    ) -> float:
        """
        Delta FX -- change in NPV for 1% move in spot rate.

        Analytical: -sign * N_f * (F_0/S_0) * P_EUR(0,T) * bump_pct

        Negative for exporter -- loses when EUR strengthens (spot rises).
        FRTB SA FX risk bucket.
        """
        eur_handle = self._disc_handle_from_curve(curve)
        f0         = self._resolve_f0(eur_handle)
        df_eur     = self._df_eur(eur_handle)
        return -self._sign * self._notional_foreign * (f0 / self._spot_rate) * df_eur * bump_pct

    def delta_fx_numerical(
        self,
        curve: DiscountCurve,
        bump_pct: float = 0.01,
    ) -> float:
        """
        Delta FX -- numerical bump on spot rate.
        Reprices with spot * (1 + bump_pct).
        """
        eur_handle = self._disc_handle_from_curve(curve)
        f0         = self._resolve_f0(eur_handle)
        df_eur     = self._df_eur(eur_handle)
        df_usd     = self._df_usd()

        spot_up  = self._spot_rate * (1 + bump_pct)
        ft_up    = spot_up * df_eur / df_usd * np.exp(self._basis * self._T)
        npv_up   = self._sign * self._notional_foreign * (f0 - ft_up) * df_eur

        ft_base  = self._spot_rate * df_eur / df_usd * np.exp(self._basis * self._T)
        npv_base = self._sign * self._notional_foreign * (f0 - ft_base) * df_eur

        return npv_up - npv_base

    def delta_ir_usd(
        self,
        curve: DiscountCurve,
        bump: float = 0.0001,
    ) -> float:
        """
        Delta IR USD -- change in NPV for 1bp shift in USD rates.

        When USD rates rise, F_t rises, (F_0 - F_t) falls -- exporter loses.
        Positive for exporter (sign convention: loss is positive sensitivity).
        FRTB GIRR bucket (USD curve).
        """
        eur_handle = self._disc_handle_from_curve(curve)
        f0         = self._resolve_f0(eur_handle)
        df_eur     = self._df_eur(eur_handle)
        return self._sign * self._notional_foreign * f0 * self._T * df_eur * bump

    def hedge_effectiveness(
        self,
        portfolio_value_foreign: float,
        curve: DiscountCurve,
    ) -> dict:
        """
        Hedge effectiveness for a foreign currency portfolio.

        AIFMD II requirement: monthly rebalancing, effectiveness documented.

        Parameters
        ----------
        portfolio_value_foreign : float
            Portfolio value in foreign currency (e.g. USD equity portfolio).
        """
        portfolio_eur  = portfolio_value_foreign / self._spot_rate
        delta_unhedged = portfolio_eur * 0.01
        delta_hedge    = self.delta_fx(curve, bump_pct=0.01)
        delta_hedged   = delta_unhedged + delta_hedge
        effectiveness  = (1 - abs(delta_hedged) / abs(delta_unhedged)) * 100
        hedge_ratio    = abs(delta_hedge) / abs(delta_unhedged) * 100

        return {
            'portfolio_eur':     portfolio_eur,
            'delta_unhedged':    delta_unhedged,
            'delta_hedge':       delta_hedge,
            'delta_hedged':      delta_hedged,
            'effectiveness_pct': effectiveness,
            'hedge_ratio_pct':   hedge_ratio,
        }

    def basis_cost(self, curve: DiscountCurve) -> float:
        """
        Cost of cross-currency basis for this forward in EUR.

        Difference between CIP forward and basis-adjusted forward,
        expressed as EUR cost to the hedger.
        """
        eur_handle = self._disc_handle_from_curve(curve)
        df_eur     = self._df_eur(eur_handle)
        df_usd     = self._df_usd()

        f_cip = self._spot_rate * df_eur / df_usd
        f_adj = f_cip * np.exp(self._basis * self._T)
        return self._notional_foreign * (f_cip - f_adj) / self._spot_rate

    def describe(self) -> str:
        side   = 'Exporter' if self._exporter else 'Importer'
        f0_str = f'{self._f0:.4f}' if self._f0 is not None else 'at_inception'
        return (
            f"FXForward | {side} | "
            f"{self._foreign_currency}/{self._domestic_currency} | "
            f"Notional={self._notional_foreign:,.0f} {self._foreign_currency} | "
            f"Maturity={self._maturity_date.ISO()} | F0={f0_str}"
        )