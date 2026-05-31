# src/quant_risk/risk/xva.py

"""
XVA engine for a netting set of interest rate swaps.

Computes CVA, DVA, FVA, and MVA using:
  - MCSimulator for path generation and stochastic discount factors
  - Hull-White bond price formula for path-wise IRS repricing (no nested MC)
  - Flat hazard rate model calibrated to CDS spread for default probabilities
  - Single-bucket SIMM approximation for initial margin

Rate convention
---------------
All rates (K, sigma) and the initial short rate passed to MCSimulator are in
decimal throughout, consistent with the project-wide convention.

Design
------
XVAEngine is a standalone calculator. It does not inherit from Instrument,
DiscountCurve, or CentralBankClient. All dependencies (simulator, curve, HW
parameters) are injected at construction — no internal API calls or rate-path
re-simulation.

The exposure profile is computed lazily on the first call to exposure_profile()
and cached for the lifetime of the engine. All CVA/DVA/FVA/MVA methods share
the cached profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quant_risk.curves.base import DiscountCurve
from quant_risk.models.simulator import MCSimulator


@dataclass
class Trade:
    """
    Single interest rate swap in the netting set.

    Parameters
    ----------
    payment_dates : np.ndarray
        Payment dates in years, e.g. np.array([1.0, 2.0, 3.0, 4.0, 5.0]).
    year_fracs : np.ndarray
        Day-count fractions for each payment period (same length as
        payment_dates), e.g. np.ones(5) for annual 30/360.
    K : float
        Fixed coupon rate, decimal (e.g. 0.025 for 2.5%).
    notional : float
        Swap notional in base currency.
    is_payer : bool
        True = pay fixed, receive floating (payer IRS).
        False = receive fixed, pay floating (receiver IRS).
    """

    payment_dates: np.ndarray
    year_fracs: np.ndarray
    K: float
    notional: float
    is_payer: bool


class XVAEngine:
    """
    XVA engine for a netting set of interest rate swaps.

    Computes the full XVA stack (CVA, DVA, FVA, MVA) for a group of trades
    under a single ISDA Master Agreement netting clause. The exposure profile
    is computed on the net portfolio MTM — netting reduces all XVA components
    relative to the sum of individual trade XVAs.

    Parameters
    ----------
    simulator : MCSimulator
        Pre-built simulator containing simulated rate paths and SDF computation.
        Typically backed by HullWhiteProcess calibrated to the OIS curve.
    trades : list[Trade]
        Trades in the netting set. All trades share the same simulator and OIS
        curve.
    exp_dates : np.ndarray
        Evaluation dates for the exposure profile in years, e.g.
        np.arange(0.5, 5.5, 0.5) for semi-annual out to 5Y.
    kappa : float
        Hull-White mean reversion speed in 1/year. Used in the bond price
        formula for path-wise IRS repricing.
    sigma : float
        Hull-White instantaneous volatility, decimal/√year (e.g. 0.005 for 0.5%/√year).
    curve : DiscountCurve
        Market term structure (OISCurve). Used for discount factors and
        instantaneous forward rates in the bond price formula.
    """

    def __init__(
        self,
        simulator: MCSimulator,
        trades: list[Trade],
        exp_dates: np.ndarray,
        kappa: float,
        sigma: float,
        curve: DiscountCurve,
    ) -> None:
        self._sim       = simulator
        self._trades    = list(trades)
        self._exp_dates = np.asarray(exp_dates)
        self._kappa     = kappa
        self._sigma     = sigma
        self._curve     = curve

        # Lazy cache for the exposure profile and discounted NEE
        self._profile:  Optional[dict]       = None
        self._nee_disc: Optional[np.ndarray] = None

    # ── Private: Hull-White bond price formula ──────────────────────────────

    @staticmethod
    def _hw_B(tau: np.ndarray, kappa: float) -> np.ndarray:
        """B(τ) = (1 - e^{-κτ}) / κ. Returns 0 at τ=0."""
        return np.where(tau > 0, (1.0 - np.exp(-kappa * tau)) / kappa, 0.0)

    def _hw_bond_price(
        self,
        t: float,
        maturities: np.ndarray,
        r_t: np.ndarray,
    ) -> np.ndarray:
        """
        P^HW(t, T; r(t)) for a vector of maturities and paths.

        P^HW = P^M(0,T)/P^M(0,t) × exp[-B(T-t)(r(t)-f^M(0,t))/100 - V(t,T)]

        where V(t,T) = (σ_dec²/4κ) × B(T-t)² × (1-e^{-2κt}).

        Parameters
        ----------
        t          : simulation time in years
        maturities : shape (n_mat,)
        r_t        : shape (n_paths,) — simulated short rate at time t

        Returns
        -------
        np.ndarray : shape (n_paths, n_mat)
        """
        kappa, sigma, curve = self._kappa, self._sigma, self._curve
        tau     = maturities - t
        B_tau   = self._hw_B(tau, kappa)
        P_0T    = np.array([curve.discount(T) for T in maturities])
        P_0t    = curve.discount(t) if t > 1e-6 else 1.0
        # 1M forward window matches HullWhiteProcess._theta() convention
        dt_fd   = 1.0 / 12.0
        t_lo    = max(t - dt_fd, dt_fd)
        f_0t    = curve.forward_rate(t_lo, t_lo + dt_fd)
        var_adj  = (sigma**2 / (4.0 * kappa)) * B_tau**2 * (1.0 - np.exp(-2.0 * kappa * t))
        rate_dev = (r_t[:, None] - f_0t) * B_tau[None, :]
        return (P_0T / P_0t)[None, :] * np.exp(-rate_dev - var_adj[None, :])

    # ── Private: trade-level MTM and DV01 ──────────────────────────────────

    def _trade_mtm(self, trade: Trade, t: float) -> np.ndarray:
        """
        IRS MTM at time t along each path.

        V_payer(t)   = N × [(1-P^HW(t,T_n)) - K × Σ δᵢ P^HW(t,Tᵢ)]
        V_receiver   = -V_payer
        """
        remaining = trade.payment_dates[trade.payment_dates > t]
        yf_r      = trade.year_fracs[trade.payment_dates > t]
        if len(remaining) == 0:
            return np.zeros(self._sim.n_paths)
        t_idx  = min(int(round(t / self._sim.dt)), self._sim.n_steps)
        r_t    = self._sim.paths[:, t_idx]
        P_ti   = self._hw_bond_price(t, remaining, r_t)
        fl     = 1.0 - P_ti[:, -1]
        fx     = trade.K * (yf_r * P_ti).sum(axis=1)
        payer  = trade.notional * (fl - fx)
        return payer if trade.is_payer else -payer

    def _trade_dv01(self, trade: Trade, t: float) -> np.ndarray:
        """
        DV01 = notional/10,000 × remaining annuity(t).

        Sensitivity of the fixed leg to a 1 bp parallel shift in rates.
        Signed: positive for payer (fixed-leg receiver benefits from rate rise),
        negative for receiver (opposite sign).
        """
        remaining = trade.payment_dates[trade.payment_dates > t]
        yf_r      = trade.year_fracs[trade.payment_dates > t]
        if len(remaining) == 0:
            return np.zeros(self._sim.n_paths)
        t_idx  = min(int(round(t / self._sim.dt)), self._sim.n_steps)
        r_t    = self._sim.paths[:, t_idx]
        P_ti   = self._hw_bond_price(t, remaining, r_t)
        annuity = (yf_r * P_ti).sum(axis=1)
        signed  = trade.notional / 10_000.0 * annuity
        return signed if trade.is_payer else -signed

    # ── Private: hazard rate helpers ────────────────────────────────────────

    @staticmethod
    def _hazard_rate(cds_spread_bps: float, recovery: float) -> float:
        """h = S_CDS / ((1-R) × 10,000). Flat hazard rate approximation."""
        return (cds_spread_bps / 10_000.0) / (1.0 - recovery)

    def _default_probs(self, h: float) -> np.ndarray:
        """
        Marginal default probability PD(tᵢ₋₁, tᵢ) = e^{-h tᵢ₋₁} - e^{-h tᵢ}
        for each evaluation period.
        """
        dates  = self._exp_dates
        t_prev = np.concatenate([[0.0], dates[:-1]])
        return np.exp(-h * t_prev) - np.exp(-h * dates)

    # ── Exposure profile (lazy, cached) ────────────────────────────────────

    def exposure_profile(self) -> dict:
        """
        Net portfolio exposure profile, computed lazily and cached.

        The net MTM is V_net(t) = Σ_k V_k(t) across all trades in the netting
        set. Netting reduces EE relative to the sum of individual trade EEs.

        Also populates the internal _nee_disc cache (discounted NEE) used by
        dva() and fva().

        Returns
        -------
        dict — same schema as MCSimulator.exposure_profile():
            dates, mtm, EE, NEE, PFE, EE_disc, EPE
        """
        if self._profile is not None:
            return self._profile

        def mtm_fn(paths: np.ndarray, t: float) -> np.ndarray:
            return sum(self._trade_mtm(tr, t) for tr in self._trades)

        self._profile  = self._sim.exposure_profile(mtm_fn, self._exp_dates)
        disc_matrix    = np.column_stack(
            [self._sim.sdf(t) for t in self._exp_dates]
        )
        self._nee_disc = (
            disc_matrix * np.minimum(self._profile['mtm'], 0)
        ).mean(axis=0)

        return self._profile

    # ── CVA ────────────────────────────────────────────────────────────────

    def cva(self, cds_spread_bps: float, recovery: float) -> dict:
        """
        CVA = (1-R) Σᵢ EE_disc(tᵢ) × PD(tᵢ₋₁, tᵢ).

        Parameters
        ----------
        cds_spread_bps : counterparty CDS spread in bps/year
        recovery       : counterparty recovery rate as a fraction (0.0–1.0)

        Returns
        -------
        dict:
            CVA           : scalar CVA in base currency
            h             : hazard rate (1/year)
            pd_period     : marginal default probabilities per period
            cva_by_period : CVA contribution per period (sums to CVA)
        """
        profile   = self.exposure_profile()
        h         = self._hazard_rate(cds_spread_bps, recovery)
        pd_period = self._default_probs(h)
        cva_p     = (1.0 - recovery) * profile['EE_disc'] * pd_period
        return {
            'CVA'           : float(cva_p.sum()),
            'h'             : h,
            'pd_period'     : pd_period,
            'cva_by_period' : cva_p,
        }

    # ── DVA ────────────────────────────────────────────────────────────────

    def dva(self, own_cds_spread_bps: float, own_recovery: float) -> dict:
        """
        DVA = (1-R_own) Σᵢ |NEE_disc(tᵢ)| × PD_own(tᵢ₋₁, tᵢ).

        Parameters
        ----------
        own_cds_spread_bps : bank's own CDS spread in bps/year
        own_recovery       : bank's recovery rate as a fraction

        Returns
        -------
        dict: DVA (scalar), h_own, pd_own, dva_by_period
        """
        self.exposure_profile()
        h_own  = self._hazard_rate(own_cds_spread_bps, own_recovery)
        pd_own = self._default_probs(h_own)
        dva_p  = (1.0 - own_recovery) * np.abs(self._nee_disc) * pd_own
        return {
            'DVA'           : float(dva_p.sum()),
            'h_own'         : h_own,
            'pd_own'        : pd_own,
            'dva_by_period' : dva_p,
        }

    # ── FVA ────────────────────────────────────────────────────────────────

    def fva(
        self,
        funding_spread_bps: float,
        invest_spread_bps: float,
        csa_type: str,
    ) -> dict:
        """
        FVA = FCA - FBA.

        FCA = s_f Σᵢ Δtᵢ EE_disc(tᵢ)       — cost when V > 0
        FBA = s_r Σᵢ Δtᵢ |NEE_disc(tᵢ)|   — benefit when V < 0

        Parameters
        ----------
        funding_spread_bps : bank's unsecured funding spread above OIS (bps/year)
        invest_spread_bps  : reinvestment spread earned above OIS (bps/year)
        csa_type           : collateral arrangement:
                             'two_way'         → FVA = 0 (full VM both ways)
                             'one_way_post'    → FBA = 0 (we post, they don't)
                             'one_way_receive' → FCA and FBA both apply
                             'none'            → no CSA, uncollateralised

        Returns
        -------
        dict: FVA, FCA, FBA (scalars), fca_by_period, fba_by_period (arrays)
        """
        profile = self.exposure_profile()
        s_f     = funding_spread_bps / 10_000.0
        s_r     = invest_spread_bps  / 10_000.0
        dates   = self._exp_dates
        t_prev  = np.concatenate([[0.0], dates[:-1]])
        dt_i    = dates - t_prev

        if csa_type == 'two_way':
            z = np.zeros(len(dates))
            return {'FVA': 0.0, 'FCA': 0.0, 'FBA': 0.0,
                    'fca_by_period': z, 'fba_by_period': z}

        fca_p = s_f * dt_i * profile['EE_disc']
        fba_p = (s_r * dt_i * np.abs(self._nee_disc)
                 if csa_type != 'one_way_post' else np.zeros(len(dates)))
        fca   = float(fca_p.sum())
        fba   = float(fba_p.sum())
        return {'FVA': fca - fba, 'FCA': fca, 'FBA': fba,
                'fca_by_period': fca_p, 'fba_by_period': fba_p}

    # ── MVA ────────────────────────────────────────────────────────────────

    def mva(
        self,
        risk_weight_bps: float,
        mpor_days: int,
        im_funding_spread_bps: float,
        mpor_base_days: int = 10,
    ) -> dict:
        """
        MVA = s_IM Σᵢ Δtᵢ EIM_disc(tᵢ).

        SIMM IM (single-bucket) = RW × |net DV01| × sqrt(mpor_days / mpor_base_days)

        DV01s are summed (with sign) across trades before taking the absolute
        value — this is the SIMM intra-bucket netting benefit. Trades with
        opposing DV01 signs (e.g. payer + receiver) reduce net IM and hence MVA.

        Parameters
        ----------
        risk_weight_bps    : SIMM risk weight for the dominant tenor bucket
                             (bps). Typical EUR IR delta, 5Y: 64 bps.
        mpor_days          : margin period of risk (business days).
                             10 = bilateral SIMM; 5 = cleared (LCH/Eurex).
        im_funding_spread_bps : funding spread on IM above OIS (bps/year).
        mpor_base_days     : reference MPOR for the risk weight (default 10).

        Returns
        -------
        dict: MVA (scalar), eim_disc (array), mva_by_period (array)
        """
        rw       = risk_weight_bps / 10_000.0
        s_im     = im_funding_spread_bps / 10_000.0
        mpor_adj = np.sqrt(mpor_days / mpor_base_days)
        dates    = self._exp_dates
        t_prev   = np.concatenate([[0.0], dates[:-1]])
        dt_i     = dates - t_prev

        eim_disc = np.zeros(len(dates))
        for j, t in enumerate(dates):
            # Net DV01 across all trades (signed sum → partial cancellation)
            net_dv01 = sum(self._trade_dv01(tr, t) for tr in self._trades)
            im_net   = rw * np.abs(net_dv01) * mpor_adj
            eim_disc[j] = (self._sim.sdf(t) * im_net).mean()

        mva_p = s_im * dt_i * eim_disc
        return {
            'MVA'           : float(mva_p.sum()),
            'eim_disc'      : eim_disc,
            'mva_by_period' : mva_p,
        }

    # ── Full XVA report ─────────────────────────────────────────────────────

    def xva(
        self,
        cds_spread_bps: float,
        recovery: float,
        own_cds_spread_bps: float,
        own_recovery: float,
        funding_spread_bps: float,
        invest_spread_bps: float,
        csa_type: str,
        risk_weight_bps: float,
        mpor_days: int,
        im_funding_spread_bps: float,
    ) -> dict:
        """
        Full XVA report for the netting set.

        V^fair_value = V^risk_free - CVA + DVA - FVA - MVA

        All components share the cached exposure profile — exposure_profile()
        is only run once regardless of how many times xva() is called.

        Parameters
        ----------
        cds_spread_bps         : counterparty CDS spread (bps/year)
        recovery               : counterparty recovery rate
        own_cds_spread_bps     : bank's own CDS spread (bps/year)
        own_recovery           : bank's own recovery rate
        funding_spread_bps     : unsecured funding spread above OIS (bps/year)
        invest_spread_bps      : reinvestment spread above OIS (bps/year)
        csa_type               : 'none' | 'one_way_post' | 'one_way_receive' | 'two_way'
        risk_weight_bps        : SIMM risk weight (bps)
        mpor_days              : margin period of risk (business days)
        im_funding_spread_bps  : IM funding spread above OIS (bps/year)

        Returns
        -------
        dict:
            CVA, DVA, BCVA (= CVA - DVA), FVA, FCA, FBA, MVA,
            total_XVA (= CVA - DVA + FVA + MVA),
            details (nested dicts from each component method)
        """
        cva_r = self.cva(cds_spread_bps, recovery)
        dva_r = self.dva(own_cds_spread_bps, own_recovery)
        fva_r = self.fva(funding_spread_bps, invest_spread_bps, csa_type)
        mva_r = self.mva(risk_weight_bps, mpor_days, im_funding_spread_bps)

        bcva  = cva_r['CVA'] - dva_r['DVA']
        total = cva_r['CVA'] - dva_r['DVA'] + fva_r['FVA'] + mva_r['MVA']

        return {
            'CVA'       : cva_r['CVA'],
            'DVA'       : dva_r['DVA'],
            'BCVA'      : bcva,
            'FVA'       : fva_r['FVA'],
            'FCA'       : fva_r['FCA'],
            'FBA'       : fva_r['FBA'],
            'MVA'       : mva_r['MVA'],
            'total_XVA' : total,
            'details'   : {'cva': cva_r, 'dva': dva_r, 'fva': fva_r, 'mva': mva_r},
        }

    # ── Sensitivities ───────────────────────────────────────────────────────

    def cs01(
        self,
        cds_spread_bps: float,
        recovery: float,
        bump_bps: float = 1.0,
    ) -> float:
        """
        CVA sensitivity to a +bump_bps shift in counterparty CDS spread.

        CS01 = CVA(S + bump) - CVA(S)

        Uses the cached exposure profile — no re-simulation.

        Parameters
        ----------
        cds_spread_bps : base CDS spread (bps)
        recovery       : recovery rate
        bump_bps       : spread bump size (default 1 bp)

        Returns
        -------
        float : change in CVA in base currency units per bump_bps
        """
        base   = self.cva(cds_spread_bps, recovery)['CVA']
        bumped = self.cva(cds_spread_bps + bump_bps, recovery)['CVA']
        return bumped - base

    def dva01(
        self,
        own_cds_spread_bps: float,
        own_recovery: float,
        bump_bps: float = 1.0,
    ) -> float:
        """
        DVA sensitivity to a +bump_bps shift in own CDS spread.

        DVA01 = DVA(S_own + bump) - DVA(S_own)
        """
        base   = self.dva(own_cds_spread_bps, own_recovery)['DVA']
        bumped = self.dva(own_cds_spread_bps + bump_bps, own_recovery)['DVA']
        return bumped - base

    def describe(self) -> str:
        total_notl = sum(tr.notional for tr in self._trades)
        return (
            f"XVAEngine | {len(self._trades)} trade(s) | "
            f"notional={total_notl:,.0f} | "
            f"κ={self._kappa:.4f} | σ={self._sigma:.4%}/√yr | "
            f"n_dates={len(self._exp_dates)}"
        )
