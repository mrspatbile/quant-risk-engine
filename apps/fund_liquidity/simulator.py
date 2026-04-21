"""
simulator.py -- Fund Liquidity Risk Simulator
QRE-38 | Sprint 4 | Quant Risk Engine

Reusable engine for AIFMD liquidity risk analysis.
Supports multi-asset, credit, leveraged, and real estate AIFs.

Regulatory basis:
    Delegated Regulation 231/2013, Articles 46-49
    AIFMD II Directive 2024/927/EU, Annex V
    ESMA34-671404336-1364 (Guidelines on LMTs, April 2025)
    ESMA/2013/232 (Annex IV reporting)
"""

import numpy as np
import pandas as pd

# ── AIFMD Annex IV liquidity bucket definitions ───────────────────────────────
BUCKET_LABELS = [
    '1 day',
    '2-7 days',
    '8-30 days',
    '31-90 days',
    '91-180 days',
    '181-365 days',
    '> 1 year',
]

BUCKET_BINS = [0, 1, 7, 30, 90, 180, 365, np.inf]

# occupancy thresholds for RE funds -- maps occupancy rate to liquidation days
# high occupancy = easier to sell, lower days
RE_OCCUPANCY_BUCKETS = [
    (0.90, 91),    # >= 90% occupied: 91-180 days bucket
    (0.75, 181),   # >= 75% occupied: 181-365 days bucket
    (0.00, 366),   # below 75%:       > 1 year bucket
]

PARTICIPATION_RATE = 0.10   # 10% of daily market volume -- standard convention


class FundLiquiditySimulator:
    """
    AIFMD liquidity risk simulator for open-ended AIFs.

    Parameters
    ----------
    positions : pd.DataFrame
        Position-level data. Required columns: name, asset_class, mv, daily_volume.
        Optional columns: occupancy_rate (RE funds), notional, delta (leveraged funds).
    investors : pd.DataFrame
        Investor base. Required columns: type, share, normal_redemption, stress_redemption.
    nav : float
        Fund NAV in EUR. If None, derived from sum of position MVs.
    notice_days : int
        Redemption notice period in business days. Default 30.
    fund_name : str
        Fund name for display purposes.
    """

    def __init__(
        self,
        positions: pd.DataFrame,
        investors: pd.DataFrame,
        nav: float = None,
        notice_days: int = 30,
        fund_name: str = 'AIF',
    ):
        self.positions    = positions.copy()
        self.investors    = investors.copy()
        self.notice_days  = notice_days
        self.fund_name    = fund_name

        # cast daily_volume to float to avoid LossySetitemError in stress scenarios
        self.positions['daily_volume'] = self.positions['daily_volume'].astype(float)

        # derive NAV from positions if not supplied
        self.nav = float(nav) if nav is not None else float(self.positions['mv'].sum())

        self.positions['pct_nav'] = self.positions['mv'] / self.nav * 100

        # run bucketing on init
        self._bucket_positions()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _days_to_liquidate(self, mv: float, daily_volume: float) -> float:
        """10% participation rate convention."""
        if daily_volume <= 0:
            return np.inf
        return np.ceil(mv / (PARTICIPATION_RATE * daily_volume))

    def _days_to_liquidate_re(self, mv: float, daily_volume: float, occupancy: float) -> float:
        """
        For direct RE positions: override days based on occupancy rate.
        Listed REITs use the standard participation rate convention.
        """
        if daily_volume > 0:
            return self._days_to_liquidate(mv, daily_volume)
        # direct property -- occupancy drives liquidation horizon
        for threshold, days in RE_OCCUPANCY_BUCKETS:
            if occupancy >= threshold:
                return days
        return 366   # fallback: > 1 year

    def _bucket_positions(self):
        """Assign each position to an AIFMD Annex IV liquidity bucket."""
        has_occupancy = 'occupancy_rate' in self.positions.columns

        def get_days(row):
            if has_occupancy and pd.notna(row.get('occupancy_rate')):
                return self._days_to_liquidate_re(
                    row['mv'], row['daily_volume'], row['occupancy_rate']
                )
            return self._days_to_liquidate(row['mv'], row['daily_volume'])

        self.positions['days_to_liq'] = self.positions.apply(get_days, axis=1)
        self.positions['bucket_label'] = pd.cut(
            self.positions['days_to_liq'],
            bins=BUCKET_BINS,
            labels=BUCKET_LABELS,
        )

    def _aggregate_buckets(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate positions to bucket level.
        Always returns all 7 buckets -- empty buckets show as zero.
        """
        agg = (
            positions.groupby('bucket_label', observed=False)
            .agg(mv=('mv', 'sum'), pct_nav=('pct_nav', 'sum'))
            .reset_index()
        )
        agg['cumulative_pct'] = agg['pct_nav'].cumsum()
        return agg

    # ── public API ────────────────────────────────────────────────────────────

    def bucket_summary(self) -> pd.DataFrame:
        """Return bucket-level liquidity profile for the base portfolio."""
        return self._aggregate_buckets(self.positions)

    def redemption_profile(self, scenario: str = 'normal') -> dict:
        """
        Compute total redemption liability under normal or stress scenario.

        Returns dict with total EUR amount and % of NAV per investor type.
        """
        col = f'{scenario}_redemption'
        inv = self.investors.copy()
        inv['aum']            = inv['share'] * self.nav
        inv['redemption_eur'] = inv['aum'] * inv[col]
        total                 = inv['redemption_eur'].sum()
        return {
            'detail':     inv[['type', 'aum', 'redemption_eur']],
            'total_eur':  total,
            'total_pct':  total / self.nav * 100,
        }

    def liquidity_coverage(self) -> pd.DataFrame:
        """
        Compute fund LCR under normal and stress scenarios.

        LCR = liquid assets within notice period / redemption liability.
        Notice period bucket is determined by self.notice_days.
        """
        # determine which buckets fall within the notice period
        notice_cutoff  = self.notice_days
        liquid_buckets = [
            b for b, edge in zip(BUCKET_LABELS, BUCKET_BINS[1:])
            if edge <= notice_cutoff or (notice_cutoff > BUCKET_BINS[i] and notice_cutoff <= edge)
            for i in [BUCKET_LABELS.index(b)]
        ]
        # simpler: buckets 1-3 cover 1d, 2-7d, 8-30d -- always within 30bd notice
        liquid_buckets = ['1 day', '2-7 days', '8-30 days']

        available = self.positions[
            self.positions['bucket_label'].isin(liquid_buckets)
        ]['mv'].sum()

        normal = self.redemption_profile('normal')
        stress = self.redemption_profile('stress')

        return pd.DataFrame([
            {
                'scenario':          'Normal',
                'available_eur':     available,
                'available_pct_nav': available / self.nav * 100,
                'redemption_eur':    normal['total_eur'],
                'redemption_pct_nav':normal['total_pct'],
                'lcr':               available / normal['total_eur'],
                'pass':              available >= normal['total_eur'],
            },
            {
                'scenario':          'Stress',
                'available_eur':     available,
                'available_pct_nav': available / self.nav * 100,
                'redemption_eur':    stress['total_eur'],
                'redemption_pct_nav':stress['total_pct'],
                'lcr':               available / stress['total_eur'],
                'pass':              available >= stress['total_eur'],
            },
        ])

    def stress_test(
        self,
        volume_haircut_equity: float = 0.50,
        volume_haircut_hy: float     = 0.70,
        redemption_scenario: str     = 'stress',
    ) -> dict:
        """
        Apply asset-side volume haircuts and compute LCR under stressed conditions.

        Parameters
        ----------
        volume_haircut_equity : float
            Fraction by which daily volume is reduced for equity and IG bonds.
        volume_haircut_hy : float
            Fraction by which daily volume is reduced for HY bonds and CLOs.
        redemption_scenario : str
            'normal' or 'stress'.

        Returns dict with shocked bucket summary and LCR.
        """
        shocked = self.positions.copy()

        equity_mask = shocked['asset_class'].str.contains(
            'Equity|Govt Bond|Corp Bond -- IG|Cash|Listed|REIT|Futures|Option', na=False
        )
        hy_mask = shocked['asset_class'].str.contains(
            'Corp Bond -- HY|CLO|Private Credit|RE Debt|Repo', na=False
        )

        shocked.loc[equity_mask, 'daily_volume'] *= (1 - volume_haircut_equity)
        shocked.loc[hy_mask,     'daily_volume'] *= (1 - volume_haircut_hy)

        # re-bucket with shocked volumes
        shocked['days_to_liq'] = shocked.apply(
            lambda r: self._days_to_liquidate(r['mv'], r['daily_volume']), axis=1
        )
        shocked['bucket_label'] = pd.cut(
            shocked['days_to_liq'], bins=BUCKET_BINS, labels=BUCKET_LABELS
        )

        bucket_summary = self._aggregate_buckets(shocked)

        liquid_buckets  = ['1 day', '2-7 days', '8-30 days']
        available       = shocked[shocked['bucket_label'].isin(liquid_buckets)]['mv'].sum()
        redemption      = self.redemption_profile(redemption_scenario)

        return {
            'bucket_summary':  bucket_summary,
            'available_eur':   available,
            'available_pct':   available / self.nav * 100,
            'redemption_eur':  redemption['total_eur'],
            'redemption_pct':  redemption['total_pct'],
            'lcr':             available / redemption['total_eur'],
            'pass':            available >= redemption['total_eur'],
        }

    def annex_iv(
        self,
        volume_haircut_equity: float = 0.50,
        volume_haircut_hy: float     = 0.70,
    ) -> pd.DataFrame:
        """
        Build AIFMD Annex IV liquidity section.

        Returns DataFrame with base and stressed portfolio profile
        plus investor redemption rights profile.
        """
        base    = self.bucket_summary()[['bucket_label', 'pct_nav']].copy()
        base.columns = ['Liquidation horizon', 'pct portfolio (base)']

        stressed = self.stress_test(volume_haircut_equity, volume_haircut_hy)
        stressed_pct = stressed['bucket_summary']['pct_nav'].values

        # investor redemption profile -- all equity redeemable within notice period bucket
        notice_bucket = '8-30 days' if self.notice_days <= 30 else '31-90 days'
        liability_pct = [
            100.0 if label == notice_bucket else 0.0
            for label in BUCKET_LABELS
        ]

        base['pct portfolio (stressed)'] = stressed_pct
        base['pct equity redeemable']    = liability_pct
        base['liquidity gap (base)']     = (
            base['pct portfolio (base)'] - base['pct equity redeemable']
        )
        return base

    def lmt_simulation(
        self,
        n_months: int                = 12,
        stress_months: dict          = None,
        gate_threshold: float        = 0.10,
        swing_threshold: float       = 0.05,
        contagion_multiplier: float  = 1.50,
        suspension_backlog: float    = 0.50,
        suspension_consec: int       = 2,
        random_seed: int             = 42,
    ) -> pd.DataFrame:
        """
        Simulate monthly redemption flows and LMT triggers over n_months.

        Parameters
        ----------
        n_months : int
            Number of months to simulate.
        stress_months : dict
            {month_index: redemption_rate} to inject stress. E.g. {4: 0.14, 9: 0.22}.
        gate_threshold : float
            Gate activates when redemptions exceed this fraction of NAV.
        swing_threshold : float
            Swing pricing activates when net flows exceed this fraction of NAV.
        contagion_multiplier : float
            New requests amplified by this factor the month after a gate fires.
        suspension_backlog : float
            Suspension triggers when backlog exceeds this fraction of liquid sleeve.
        suspension_consec : int
            Suspension requires gate to be active for this many consecutive months.
        random_seed : int
            Seed for reproducibility.

        Returns
        -------
        pd.DataFrame with monthly simulation results.
        """
        np.random.seed(random_seed)

        if stress_months is None:
            stress_months = {4: 0.14, 9: 0.22}

        base_rate   = self.redemption_profile('normal')['total_pct'] / 100
        normal_draw = np.random.lognormal(
            mean=np.log(base_rate) - 0.3,
            sigma=0.5,
            size=n_months,
        )
        for idx, rate in stress_months.items():
            if idx < n_months:
                normal_draw[idx] = rate

        months = [f'M{i+1}' for i in range(n_months)]

        # sleeve tracking
        liquid_nav   = float(
            self.positions[
                self.positions['bucket_label'].isin(['1 day', '2-7 days', '8-30 days'])
            ]['mv'].sum()
        )
        illiquid_nav = float(self.nav - liquid_nav)
        nav          = liquid_nav + illiquid_nav
        backlog      = 0.0
        gate_consec  = 0

        rows = []
        for i, (month, gross) in enumerate(zip(months, normal_draw)):
            # contagion amplification
            if i > 0 and rows[i-1]['gate_active']:
                gross = gross * contagion_multiplier

            nav_before   = nav
            requested    = gross * nav_before + backlog
            gate_limit   = gate_threshold * nav_before
            paid         = min(requested, gate_limit)
            backlog      = requested - paid

            # liquid sleeve depletes first -- illiquid stays fixed
            liquid_nav   = max(0.0, liquid_nav - paid)
            nav          = liquid_nav + illiquid_nav

            gate_active  = requested > gate_limit
            gate_consec  = gate_consec + 1 if gate_active else 0

            suspension_active = (
                gate_consec >= suspension_consec
                and liquid_nav > 0
                and backlog > suspension_backlog * liquid_nav
            )

            rows.append({
                'month':                month,
                'nav_eur':              nav,
                'liquid_nav_eur':       liquid_nav,
                'illiquid_nav_eur':     illiquid_nav,
                'liquid_pct_nav':       liquid_nav / nav * 100 if nav > 0 else 0.0,
                'gross_redemption_pct': gross * 100,
                'requested_eur':        requested,
                'requested_pct':        requested / nav_before * 100,
                'gate_limit_pct':       gate_limit / nav_before * 100,
                'paid_eur':             paid,
                'backlog_eur':          backlog,
                'backlog_pct_nav':      backlog / nav * 100 if nav > 0 else 0.0,
                'gate_active':          gate_active,
                'gate_consecutive':     gate_consec,
                'swing_active':         gross > swing_threshold,
                'swing_factor_bps':     50.0 if gross > swing_threshold else 0.0,
                'suspension_active':    suspension_active,
            })

        return pd.DataFrame(rows)

    def leverage(self) -> dict:
        """
        Compute AIFMD gross and commitment leverage.
        Only meaningful for leveraged funds with notional and delta columns.
        Falls back to simple MV/NAV if those columns are absent.
        """
        pos = self.positions

        if 'notional' in pos.columns and 'delta' in pos.columns:
            gross_exposure      = (pos['notional'].abs() * pos['delta'].abs()).sum()
            # commitment: net notional where offsets allowed -- simple abs sum for now
            commitment_exposure = pos['notional'].abs().sum()
        else:
            gross_exposure      = pos['mv'].abs().sum()
            commitment_exposure = pos['mv'].abs().sum()

        return {
            'gross_leverage':      gross_exposure / self.nav,
            'commitment_leverage': commitment_exposure / self.nav,
            'gross_exposure_eur':  gross_exposure,
            'nav':                 self.nav,
        }

    def summary(self) -> dict:
        """Return top-level KPIs for dashboard header."""
        lcr_df  = self.liquidity_coverage()
        buckets = self.bucket_summary()
        liquid_pct = buckets[buckets['bucket_label'].isin(
            ['1 day', '2-7 days', '8-30 days']
        )]['pct_nav'].sum()

        return {
            'fund_name':       self.fund_name,
            'nav_eur':         self.nav,
            'liquid_pct':      liquid_pct,
            'illiquid_pct':    100 - liquid_pct,
            'lcr_normal':      lcr_df.loc[lcr_df['scenario'] == 'Normal', 'lcr'].values[0],
            'lcr_stress':      lcr_df.loc[lcr_df['scenario'] == 'Stress',  'lcr'].values[0],
            'n_positions':     len(self.positions),
        }
