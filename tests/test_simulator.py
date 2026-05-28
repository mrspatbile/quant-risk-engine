"""
tests/test_simulator.py -- FundLiquiditySimulator test suite
QRE-83 / QRE-84 / QRE-85 / QRE-86
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "fund_liquidity"))
from simulator import FundLiquiditySimulator, BUCKET_LABELS


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def positions():
    return pd.DataFrame([
        {"name": "Govt Bond", "asset_class": "Govt Bond",      "mv": 3_000_000, "daily_volume": 50_000_000},
        {"name": "Equity",    "asset_class": "Equity",          "mv": 5_000_000, "daily_volume": 10_000_000},
        {"name": "HY Bond",   "asset_class": "Corp Bond -- HY", "mv": 2_000_000, "daily_volume":    200_000},
        {"name": "CLO",       "asset_class": "CLO",             "mv": 1_000_000, "daily_volume":      5_000},
    ])


@pytest.fixture
def investors():
    return pd.DataFrame([
        {"type": "Retail",        "share": 0.40, "normal_redemption": 0.05, "stress_redemption": 0.15},
        {"type": "Institutional", "share": 0.60, "normal_redemption": 0.03, "stress_redemption": 0.10},
    ])


@pytest.fixture
def sim(positions, investors):
    return FundLiquiditySimulator(positions, investors, nav=11_000_000, fund_name="Test Fund")


@pytest.fixture
def re_positions():
    return pd.DataFrame([
        {"name": "Office",      "asset_class": "Direct RE",  "mv": 5_000_000, "daily_volume": 0,         "occupancy_rate": 0.95},
        {"name": "Retail",      "asset_class": "Direct RE",  "mv": 3_000_000, "daily_volume": 0,         "occupancy_rate": 0.80},
        {"name": "Vacant",      "asset_class": "Direct RE",  "mv": 2_000_000, "daily_volume": 0,         "occupancy_rate": 0.50},
        {"name": "Listed REIT", "asset_class": "Listed REIT","mv": 1_000_000, "daily_volume": 5_000_000, "occupancy_rate": 0.85},
    ])


@pytest.fixture
def re_sim(re_positions, investors):
    return FundLiquiditySimulator(re_positions, investors)


# ── TestBucketAssignment ──────────────────────────────────────────────────────

class TestBucketAssignment:

    def test_govt_bond_in_1_day_bucket(self, sim):
        # ceil(3M / (0.10 * 50M)) = ceil(0.6) = 1 → "1 day"
        bucket = sim.positions.loc[sim.positions['name'] == 'Govt Bond', 'bucket_label'].iloc[0]
        assert str(bucket) == '1 day'

    def test_equity_in_2_7_day_bucket(self, sim):
        # ceil(5M / (0.10 * 10M)) = ceil(5) = 5 → "2-7 days"
        bucket = sim.positions.loc[sim.positions['name'] == 'Equity', 'bucket_label'].iloc[0]
        assert str(bucket) == '2-7 days'

    def test_hy_bond_in_91_180_day_bucket(self, sim):
        # ceil(2M / (0.10 * 200K)) = ceil(100) = 100 → "91-180 days"
        bucket = sim.positions.loc[sim.positions['name'] == 'HY Bond', 'bucket_label'].iloc[0]
        assert str(bucket) == '91-180 days'

    def test_clo_in_gt_1_year_bucket(self, sim):
        # ceil(1M / (0.10 * 5K)) = ceil(2000) = 2000 → "> 1 year"
        bucket = sim.positions.loc[sim.positions['name'] == 'CLO', 'bucket_label'].iloc[0]
        assert str(bucket) == '> 1 year'

    def test_zero_daily_volume_returns_inf(self, sim):
        assert sim._days_to_liquidate(1_000_000, 0) == np.inf

    def test_participation_rate_applied(self, sim):
        # 10% convention: ceil(1M / (0.10 * 5M)) = 2 days
        assert sim._days_to_liquidate(1_000_000, 5_000_000) == 2.0

    def test_bucket_summary_always_has_7_rows(self, sim):
        assert len(sim.bucket_summary()) == 7

    def test_bucket_summary_has_all_labels(self, sim):
        labels = sim.bucket_summary()['bucket_label'].astype(str).tolist()
        assert labels == BUCKET_LABELS

    def test_pct_nav_sums_to_100(self, sim):
        assert abs(sim.bucket_summary()['pct_nav'].sum() - 100.0) < 0.01


# ── TestREFundLiquidation ─────────────────────────────────────────────────────

class TestREFundLiquidation:

    def test_high_occupancy_maps_to_91_180_days_bucket(self, re_sim):
        # occupancy 0.95 >= 0.90, daily_volume=0 → 91 days
        bucket = re_sim.positions.loc[re_sim.positions['name'] == 'Office', 'bucket_label'].iloc[0]
        assert str(bucket) == '91-180 days'

    def test_medium_occupancy_maps_to_181_365_days_bucket(self, re_sim):
        # occupancy 0.80: 0.75 <= 0.80 < 0.90 → 181 days
        bucket = re_sim.positions.loc[re_sim.positions['name'] == 'Retail', 'bucket_label'].iloc[0]
        assert str(bucket) == '181-365 days'

    def test_low_occupancy_maps_to_gt_1_year_bucket(self, re_sim):
        # occupancy 0.50 < 0.75 → 366 days
        bucket = re_sim.positions.loc[re_sim.positions['name'] == 'Vacant', 'bucket_label'].iloc[0]
        assert str(bucket) == '> 1 year'

    def test_listed_reit_uses_participation_rate(self, re_sim):
        # daily_volume > 0 overrides occupancy → ceil(1M / (0.10 * 5M)) = 2 → "2-7 days"
        bucket = re_sim.positions.loc[re_sim.positions['name'] == 'Listed REIT', 'bucket_label'].iloc[0]
        assert str(bucket) == '2-7 days'

    def test_re_occupancy_thresholds_direct(self, sim):
        assert sim._days_to_liquidate_re(5_000_000, 0, 0.95) == 91
        assert sim._days_to_liquidate_re(5_000_000, 0, 0.80) == 181
        assert sim._days_to_liquidate_re(5_000_000, 0, 0.50) == 366


# ── TestLCR ───────────────────────────────────────────────────────────────────

class TestLCR:

    def test_lcr_returns_two_rows(self, sim):
        assert len(sim.liquidity_coverage()) == 2

    def test_lcr_columns(self, sim):
        expected = [
            'scenario', 'available_eur', 'available_pct_nav',
            'redemption_eur', 'redemption_pct_nav', 'lcr', 'pass',
        ]
        assert all(c in sim.liquidity_coverage().columns for c in expected)

    def test_stress_lcr_below_normal_lcr(self, sim):
        lcr    = sim.liquidity_coverage()
        normal = lcr.loc[lcr['scenario'] == 'Normal', 'lcr'].values[0]
        stress = lcr.loc[lcr['scenario'] == 'Stress',  'lcr'].values[0]
        assert stress < normal

    def test_lcr_passes_when_liquid_exceeds_redemptions(self, sim):
        # available (8M liquid) >> normal redemptions (~418K) → pass
        lcr = sim.liquidity_coverage()
        assert lcr.loc[lcr['scenario'] == 'Normal', 'pass'].values[0] == True

    def test_available_eur_same_across_scenarios(self, sim):
        # asset side is identical for both rows -- only redemptions differ
        lcr = sim.liquidity_coverage()
        assert lcr['available_eur'].iloc[0] == lcr['available_eur'].iloc[1]

    def test_lcr_fails_when_all_positions_illiquid(self, investors):
        illiquid = pd.DataFrame([
            {"name": "RE A", "asset_class": "Direct RE", "mv": 10_000_000, "daily_volume": 0},
        ])
        s   = FundLiquiditySimulator(illiquid, investors)
        lcr = s.liquidity_coverage()
        assert not lcr.loc[lcr['scenario'] == 'Normal', 'pass'].values[0]
        assert lcr.loc[lcr['scenario'] == 'Normal', 'available_eur'].values[0] == 0.0


# ── TestAnnexIV ───────────────────────────────────────────────────────────────

class TestAnnexIV:

    def test_annex_iv_has_7_rows(self, sim):
        assert len(sim.annex_iv()) == 7

    def test_annex_iv_columns(self, sim):
        expected = [
            'Liquidation horizon',
            'pct portfolio (base)',
            'pct portfolio (stressed)',
            'pct equity redeemable',
            'liquidity gap (base)',
        ]
        assert list(sim.annex_iv().columns) == expected

    def test_annex_iv_base_pct_sums_to_100(self, sim):
        assert abs(sim.annex_iv()['pct portfolio (base)'].sum() - 100.0) < 0.01

    def test_stressed_1day_bucket_below_base(self, sim):
        # Govt Bond is in "1 day" at base; 50% volume haircut moves it to "2-7 days"
        av      = sim.annex_iv()
        base_1d = av.loc[av['Liquidation horizon'] == '1 day', 'pct portfolio (base)'].values[0]
        str_1d  = av.loc[av['Liquidation horizon'] == '1 day', 'pct portfolio (stressed)'].values[0]
        assert str_1d < base_1d

    def test_equity_redeemable_sums_to_100(self, sim):
        # all redemption liability concentrated in the notice period bucket
        assert abs(sim.annex_iv()['pct equity redeemable'].sum() - 100.0) < 0.01


# ── TestLMTSimulation ─────────────────────────────────────────────────────────

class TestLMTSimulation:

    def test_lmt_returns_n_months_rows(self, sim):
        assert len(sim.lmt_simulation(n_months=12)) == 12
        assert len(sim.lmt_simulation(n_months=6))  == 6

    def test_lmt_columns(self, sim):
        expected = [
            'month', 'nav_eur', 'liquid_nav_eur', 'illiquid_nav_eur',
            'liquid_pct_nav', 'gross_redemption_pct', 'requested_eur',
            'requested_pct', 'gate_limit_pct', 'paid_eur', 'backlog_eur',
            'backlog_pct_nav', 'gate_active', 'gate_consecutive',
            'swing_active', 'swing_factor_bps', 'suspension_active',
        ]
        assert all(c in sim.lmt_simulation().columns for c in expected)

    def test_gate_triggers_when_redemptions_exceed_threshold(self, sim):
        # stress_months index 2 → M3; 20% redemption > 10% gate threshold
        result = sim.lmt_simulation(n_months=6, stress_months={2: 0.20}, gate_threshold=0.10)
        assert result.loc[result['month'] == 'M3', 'gate_active'].values[0] == True

    def test_backlog_nonzero_when_gate_fires(self, sim):
        result = sim.lmt_simulation(n_months=6, stress_months={2: 0.20}, gate_threshold=0.10)
        assert result.loc[result['month'] == 'M3', 'backlog_eur'].values[0] > 0

    def test_swing_triggers_above_threshold(self, sim):
        # 15% gross redemption exceeds 5% swing threshold
        result = sim.lmt_simulation(n_months=6, stress_months={1: 0.15}, swing_threshold=0.05)
        assert result.loc[result['month'] == 'M2', 'swing_active'].values[0] == True

    def test_suspension_triggers_after_consecutive_gates(self, sim):
        # two consecutive 20% stress months build backlog past 50% of liquid sleeve
        result = sim.lmt_simulation(
            n_months=8,
            stress_months={3: 0.20, 4: 0.20},
            gate_threshold=0.10,
            suspension_consec=2,
            suspension_backlog=0.50,
        )
        assert result['suspension_active'].any()

    def test_reproducible_with_same_seed(self, sim):
        r1 = sim.lmt_simulation(random_seed=99)
        r2 = sim.lmt_simulation(random_seed=99)
        assert r1['requested_eur'].equals(r2['requested_eur'])

    def test_no_gate_in_calm_scenario(self, sim):
        # gate threshold at 99.9% -- no realistic redemption will ever breach it
        result = sim.lmt_simulation(
            n_months=12,
            stress_months={},
            gate_threshold=0.999,
            random_seed=42,
        )
        assert not result['gate_active'].any()


# ── TestEdgeCases (QRE-86) ────────────────────────────────────────────────────

class TestEdgeCases:

    def test_gate_triggered_on_month_1(self, sim):
        # 30% redemption on index 0 → M1 gate fires immediately
        result = sim.lmt_simulation(
            n_months=3,
            stress_months={0: 0.30},
            gate_threshold=0.10,
        )
        assert result.loc[result['month'] == 'M1', 'gate_active'].values[0] == True

    def test_all_positions_illiquid_available_is_zero(self, investors):
        illiquid = pd.DataFrame([
            {"name": "RE A", "asset_class": "Direct RE", "mv": 5_000_000, "daily_volume": 0},
            {"name": "RE B", "asset_class": "Direct RE", "mv": 5_000_000, "daily_volume": 0},
        ])
        s   = FundLiquiditySimulator(illiquid, investors)
        lcr = s.liquidity_coverage()
        assert lcr['available_eur'].iloc[0] == 0.0

    def test_all_positions_illiquid_lcr_fails(self, investors):
        illiquid = pd.DataFrame([
            {"name": "RE A", "asset_class": "Direct RE", "mv": 5_000_000, "daily_volume": 0},
            {"name": "RE B", "asset_class": "Direct RE", "mv": 5_000_000, "daily_volume": 0},
        ])
        s   = FundLiquiditySimulator(illiquid, investors)
        lcr = s.liquidity_coverage()
        assert not lcr['pass'].any()

    def test_zero_redemptions_lcr_is_inf(self, positions):
        # NumPy float division x/0.0 → inf (not ZeroDivisionError); unguarded edge case.
        # Any non-zero liquid position divided by zero total redemptions produces inf LCR.
        zero_investors = pd.DataFrame([
            {"type": "Retail", "share": 1.0, "normal_redemption": 0.0, "stress_redemption": 0.0},
        ])
        import math
        s   = FundLiquiditySimulator(positions, zero_investors)
        lcr = s.liquidity_coverage()
        assert math.isinf(lcr.loc[lcr['scenario'] == 'Normal', 'lcr'].values[0])

    def test_nav_zero_lcr_is_nan(self, investors):
        # NumPy 0/0 → nan (not ZeroDivisionError); unguarded edge case.
        # When all positions have mv=0, NAV=0 and all rate calculations produce nan.
        zero_positions = pd.DataFrame([
            {"name": "Empty", "asset_class": "Cash", "mv": 0.0, "daily_volume": 1_000_000},
        ])
        s   = FundLiquiditySimulator(zero_positions, investors)
        lcr = s.liquidity_coverage()
        assert lcr['lcr'].isna().any()
