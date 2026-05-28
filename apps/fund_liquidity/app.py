"""
app.py -- Fund Liquidity Risk Dashboard
QRE-38 | Sprint 4 | Quant Risk Engine

Run with:
    streamlit run apps/fund_liquidity/app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import sys

# ── path setup -- allow import of simulator from same folder ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from simulator import FundLiquiditySimulator, BUCKET_LABELS

# ── logging -- file only, not Streamlit UI ────────────────────────────────────
# st.cache_resource ensures this runs once per server process, not once per
# rerun, so handlers are never duplicated across user interactions.
@st.cache_resource
def _setup_logging():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from quant_risk.logging import configure_file_logging
    configure_file_logging()

_setup_logging()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Fund Liquidity Risk',
    page_icon='💧',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
})

C1, C2, C3, C4 = '#2E75B6', '#ED7D31', '#A9D18E', '#C00000'

# ── data paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parents[2] / 'data' / 'processed'

FUNDS = {
    'Long-Bias AIF': {
    'positions': DATA_DIR / 'aif_longbias_positions.csv',
    'investors': DATA_DIR / 'aif_longbias_investors.csv',


        'nav':       500_000_000,
        'notice':    30,
    },
    'Credit AIF': {
        'positions': DATA_DIR / 'aif_credit_fund_positions.csv',
        'investors': DATA_DIR / 'aif_credit_fund_investors.csv',
        'nav':       None,
        'notice':    30,
    },
    'Leveraged AIF': {
        'positions': DATA_DIR / 'aif_leveraged_fund_positions.csv',
        'investors': DATA_DIR / 'aif_leveraged_fund_investors.csv',
        'nav':       None,
        'notice':    30,
    },
    'Real Estate AIF': {
        'positions': DATA_DIR / 'aif_re_fund_positions.csv',
        'investors': DATA_DIR / 'aif_re_fund_investors.csv',
        'nav':       None,
        'notice':    90,   # RE funds typically have longer notice periods
    },
}

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_fund(fund_name: str) -> FundLiquiditySimulator:
    cfg       = FUNDS[fund_name]
    positions = pd.read_csv(cfg['positions'])
    investors = pd.read_csv(cfg['investors'])
    positions['daily_volume'] = positions['daily_volume'].astype(float)
    return FundLiquiditySimulator(
        positions=positions,
        investors=investors,
        nav=cfg['nav'],
        notice_days=cfg['notice'],
        fund_name=fund_name,
    )

def fmt_eur(value: float) -> str:
    if value >= 1e9:
        return f'EUR {value/1e9:.2f}bn'
    return f'EUR {value/1e6:.0f}m'

def lcr_color(lcr: float) -> str:
    if lcr >= 2.0: return '#1b5e20'
    if lcr >= 1.0: return '#f57f17'
    return '#b71c1c'

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 💧 Fund Liquidity Risk')
    st.markdown('---')

    fund_name = st.selectbox('Fund', list(FUNDS.keys()))
    sim       = load_fund(fund_name)
    kpis      = sim.summary()

    st.markdown('---')
    st.markdown('### Stress Test Parameters')
    eq_haircut = st.slider('Equity / IG volume haircut', 0, 80, 50, step=5,
                           format='%d%%') / 100
    hy_haircut = st.slider('HY / CLO volume haircut', 0, 90, 70, step=5,
                           format='%d%%') / 100

    st.markdown('---')
    st.markdown('### LMT Simulation Parameters')
    gate_thr   = st.slider('Gate threshold (% NAV)', 5, 25, 10, step=1,
                           format='%d%%') / 100
    swing_thr  = st.slider('Swing threshold (% NAV)', 1, 15, 5, step=1,
                           format='%d%%') / 100
    contagion  = st.slider('Contagion multiplier', 1.0, 3.0, 1.5, step=0.1)
    n_months   = st.slider('Simulation months', 6, 24, 12, step=1)

    st.markdown('---')
    st.caption('QRE-38 | Sprint 4 | Quant Risk Engine')
    st.caption('Reg: Del. Reg. 231/2013 | AIFMD II 2024/927/EU')

# ── main layout ───────────────────────────────────────────────────────────────
st.markdown(f'# {fund_name}')
st.markdown('---')

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('NAV',            fmt_eur(kpis['nav_eur']))
k2.metric('Positions',      kpis['n_positions'])
k3.metric('Liquid (<=30d)', f"{kpis['liquid_pct']:.1f}%")
k4.metric('LCR Normal',     f"{kpis['lcr_normal']:.1f}x",
          delta='PASS' if kpis['lcr_normal'] >= 1 else 'FAIL')
k5.metric('LCR Stress',     f"{kpis['lcr_stress']:.1f}x",
          delta='PASS' if kpis['lcr_stress'] >= 1 else 'FAIL')

st.markdown('---')

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '📊 Liquidity Profile',
    '⚡ Stress Testing',
    '📋 Annex IV',
    '🔧 LMT Simulation',
])

# ── tab 1: liquidity profile ──────────────────────────────────────────────────
with tab1:
    buckets = sim.bucket_summary()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('#### Liquidity Profile (% NAV per bucket)')
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.set_prop_cycle(color=plt.cm.Blues(np.linspace(0.35, 0.85, len(buckets))))
        ax.bar(buckets['bucket_label'], buckets['pct_nav'], edgecolor='white')
        ax.set_ylabel('% NAV')
        ax.tick_params(axis='x', rotation=30)
        ax.spines['bottom'].set_color('#555555')
        for i, v in enumerate(buckets['pct_nav']):
            if v > 0:
                ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('#### Cumulative Liquidation Curve')
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(buckets['bucket_label'], buckets['cumulative_pct'],
                marker='o', color=C1, linewidth=2)
        ax.axhline(100, color='grey', linestyle='--', linewidth=1)
        ax.set_ylabel('Cumulative % NAV')
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, 115)
        ax.spines['bottom'].set_color('#555555')
        for i, v in enumerate(buckets['cumulative_pct']):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('#### Position Detail')
    display_cols = ['name', 'asset_class', 'mv', 'days_to_liq', 'bucket_label', 'pct_nav']
    st.dataframe(
        sim.positions[display_cols].style.format({
            'mv': '{:,.0f}',
            'days_to_liq': lambda x: '> 1yr' if x == np.inf else f'{x:.0f}d',
            'pct_nav': '{:.1f}%',
        }),
        use_container_width=True,
    )

    # leverage metrics for leveraged fund
    if 'notional' in sim.positions.columns:
        lev = sim.leverage()
        st.markdown('#### Leverage (AIFMD Gross / Commitment)')
        l1, l2 = st.columns(2)
        l1.metric('Gross leverage',      f"{lev['gross_leverage']:.2f}x")
        l2.metric('Commitment leverage', f"{lev['commitment_leverage']:.2f}x")

# ── tab 2: stress testing ─────────────────────────────────────────────────────
with tab2:
    st.markdown(f'**Equity / IG haircut:** {eq_haircut*100:.0f}%  |  '
                f'**HY / CLO haircut:** {hy_haircut*100:.0f}%')

    scenarios = {
        'Base':              {'eq': 0.00, 'hy': 0.00, 'red': 'normal'},
        'Market stress':     {'eq': eq_haircut, 'hy': hy_haircut, 'red': 'normal'},
        'Redemption stress': {'eq': 0.00, 'hy': 0.00, 'red': 'stress'},
        'Combined stress':   {'eq': eq_haircut, 'hy': hy_haircut, 'red': 'stress'},
    }

    results = {}
    for s_name, params in scenarios.items():
        res = sim.stress_test(params['eq'], params['hy'], params['red'])
        results[s_name] = res

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('#### LCR by Scenario')
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names  = list(results.keys())
        lcrs   = [results[s]['lcr'] for s in names]
        colors = [C1 if r['pass'] else C4 for r in results.values()]
        bars   = ax.bar(names, lcrs, color=colors, edgecolor='white', width=0.5)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
        for bar, val in zip(bars, lcrs):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f'{val:.2f}x', ha='center', fontsize=8, fontweight='bold')
        ax.set_ylabel('LCR')
        ax.tick_params(axis='x', rotation=15)
        ax.spines['bottom'].set_color('#555555')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('#### Stressed Liquidity Profile vs Base')
        base_pct     = sim.bucket_summary()['pct_nav'].values
        stressed_res = sim.stress_test(eq_haircut, hy_haircut, 'stress')
        stressed_pct = stressed_res['bucket_summary']['pct_nav'].values
        x = np.arange(len(BUCKET_LABELS))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(x - w/2, base_pct,    w, color=C1, label='Base',    edgecolor='white')
        ax.bar(x + w/2, stressed_pct, w, color=C4, label='Stressed', edgecolor='white', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(BUCKET_LABELS, rotation=30)
        ax.set_ylabel('% NAV')
        ax.legend()
        ax.spines['bottom'].set_color('#555555')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('#### Stress Test Summary')
    summary_rows = []
    for s_name, res in results.items():
        summary_rows.append({
            'Scenario':         s_name,
            'Available (EUR)':  f"{res['available_eur']:,.0f}",
            'Available (% NAV)':f"{res['available_pct']:.1f}%",
            'Redemption (EUR)': f"{res['redemption_eur']:,.0f}",
            'Redemption (% NAV)':f"{res['redemption_pct']:.1f}%",
            'LCR':              f"{res['lcr']:.2f}x",
            'Result':           'PASS' if res['pass'] else 'FAIL',
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# ── tab 3: annex iv ───────────────────────────────────────────────────────────
with tab3:
    annex = sim.annex_iv(eq_haircut, hy_haircut)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('#### Annex IV -- Liquidity Profile Table')
        st.dataframe(annex.style.format({
            'pct portfolio (base)':     '{:.1f}%',
            'pct portfolio (stressed)': '{:.1f}%',
            'pct equity redeemable':    '{:.1f}%',
            'liquidity gap (base)':     '{:.1f}%',
        }), use_container_width=True)

    with col2:
        st.markdown('#### Annex IV -- Asset vs Redemption Rights')
        x     = np.arange(len(annex))
        width = 0.28
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.set_prop_cycle(color=plt.cm.Blues(np.linspace(0.4, 0.85, 3)))
        ax.bar(x - width, annex['pct portfolio (base)'],     width, label='Portfolio (base)')
        ax.bar(x,         annex['pct portfolio (stressed)'],  width, label='Portfolio (stressed)')
        ax.bar(x + width, annex['pct equity redeemable'],     width, label='Investor equity redeemable')
        ax.set_xticks(x)
        ax.set_xticklabels(annex['Liquidation horizon'], rotation=30)
        ax.set_ylabel('% NAV / AIF equity')
        ax.legend()
        ax.spines['bottom'].set_color('#555555')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── tab 4: lmt simulation ─────────────────────────────────────────────────────
with tab4:
    st.markdown(f'**Gate:** {gate_thr*100:.0f}% NAV  |  '
                f'**Swing:** {swing_thr*100:.0f}% NAV  |  '
                f'**Contagion:** {contagion:.1f}x  |  '
                f'**Months:** {n_months}')

    lmt_df = sim.lmt_simulation(
        n_months=n_months,
        gate_threshold=gate_thr,
        swing_threshold=swing_thr,
        contagion_multiplier=contagion,
    )

    months   = lmt_df['month'].tolist()
    gate_pct = lmt_df['gate_limit_pct']
    def_pct  = (lmt_df['requested_pct'] - gate_pct).clip(lower=0)
    paid_pct = lmt_df['requested_pct'] - def_pct

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('#### Redemption Flows -- Paid vs Deferred')
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(months, paid_pct, color=C1, edgecolor='white', label='Paid')
        ax.bar(months, def_pct, bottom=paid_pct, color=C4, edgecolor='white',
               label='Deferred (gate applied)')
        ax.plot(months, lmt_df['gate_limit_pct'], color='black', linestyle='--',
                linewidth=1.5, label='Gate threshold')
        for i, (swing, bps) in enumerate(zip(lmt_df['swing_active'], lmt_df['swing_factor_bps'])):
            if swing:
                ax.text(i, lmt_df['requested_pct'].iloc[i] + 0.3,
                        f'Swing\n{bps:.0f}bps', ha='center', fontsize=6.5, color='darkblue')
        ax.set_ylabel('Redemptions (% NAV)')
        ax.legend()
        ax.tick_params(axis='x', rotation=30)
        ax.spines['bottom'].set_color('#555555')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('#### NAV Composition -- Liquid Sleeve Depletion')
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.stackplot(months,
                     lmt_df['liquid_nav_eur'] / 1e6,
                     lmt_df['illiquid_nav_eur'] / 1e6,
                     labels=['Liquid sleeve', 'Illiquid sleeve'],
                     colors=[C1, C3], alpha=0.6)
        ax.plot(months, lmt_df['backlog_eur'] / 1e6,
                color=C4, linewidth=2, marker='o', linestyle='--', label='Backlog')
        for i, susp in enumerate(lmt_df['suspension_active']):
            if susp:
                ax.axvspan(i - 0.5, i + 0.5, color=C4, alpha=0.12,
                           label='Suspension' if i == lmt_df['suspension_active'].values.argmax() else '')
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='upper right')
        ax.set_ylabel('EUR million')
        ax.tick_params(axis='x', rotation=30)
        ax.spines['bottom'].set_color('#555555')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('#### Monthly LMT Status')

    def highlight_lmt(val):
        if val is True:  return 'color: #b71c1c; font-weight: bold'
        if val is False: return 'color: #1b5e20; font-weight: bold'
        return ''

    cols = ['month', 'gross_redemption_pct', 'liquid_pct_nav',
            'backlog_pct_nav', 'gate_active', 'swing_active', 'suspension_active']
    st.dataframe(
        lmt_df[cols].set_index('month').style.map(
            highlight_lmt, subset=['gate_active', 'swing_active', 'suspension_active']
        ).format({
            'gross_redemption_pct': '{:.1f}%',
            'liquid_pct_nav':       '{:.1f}%',
            'backlog_pct_nav':      '{:.1f}%',
        }),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric('Gate activated',   f"{lmt_df['gate_active'].sum()} months")
    c2.metric('Swing active',     f"{lmt_df['swing_active'].sum()} months")
    c3.metric('Suspension active',f"{lmt_df['suspension_active'].sum()} months")
