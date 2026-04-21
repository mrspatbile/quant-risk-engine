# Fund Liquidity Risk Dashboard

Interactive AIFMD liquidity risk dashboard built with Streamlit.

**QRE-38 | Sprint 4 | Quant Risk Engine**

## Regulatory basis

- Delegated Regulation 231/2013, Articles 46-49 (liquidity management)
- AIFMD II Directive 2024/927/EU, Annex V (LMTs)
- ESMA34-671404336-1364 (Guidelines on LMTs, April 2025)
- ESMA/2013/232 (Annex IV reporting)

## Structure

```
apps/fund_liquidity/
├── app.py          -- Streamlit dashboard
├── simulator.py    -- FundLiquiditySimulator class (reusable)
└── README.md       -- this file
```

## Installation

From the project root:

```bash
pip install streamlit --break-system-packages
```

Or if using the project venv:

```bash
source .venv/bin/activate
pip install streamlit
```

## Running

From the project root:

```bash
streamlit run apps/fund_liquidity/app.py
```

The app will open at `http://localhost:8501`.

## Data

The app reads position and investor CSV files from `data/processed/`.
Required files:

```
data/processed/
├── aif_positions_example.csv         -- Multi-Asset AIF
├── aif_investors_example.csv
├── aif_credit_fund_positions.csv     -- Credit AIF
├── aif_credit_fund_investors.csv
├── aif_leveraged_fund_positions.csv  -- Leveraged AIF
├── aif_leveraged_fund_investors.csv
├── aif_re_fund_positions.csv         -- Real Estate AIF
└── aif_re_fund_investors.csv
```

## Using the simulator independently

```python
import pandas as pd
from apps.fund_liquidity.simulator import FundLiquiditySimulator

positions = pd.read_csv('data/processed/aif_credit_fund_positions.csv')
investors = pd.read_csv('data/processed/aif_credit_fund_investors.csv')
positions['daily_volume'] = positions['daily_volume'].astype(float)

sim = FundLiquiditySimulator(
    positions=positions,
    investors=investors,
    fund_name='Credit AIF',
    notice_days=30,
)

print(sim.summary())
print(sim.bucket_summary())
print(sim.liquidity_coverage())

lmt = sim.lmt_simulation(n_months=12, stress_months={4: 0.14, 9: 0.22})
print(lmt)
```

## Dashboard features

| Tab | Content |
|-----|---------|
| Liquidity Profile | Bucket bar chart, cumulative curve, position detail, leverage (leveraged fund) |
| Stress Testing | LCR by scenario, stressed vs base profile, summary table |
| Annex IV | Regulatory reporting table + asset vs redemption rights chart |
| LMT Simulation | Redemption flows, NAV sleeve depletion, monthly LMT status table |

All stress parameters (haircuts, gate threshold, swing threshold, contagion multiplier)
are controlled from the sidebar and update all tabs live.
