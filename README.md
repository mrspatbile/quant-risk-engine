# Quant Risk Engine

A Python-based quantitative risk engine covering yield curve construction,
asset pricing, simulation, and regulatory risk KPIs for banks and funds
under European regulation.

Built as a portfolio project demonstrating production-grade quant risk
skills relevant to Luxembourg and EU financial institutions.

---

## Structure

notebooks/          # Exploratory prototypes by module
src/quant_risk/     # OOP risk engine (VS Code, version controlled)
dashboard/          # Streamlit risk dashboards
tests/              # Unit tests
data/               # Raw and processed market data
configs/            # Market configuration files

## Modules

### Module 1 -- Yield Curves and Asset Pricing
| Notebook | Status | Description |
|----------|--------|-------------|
| 01_nss_ecb.ipynb | Done | NSS curve from ECB parameters -- AAA government curve, parameter sensitivity, historical analysis |
| 02_bootstrapping_ois.ipynb | Done | OIS curve bootstrapping from ECB MMSR data via QuantLib -- interpolation comparison, OIS vs government spread |
| 03_irrbb_eve.ipynb | In progress | EBA/RTS/2022/10 shock scenarios applied to OIS curve, EVE supervisory outlier test |
| 04_asset_pricing.ipynb | Planned | Bond pricing, IR swaps, FX forwards, BSM options, vol surface |

### Module 2 -- Simulation Engine
| Notebook | Status | Description |
|----------|--------|-------------|
| 01_short_rate_models.ipynb | Planned | Vasicek, CIR, Hull-White calibration and path generation |
| 02_hjm.ipynb | Planned | Heath-Jarrow-Morton forward rate dynamics |
| 03_monte_carlo.ipynb | Planned | MC path generation, variance reduction, convergence |
| 04_xva.ipynb | Planned | CVA, DVA, FVA using simulated exposure profiles |

### Module 3 -- Bank Risk KPIs
| Notebook | Status | Description |
|----------|--------|-------------|
| 01_icaap.ipynb | Planned | Economic capital, RAROC, concentration risk |
| 02_ilaap.ipynb | Planned | LCR, NSFR, survival horizon |
| 03_frtb_sa.ipynb | Planned | FRTB standardised approach, sensitivity-based capital |

### Module 4 -- Fund Risk KPIs
| Notebook | Status | Description |
|----------|--------|-------------|
| 01_aifm.ipynb | Planned | AIFMD II liquidity stress testing, redemption coverage |
| 02_ucits.ipynb | Planned | UCITS VaR approach, global exposure, commitment method |
| 03_priips.ipynb | Planned | PRIIPs SRI calculation, performance scenarios |

### Module 5 -- OOP Risk Engine and Dashboard
| Component | Status | Description |
|-----------|--------|-------------|
| src/quant_risk/data/ | In progress | ECB, Fed, BCB market data clients |
| src/quant_risk/curves/ | Planned | Curve classes wrapping QuantLib |
| src/quant_risk/instruments/ | Planned | Bond, swap, option pricers |
| src/quant_risk/risk/ | Planned | VaR, ES, DV01, EVE calculators |
| dashboard/ | Planned | Streamlit curve viewer, risk report |

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| ECB SDW API | ESTR fixings, NSS parameters, MMSR OIS rates, government spot rates | Free, no key |
| ECB EST dataset | ESTR daily fixing | Free, no key |
| ECB MMSR dataset | OIS weighted average rates by maturity bucket | Free, no key |
| Eurex | ESTR futures settlement prices | Free, manual download |
| FRED API | SOFR, US Treasury rates | Free, key required |
| BCB API | CDI, Selic, IPCA | Free, no key |
| yfinance | Equity options, ETF prices | Free, rate limited |

---

## Regulation Coverage

| Regulation | Module | Description |
|------------|--------|-------------|
| EBA/GL/2022/14 | Module 1, 3 | IRRBB -- six prescribed shock scenarios, EVE and NII supervisory outlier test |
| EBA/RTS/2022/09 | Module 1, 3 | IRRBB standardised approach -- OIS discounting, cash flow bucketing |
| EBA/RTS/2022/10 | Module 1, 3 | IRRBB supervisory outlier test -- 15% Tier 1 threshold |
| CRR3 / FRTB | Module 3 | Market risk capital -- SA sensitivities, ES 97.5% |
| AIFMD II | Module 4 | Liquidity stress testing, redemption coverage ratio |
| UCITS | Module 4 | Global exposure, VaR approach, commitment method |
| PRIIPs | Module 4 | SRI, performance scenarios, transaction costs |
| EMIR | Module 2 | OIS discounting for collateralised derivatives |
| IFRS 9 | Module 3 | ECL, fair value hierarchy |

---

## Setup

```bash
git clone https://github.com/yourusername/quant-risk-engine.git
cd quant-risk-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter lab
```

---

## Stack

- Python 3.13
- QuantLib 1.42 -- curve bootstrapping, instrument pricing
- NumPy, SciPy, pandas -- numerical computation
- Matplotlib, Plotly -- visualisation
- Streamlit -- dashboard
- ECB SDW, FRED, BCB -- market data

---

*Work in progress -- actively developed.*
