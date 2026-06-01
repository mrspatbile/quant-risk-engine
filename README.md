![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![QuantLib](https://img.shields.io/badge/QuantLib-1.42.1-orange)
![Tests](https://github.com/mrspatbile/quant-risk-engine/actions/workflows/test.yml/badge.svg)
![Tests passing](https://img.shields.io/badge/tests-310%20passing-brightgreen)

# Quant Risk Engine

Python quantitative risk **library** built on QuantLib. Covers market data ingestion,
yield curve construction, fixed-income and derivatives pricing, stochastic rate and
equity models, Monte Carlo simulation, and XVA.

This is the **pricing and modelling core**. Banking regulatory applications
(IRRBB, FRTB) consume it from [`banking-risk`](https://github.com/mrspatbile/banking-risk).
Fund risk applications consume it from [`manco-risk-mngmt`](https://github.com/mrspatbile/manco-risk-mngmt).

**Stack:** Python 3.13 | QuantLib 1.42.1 | ECB SDW | FRED API  
**Status:** Sprints 1–6 complete

---

## Layer Separation

This library has a hard scope boundary. Understanding what belongs here and what
does not is the most important design constraint in the project.

**What this library owns:**

- Curve construction — yield curve bootstrapping, NSS parametric curves, `ArrayCurve`
  for downstream injection
- Instrument pricing — `Bond`, `IRSwap`, `FXForward`, `VanillaOption`, `CreditDefaultSwap`
- Sensitivities — `npv()`, `dv01()`, `rate_sensitivities()`, `cs01()`, greeks
- Stochastic models — Vasicek, Hull-White, CIR, GBM, local vol
- XVA math — CVA, DVA, FVA, MVA under netting sets (`XVAEngine`)
- Data ingestion — thin API clients for ECB SDW and FRED (no caching, no persistence)

**What this library does not own:**

- Regulatory thresholds and parameters — IRRBB shock sizes, FRTB risk weights, SIMM
  calibrations are constants defined in the application layer, not here
- Capital calculations — SA-CVA, FRTB capital charges, ICAAP outputs
- Reporting — no DataFrames shaped for regulatory templates, no dashboards
- Visualisation — no `matplotlib`, no plots. This is not a presentation layer.
  Plots belong in `banking-risk` or `manco-risk-mngmt` where the narrative context
  exists. A pricing engine notebook answers "is this number right?" with a printed
  scalar or a small table — not a chart.
- Persistence — data caching, parquet stores, and history management are the
  caller's responsibility. The ECB and FRED clients are stateless HTTP wrappers.

**Downstream repos and what they add:**

| Repo | Adds on top of this library |
|------|-----------------------------|
| [`banking-risk`](https://github.com/mrspatbile/banking-risk) | IRRBB EVE/NII shock scenarios and SOT thresholds, FRTB SA capital, ICAAP/ILAAP, regulatory reporting |
| [`manco-risk-mngmt`](https://github.com/mrspatbile/manco-risk-mngmt) | AIFMD II fund liquidity risk, Annex IV reporting, LMT simulation, board risk dashboards |

If a new feature requires knowing a regulatory threshold, a capital formula, or
produces a chart — it belongs in a downstream repo, not here.

---

## Scope

| Domain | Status |
|--------|--------|
| Yield curve construction — OIS bootstrapping, NSS parametric | Done |
| Instrument pricing — bonds, IR swaps, FX forwards, options, CDS, callable bonds | Done |
| Fixed income strategies — carry trade, curve plays, futures convexity | Done |
| Short rate models — Vasicek, Hull-White, CIR (exact simulation) | Done |
| Monte Carlo — path generation, antithetic sampling, convergence analysis | Done |
| XVA — CVA, DVA, FVA, MVA; XVAEngine with netting set | Done |
| Portfolio management — macro factors, regime detection, factor models | Done |
| FRTB SA GIRR delta — key rate DV01 at prescribed vertices | Done |
| Data layer — ECB/FRED stateless API clients | Done |

---

## Project Structure

```
quant-risk-engine/
├── configs/            # market_config.yaml
├── data/
│   ├── cache/          # Parquet store — external/ (yfinance, FF factors, GPR)
│   └── raw/            # Gitignored
├── docs/
├── notebooks/
│   ├── 01_yield_curves/   # NSS, OIS bootstrapping
│   ├── 02_asset_pricing/  # Bonds, swaps, FX, options, CDS, callable bonds, strategies
│   ├── 03_simulations/    # Vasicek, HW, CIR, MC, XVA (complete)
│   ├── 04_bank_risk/      # Reference docs
│   └── 06_portfolio_management/
├── src/quant_risk/
│   ├── curves/         # OISCurve, NSSCurve; DiscountCurve ABC with _select_params_row
│   ├── data/           # ECBClient, FedClient (CacheMixin-backed); ExternalStore
│   ├── instruments/    # Bond, IRSwap, FXForward, VanillaOption, CDS
│   ├── models/         # VasicekProcess, HullWhiteProcess, CIRProcess, MCSimulator
│   └── risk/           # XVAEngine, Trade
└── tests/              # Pytest unit tests — 268 passing
```

---

## Module 1 -- Yield Curves

| Notebook | Description |
|----------|-------------|
| 01_nss_ecb.ipynb | ECB AAA government curve -- Nelson-Siegel-Svensson, DV01, historical analysis |
| 02_bootstrapping_ois.ipynb | OIS curve bootstrapping from ECB MMSR data via QuantLib PiecewiseNaturalLogCubicDiscount -- interpolation comparison, OIS vs NSS spread |
| 03_irrbb_eve.ipynb | EBA/RTS/2022/10 -- six prescribed shock scenarios, EVE SOT, NII SOT, repricing gap, dashboard |

**OOP:** `OISCurve`, `NSSCurve` -- abstract base class, dependency injection, full test coverage.

---

## Module 2 -- Asset Pricing

| Notebook | Description |
|----------|-------------|
| 01_bonds.ipynb | Bund pricing, z-spread, key rate DV01, FRTB GIRR context |
| 02_swaps.ipynb | Multi-curve IRS (OIS + EURIBOR), par rate, MTM lifecycle, DV01, leg decomposition |
| 03_fx_forwards.ipynb | CIP, EUR/USD forward curve, delta FX, delta IR, hedge effectiveness, cross-currency basis |
| 04_options_vol.ipynb | BSM, greeks, implied vol, smile, SVI, greeks surfaces |
| 05_cds.ipynb | cds spreads, pricing dynamics |
| 06_carry_trade.ipynb | CIP vs UIP, carry trade breaks, P&L decomposition: carry + spot return |
| 07_futures.ipynb | Futures convexity (Hull White approximation), margining |
| 08_fi_strategies.ipynb | curve plays: flattening, steepening, butterfly, rolling the curve |
| 09_callable_bonds.ipynb | binomial tree |


**OOP:** `Bond`, `IRSwap`, `FXForward`, `VanillaOption` -- all inherit from `Instrument` ABC.

**Regulatory context:** EMIR (OIS discounting), FRTB SA GIRR delta, IFRS 13 fair value levels.

---

## Module 3 -- Simulations

| Notebook | Description |
|----------|-------------|
| 05_vasicek_model.ipynb | SDE, exact Gaussian simulation, affine bond pricing, OIS calibration |
| 06_hw_model.ipynb | Hull-White no-arbitrage, future bond prices, Jamshidian swaption, κ-σ landscape |
| 07_cir_model.ipynb | Non-central χ² simulation, Feller condition, CIR bond pricing |
| 08_cva.ipynb | Expected exposure, hazard rate model, CVA, DVA, SA-CVA (CRR3 Art. 383) |
| 09_fva.ipynb | Funding cost/benefit, CSA types, OIS vs SOFR funding |
| 10_mva.ipynb | SIMM initial margin, MPOR scaling, bilateral vs cleared |
| 11_xva_aggregation.ipynb | CVA + DVA + FVA + MVA under netting set, netting benefit |

**OOP:** `VasicekProcess`, `HullWhiteProcess`, `CIRProcess`, `MCSimulator`, `XVAEngine`, `Trade`.

---

## Module 4 -- Bank Risk

| Document | Description |
|----------|-------------|
| bank_reg.md | Regulation reference -- Basel IV, CRR3, FRTB, IRRBB, ICAAP, ILAAP, XVA |

FRTB SA and IRRBB implemented in Module 1/2 notebooks. Dedicated bank risk
dashboard planned.

---

## Module 5 -- Fund Risk

Fund liquidity risk has moved to [`manco-risk-mngmt`](https://github.com/mrspatbile/manco-risk-mngmt)
(AIFMD II, Annex IV, LMT simulation, board risk reporting).

---

## Module 6 -- Portfolio Management

Factor Models: macro factors data ingestion and visualization 
Factor model theory, portfolio optimization, attribution and hedging.



---

## Regulation Coverage

| Regulation | Where implemented |
|------------|------------------|
| EBA/GL/2022/14 | IRRBB notebook -- six shock scenarios |
| EBA/RTS/2022/09 | IRRBB -- OIS discounting, 19 maturity buckets |
| EBA/RTS/2022/10 | EVE SOT 15%, NII SOT 5% |
| CRR3 / FRTB SA | Bond and swap notebooks -- GIRR delta, key rate DV01 |
| EMIR | OIS discounting throughout -- collateralised derivatives |
| IFRS 13 | Fair value levels -- FX forwards notebook, fund_reg.md |
| Basel IV / CRR3 | bank_reg.md reference |
| ICAAP / ILAAP | bank_reg.md reference |

---

## OOP Engine

```
src/quant_risk/
├── curves/
│   ├── base.py         # DiscountCurve ABC — _select_params_row() shared date utility
│   ├── ois.py          # OISCurve — QuantLib log-cubic bootstrapping, parquet persistence
│   └── nss.py          # NSSCurve — Nelson-Siegel-Svensson, date-aware from_ecb()
├── data/
│   ├── base.py         # CentralBankClient ABC
│   ├── ecb.py          # ECBClient — ESTR, NSS, MMSR OIS, FX (stateless HTTP wrapper)
│   ├── fed.py          # FedClient — SOFR, US Treasury CMT, FX (stateless HTTP wrapper)
│   └── external_store.py  # ExternalStore — yfinance, FF factors, GPR; parquet-cached
├── instruments/
│   ├── base.py         # Instrument ABC
│   ├── bond.py         # Bond — OIS discounting, z-spread, key rate DV01
│   ├── cds.py          # CDS — hazard rate, credit triangle, ISDA standards
│   ├── swap.py         # IRSwap — multi-curve, par rate, MTM, DV01
│   ├── fx_forward.py   # FXForward — CIP, NPV, delta FX/IR
│   └── option.py       # VanillaOption — BSM, implied vol, greeks
├── models/
│   ├── rates.py        # VasicekProcess, HullWhiteProcess, CIRProcess
│   ├── equity.py       # GBMProcess, LocalVolProcess
│   └── simulator.py    # MCSimulator — paths, antithetic, exposure profiles, SDF
└── risk/
    └── xva.py          # XVAEngine, Trade — CVA/DVA/FVA/MVA under netting set
```

**Design principles:** abstract base classes, dependency injection, no global state,
QuantLib global clock managed via context manager.

**Tests:** 310 passing, 1 skipped — no live API calls, all fixtures synthetic.

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| ECB SDW API | ESTR, NSS parameters, MMSR OIS rates, government rates, FX | Free, no key |
| FRED API | SOFR, US Treasury CMT, EUR/USD spot | Free, API key required |
| BCB API | CDI, Selic, spot rates | Free, no key |

---

## Notebook Conventions

Notebooks in this repo demonstrate library usage — they are not analysis documents.

- Output is `print()` or a `pd.DataFrame` displayed inline. No `matplotlib`. No plots.
- No regulatory thresholds or capital formulas. Those live in `banking-risk`.
- No data persistence. Notebooks fetch or construct data inline; they do not read
  from `data/cache/` or `data/processed/`.
- Cells are short. If a cell is doing more than one thing, it should be two cells.

---

## Setup

```bash
git clone https://github.com/mrspatbile/quant-risk-engine.git
cd quant-risk-engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
echo "FRED_API_KEY=your_key_here" > .env
jupyter lab
```

---

*Regulation references: CRR3 (EU 2024/1623), EBA/GL/2022/14, EBA/RTS/2022/09,*
*EBA/RTS/2022/10, BCBS d457, EMIR (EU 648/2012), IFRS 13.*