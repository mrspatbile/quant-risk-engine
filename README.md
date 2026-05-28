![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![QuantLib](https://img.shields.io/badge/QuantLib-1.42.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://github.com/mrspatbile/quant-risk-engine/actions/workflows/test.yml/badge.svg)
![Regulatory](https://img.shields.io/badge/Regulatory-IRRBB%20%7C%20FRTB%20SA%20%7C%20AIFMD%20II-1a1a2e)


# Quant Risk Engine

Python-based quantitative risk framework built on QuantLib, designed as a modular platform covering both banking and asset management use cases in a European regulatory context.

Implements market data ingestion (ECB, FRED), yield curve construction, and fixed-income instrument pricing. Includes selected regulatory components, with initial focus on IRRBB-style interest rate risk metrics and exploratory extensions toward FRTB and AIFMD-related use cases.

Extended with a Streamlit dashboard for portfolio and liquidity risk visualisation across fund strategies.

**Stack:** Python 3.13 | QuantLib 1.42 | Streamlit | ECB SDW | FRED API  
**Status:** Actively developed -- Sprint 6 in progress

---

## Scope

| Domain | Status |
|--------|--------|
| Yield curve construction -- OIS bootstrapping, NSS | Done |
| Instrument pricing -- bonds, IR swaps, FX forwards, options, callable bonds and strategies (carry trade and FI strats) | Done |
| IRRBB -- EVE and NII supervisory outlier tests | Done |
| FRTB SA -- GIRR delta, key rate DV01 | Done |
| Fund liquidity risk -- AIFMD II, Annex IV, LMTs | Done |
| Portfolio Management -- macrofactors, risk-on, riskoff regimes + factor model portf. construction, attribution, hedging | Done |
| ETF liquidity stress -- creation/redemption mechanism, AP arbitrage breakdown | In progress |
| Monte Carlo + short rate models | In progress -- Sprint 6 |
| XVA -- CVA, FVA, MVA + neural network approximator | In progress -- Sprint 6 |
| Credit risk -- IRB, IFRS 9 ECL | Planned -- Module 7 |

---

## Project Structure

```
quant-risk-engine/
├── apps/
│   ├── fund_liquidity/ # Streamlit dashboard -- AIFMD liquidity risk
│   └── etf/            # Streamlit dashboard -- ETF liquidity stress (in progress)
├── configs/            # market_config.yaml
├── data/
│   ├── cache/          # Parquet cache -- ECB, FRED, yfinance, FF factors, GPR
│   │   └── external/   # ExternalStore cache subdirectory
│   ├── processed/      # Bootstrapped curves, fund position data
│   └── raw/
├── docs/
├── notebooks/
│   ├── 01_yield_curves/
│   ├── 02_asset_pricing/
│   ├── 03_simulations/
│   ├── 04_bank_risk/
│   ├── 05_fund_risk/
│   └── 06_portfolio_management/
├── scripts/            # Live integration scripts -- real API calls, not pytest tests
├── src/quant_risk/
│   ├── data/           # ECBClient, FedClient -- ABC-based; ExternalStore (yfinance/FF/GPR)
│   ├── curves/         # OISCurve, NSSCurve -- QuantLib wrappers
│   ├── instruments/    # Bond, IRSwap, FXForward, VanillaOption, CDS
│   └── risk/           # Risk calculators (in progress)
└── tests/              # Pytest unit tests -- 184+ passing

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

Planned -- Sprint 6/7/8.

| Notebook | Description |
|----------|-------------|
| Vasicek, Hull-White, CIR | Short rate model calibration and path generation |
| Monte Carlo | Path generation, variance reduction, convergence |
| XVA | CVA, FVA, MVA using simulated exposure profiles |
| Neural network XVA | Deep XVA -- NN approximation of Expected Exposure |

---

## Module 4 -- Bank Risk

| Document | Description |
|----------|-------------|
| bank_reg.md | Regulation reference -- Basel IV, CRR3, FRTB, IRRBB, ICAAP, ILAAP, XVA |

FRTB SA and IRRBB implemented in Module 1/2 notebooks. Dedicated bank risk
dashboard planned.

---

## Module 5 -- Fund Risk

| Notebook / App | Description |
|----------------|-------------|
| fund_reg.md | Regulation reference -- AIFMD, AIFMD II, UCITS, IFRS 13, Luxembourg vehicles |
| liquidity_fund_risk.ipynb | Full AIFMD liquidity risk framework -- bucketing, LCR, stress testing, Annex IV, LMT simulation |
| apps/fund_liquidity/ | Streamlit dashboard -- interactive liquidity risk analysis for four fund types |

**Fund types covered:** Multi-asset AIF, Credit AIF (IG/HY/CLO/private credit),
Leveraged AIF (long/short equity with derivatives), Real Estate AIF (REITs + direct property).

**LMT simulation:** Redemption gate with contagion multiplier, swing pricing,
suspension trigger -- liquid sleeve depletion modelled explicitly per
ESMA34-671404336-1364 board-decision framework.

**Regulatory basis:** Delegated Regulation 231/2013, AIFMD II Directive 2024/927/EU,
ESMA34-671404336-1364 (Guidelines on LMTs, April 2025), ESMA/2013/232 (Annex IV).


Implemented as STREAMLIT DASHBOARD

ETF-specific liquidity risk analysis covering the creation/redemption mechanism
under market stress. Covers four ETF types common in the Luxembourg UCITS market:

| ETF | Underlying | Key risk |
|-----|-----------|---------|
| Equity ETF | EuroStoxx 50 large caps | Bid-ask widening, tracking error under stress |
| Fixed Income ETF | EUR IG corporate bonds | Underlying illiquidity drives premium/discount |
| Smart Beta ETF | Factor -- value + low vol | Mixed liquidity profile, rebalancing risk |
| Commodity ETF | Physically backed / swap-based | Collateral liquidity, roll cost |

**Metrics:** tracking error, tracking difference, premium/discount to NAV,
authorised participant arbitrage cost, creation/redemption breakdown indicator.

**Regulatory basis:** ESMA guidelines on ETF liquidity under stressed conditions,
UCITS Directive 2009/65/EC, PRIIPs Regulation 1286/2014.

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
| AIFMD (2011/61/EU) | Fund liquidity notebook and dashboard |
| Delegated Reg. 231/2013 | Liquidity bucketing, stress testing, Annex IV |
| AIFMD II (2024/927/EU) | LMT simulation -- gate, swing, suspension |
| ESMA34-671404336-1364 | Suspension trigger calibration |
| Basel IV / CRR3 | bank_reg.md reference |
| ICAAP / ILAAP | bank_reg.md reference |

---

## OOP Engine

```
src/quant_risk/
├── curves/
│   ├── base.py         # DiscountCurve ABC
│   ├── ois.py          # OISCurve -- QuantLib bootstrapping
│   └── nss.py          # NSSCurve -- Nelson-Siegel-Svensson
├── data/
│   ├── base.py         # CentralBankClient ABC
│   ├── ecb.py          # ECBClient -- ESTR, NSS, MMSR, FX
│   ├── fed.py          # FedClient -- SOFR, US Treasury CMT
│   └── external_store.py  # ExternalStore -- yfinance, FF factors, GPR
└── instruments/
    ├── base.py         # Instrument ABC
    ├── bond.py         # Bond -- OIS discounting, z-spread, key rate DV01
    ├── cds.py          # hazard rate, credit triangle, recovery rate, ISDA standards
    ├── swap.py         # IRSwap -- multi-curve, par rate, MTM, DV01
    ├── fx_forward.py   # FXForward -- CIP, NPV, delta FX/IR
    └── option.py       # VanillaOption -- BSM, implied vol (in progress)
```

**Design principles:** abstract base classes, dependency injection, no global state,
QuantLib global clock managed explicitly.

**Tests:** 184+ tests, all passing.

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| ECB SDW API | ESTR, NSS parameters, MMSR OIS rates, government rates, FX | Free, no key |
| FRED API | SOFR, US Treasury CMT, EUR/USD spot | Free, API key required |
| BCB API | CDI, Selic, spot rates | Free, no key |

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

**Run the fund liquidity dashboard:**

```bash
streamlit run apps/fund_liquidity/app.py
```

---

*Regulation references: CRR3 (EU 2024/1623), EBA/GL/2022/14, EBA/RTS/2022/09,*
*EBA/RTS/2022/10, BCBS d457, EMIR (EU 648/2012), Directive 2011/61/EU,*
*Delegated Regulation 231/2013, Directive 2024/927/EU, ESMA34-671404336-1364.*