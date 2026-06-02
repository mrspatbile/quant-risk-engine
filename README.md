![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![QuantLib](https://img.shields.io/badge/QuantLib-1.42.1-orange)
![Tests](https://github.com/mrspatbile/quant-risk-engine/actions/workflows/test.yml/badge.svg)

# quant-risk-engine

Python library for curve construction, instrument pricing, stochastic simulation, and XVA, built on `QuantLib`. Not tied to any specific regulatory framework. Usable as a standalone pricing engine. All data sources are free, no vendor subscription required.

A downstream application implementing EU banking regulatory frameworks (IRRBB, FRTB, CSRBB, liquidity risk) and consuming this library lives in [`banking-risk`](https://github.com/mrspatbile/banking-risk).

---

## Design

Abstract base classes throughout. Dependency injection. QuantLib evaluation date isolated via `_with_eval_date` context manager. Curve inputs are typed immutable dataclasses (`OISCurveInput`, `NSSParameters`) validated on construction.

---

## Curves

`OISCurve` bootstraps from overnight and par rates across three maturity zones, currently sourced from ECB SDW, swappable to any equivalent input. `NSSCurve` fits the Nelson-Siegel-Svensson model from ECB parameters. `ArrayCurve` builds from any externally supplied zero rates.

All three expose the same interface: discount factors, forward rates, and zero rate projection at arbitrary maturities. Used internally by the pricing layer, but also useful standalone for sensitivity analysis and regulatory scenario work: pass any set of tenors and get back interpolated zero rates instantly, ready to apply prescribed shocks.

---

## Instruments

Instruments are constructed from their contractual characteristics and priced by passing a curve at call time. The same object reprices against any curve, making scenario analysis and rate bumping straightforward.

### Standard Instruments

Priced analytically from a curve. No simulation required.

`Bond`, `IRSwap`, `FXForward`, `EquityForward`, `TotalReturnSwap`, `VanillaOption`.

### Credit

- `CreditDefaultSwap` — flat hazard rate or bootstrapped piecewise curve, CS01, par spread.

### Closed-Form Exotics

- `DigitalOption` — Cash-or-Nothing / Asset-or-Nothing
- `BarrierOption` — DownOut, DownIn, UpOut, UpIn, analytic Haug/Rubinstein
- `ChooserOption` — simple chooser, choose call or put at choice date
- `CompoundOption` — option on an option, Geske analytic engine

### Path-Dependent (GBM Monte Carlo)

- `AsianOption` — arithmetic/geometric, fixed/floating strike
- `LookbackOption` — fixed/floating strike, continuous monitoring
- `CliquetOption` — ratchet, sum of forward-start returns
- `ShoutOption` — lock in intrinsic value once
- `NapoleonOption` — sum of worst periodic returns
- `Accumulator` — forced buy at discount with knock-out
- `Decumulator` — forced sell at premium with knock-out

### Multi-Asset (Bivariate GBM Monte Carlo)

- `WorstOfOption` — worst performer, `from_rho` / `from_prices`
- `BestOfOption` — best performer, `from_rho` / `from_prices`

---

## Simulation Engine

`MCSimulator` drives path-dependent and multi-asset pricing. Antithetic sampling, exposure profiles, and stochastic discount factor computation. Supports Vasicek, Hull-White, CIR, GBM, and local volatility processes.

XVA is the applied output of the simulation layer: `XVAEngine` computes CVA, DVA, FVA, and MVA under netting sets from simulated exposure profiles.

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| ECB SDW | ESTR fixing, MMSR OIS rates, NSS parameters | Free, no key |
| FRED | SOFR, US Treasury CMT, FX spot rates | Free, API key required |
| yfinance | Equity prices, ETFs | Free, no key |

---

## Setup

```bash
git clone https://github.com/mrspatbile/quant-risk-engine.git
cd quant-risk-engine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
echo "FRED_API_KEY=your_key_here" > .env
pytest tests/ -v
```

---

## Tests

CI runs on every push and pull request to `main` via GitHub Actions. Current suite: 460 tests, no live API calls, all fixtures synthetic.