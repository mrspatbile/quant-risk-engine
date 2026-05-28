# CLAUDE.md — Quant Risk Engine

This file tells Claude Code how to work in this repository. Read it before doing anything else.

---

## What this project is

A Python-based quantitative risk framework built on QuantLib, covering banking and
asset management use cases in a European regulatory context. Scope includes yield
curve construction, fixed-income and derivatives pricing, stochastic rate and equity
models, XVA, IRRBB and FRTB SA regulatory metrics, AIFMD fund liquidity risk, and
portfolio management.

The library under `src/quant_risk/` is the core infrastructure: production-quality
implementations of curves, instruments, models, and risk metrics, built on QuantLib
and real market data sources. The notebooks consume this infrastructure — they derive
the mathematics, walk through the implementation, and demonstrate the full
computation chain from market data ingestion to regulatory output. They are worked
examples of a real system, not standalone scripts. Two Streamlit dashboards expose
selected outputs interactively: one for fund liquidity risk, one for ETF liquidity
stress (in progress).

Regulatory accuracy is non-negotiable. A wrong number here is not just a bug — it
can be a reportable compliance incident.

There is a separate project, manco-risk-mngmt, that covers similar regulatory
territory with intentional simplifications for learning purposes. Do not conflate
the two. This project has none of those simplifications.

---

## Stack

- Python 3.13
- QuantLib 1.42.1
- ECB SDW API (no key required)
- FRED API (key required — set in `.env` as `FRED_API_KEY`)
- BCB API (planned — `BCB_API_KEY` is wired in `config.py` but no client exists yet)

---

## Project layout

quant-risk-engine/
├── data/
│   ├── cache/              # Parquet cache — ECB, FRED, yfinance, FF factors, GPR
│   │   └── external/       # ExternalStore cache subdirectory
│   ├── processed/          # Bootstrapped OIS curves, fund position CSVs
│   └── raw/                # Gitignored
├── docs/                   # Regulatory reference documents (not executable)
├── notebooks/
│   ├── 01_yield_curves/
│   ├── 02_asset_pricing/
│   ├── 03_simulations/     # Planned — Hull-White, Vasicek, Monte Carlo, XVA
│   ├── 04_bank_risk/       # Planned — FRTB SA, ICAAP
│   ├── 05_fund_risk/
│   └── 06_portfolio_management/
├── configs/                # market_config.yaml
├── scripts/                # Live integration scripts — test_ecb.py, test_ecb_full.py
│                           # These make real API calls. Not the same as tests/.
├── src/quant_risk/
│   ├── config.py           # Env vars, path constants — imported everywhere
│   ├── setup.py            # Notebook setup helpers (base, asset_pricing, macro variants)
│   ├── curves/             # OISCurve, NSSCurve — QuantLib wrappers
│   ├── data/               # ECBClient, FedClient — ABC-based; ExternalStore (yfinance/FF/GPR)
│   ├── instruments/        # Bond, IRSwap, FXForward, VanillaOption, CDS
│   ├── risk/               # Empty placeholder — risk calculators planned
│   └── utils/              # Empty placeholder
└── tests/                  # Pytest unit tests — no live API calls



The following source files exist but are empty placeholders:
`data/ecb_registry.py`, `data/ecb_store.py`, `data/fed_registry.py`, `data/fed_store.py`,
`risk/__init__.py`, `utils/__init__.py`.

---

## How we work together

Do not make changes without checking with me first.

The preferred flow for every task:

1. Read the relevant Jira ticket before touching anything.
2. Explain your understanding of the task and your proposed approach.
3. Wait for my go-ahead before writing or changing any code.
4. Make changes one logical step at a time — not everything at once.
5. After each step, explain what you did and why, including the regulatory
   reasoning where relevant. Do not over-explain Python basics, but never
   skip the reasoning behind implementation choices.
6. When I confirm a step is done, give me a commit message. I will commit
   myself. Format: `QRE-NNN: short description`. Ask me for the ticket number
   if you do not have it. Include the exact git commands: `git add <files>` and
   `git commit -m "..."`.

---

## Things to never do without explicit permission

- Modify or delete existing passing tests.
- Change the ABC hierarchy (`Instrument`, `DiscountCurve`, `CentralBankClient`)
  or their existing method signatures.
- Refactor across multiple files in one go.
- Change data structures or schemas (instrument constructor signatures, dict keys
  returned by `price()`, DataFrame column names).
- Touch notebook files (`.ipynb`) — treat them as read-only unless the task
  explicitly says otherwise.
- Delete or rename anything.
- Suggest or generate a commit directly to `main`.
- Introduce new dependencies without flagging them first.

---

## OOP design — do not break these

**Abstract base classes.** `DiscountCurve`, `Instrument`, and `CentralBankClient`
are ABCs in `src/quant_risk/curves/base.py`, `instruments/base.py`, and `data/base.py`.
New instruments inherit from `Instrument`. New curve types inherit from `DiscountCurve`.
New data clients inherit from `CentralBankClient`.

**Dependency injection.** Instruments receive a `DiscountCurve` as an argument to
`price()`, `dv01()`, `duration()`, and `key_rate_dv01()`. They never construct or
fetch curves internally. The multi-curve IRS passes both an OIS curve handle and a
EURIBOR basis spread — that pattern is deliberate.

**Factory methods on curves.** `OISCurve` and `NSSCurve` have `from_processed()` and
`from_ecb()` constructors. Use these in tests and notebooks rather than constructing
directly. `from_processed()` reads from `data/processed/` and requires no API call.
`from_ecb()` makes a live API call.

**No notebook logic in `src/`.** Notebooks are for exploration and demonstration.
Business logic belongs in `src/quant_risk/`.

**`ExternalStore` is not a `CentralBankClient`.** It is a standalone class that wraps
yfinance, Kenneth French factor downloads (FF5 + MOM, daily and monthly), and the GPR
index. It uses `CacheMixin` and stores parquet files under `data/cache/external/`.

---

## Hard constraint — QuantLib global clock

`ql.Settings.instance().evaluationDate` is process-global. It is currently mutated
inside `Bond.price()`, `Bond.key_rate_dv01()`, `Bond.z_spread()`, and
`Bond._build_engine()`. This is a known risk. Any parallel execution or test
interaction can silently price at the wrong date.

Never set the evaluation date inside a method and leave it set. Always restore the
previous value. A context manager for this is an active sprint task. Until it exists,
wrap any date mutation in try/finally. This is not optional — silent mispricing from a
stale evaluation date is the highest hidden correctness risk in the codebase.

---

## Code style

- PEP 8 throughout.
- Type hints on all public functions and methods.
- Docstrings on all public classes and functions. Where a parameter has a
  non-obvious convention (rates in percent, basis in bps), state it explicitly.
- No new dependencies without flagging first.

---

## Data conventions

**Rate convention: percent throughout.** `coupon_rate=2.60` means 2.60%, not 0.026.
`zero_rate()` returns percent. `forward_rate()` returns percent. Values are divided
by 100 only when passed into QuantLib internals. Document this in every docstring
where a rate parameter appears — the percent/decimal ambiguity is a silent pricing
risk with real data.

**Basis spreads:** in basis points (bps). `basis_bps=-20.0` means −20bps cross-currency
basis. Divided by 10000 before use.

**Dates:** `ql.Date` objects inside QuantLib calls; ISO strings `"YYYY-MM-DD"` at
public method boundaries. The `_parse_date()` static method on `Bond` is the canonical
ISO → `ql.Date` conversion.

**Notional:** base currency units, no implicit scaling.

**Caching:** parquet files under `data/cache/`. `CacheMixin` handles serialisation.
Single-column data is stored as a one-column DataFrame and returned as a `pd.Series`
with the original name preserved. Never commit cached or processed data.

**`Bond._curve_pillars()`** extracts pillars up to 15Y. Bonds with maturities beyond
15Y use `enableExtrapolation()` on the zero curve. Be aware of this when building
very long-dated instruments.

---

## Testing

- Tests live in `tests/`. Run with `pytest tests/ -v`.
- Do not confuse with `scripts/` — those make live API calls and are not pytest tests.
- 184+ tests currently passing. Do not break them.
- New functionality needs tests before or alongside the implementation, not after.
- Tests must not depend on live API calls. Use fixtures or synthetic data.
- Tests must not depend on files in `data/processed/` or `data/cache/`. The current
  test suite uses `OISCurve.from_processed()` as a shared fixture — this is a known
  fragility (clean checkout will fail). When adding new tests, build curves
  synthetically in the fixture rather than extending this pattern.

---

## Active work

Sprint 5 is complete (CI/CD pipeline, `FundLiquiditySimulator` test coverage,
QuantLib global state context manager, structured logging, input validation).

Current sprint (Sprint 6) covers Module 3 — Monte Carlo simulations and XVA:

- **QRE-48** MC interest rate paths notebook — antithetic sampling
- **QRE-49** MC bond and swap pricing notebook — convergence analysis
- **QRE-50** MC equity paths notebook — GBM, local vol
- **QRE-51** MC simulator OOP class
- **QRE-53** CVA notebook — expected exposure, default modelling
- **QRE-54** FVA notebook — funding cost, CSA impact, OIS vs SOFR
- **QRE-55** MVA notebook — initial margin, SIMM methodology
- **QRE-56** XVA aggregation notebook — CVA + DVA + FVA
- **QRE-57** XVA OOP class

When starting a task, ask me for the Jira ticket number to include in the commit
message.

---

## Regulatory context

The numbers this system produces feed into regulatory reporting. Treat correctness
with the same care you would in a production trading system. If a task touches any
of the items below, say so explicitly when you explain your approach.

| Regulation | Where it matters |
|---|---|
| EBA/RTS/2022/10 | IRRBB EVE SOT 15%, NII SOT 5% — do not change shock parameters |
| FRTB SA (CRR3 Art. 325) | GIRR delta, key rate DV01 at prescribed vertices [0.25Y, 0.5Y, 1Y, 2Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y] |
| AIFMD II (2024/927/EU) | Fund liquidity dashboard — LCR, Annex IV, LMT simulation |
| ESMA34-671404336-1364 | Gate/swing/suspension calibration (April 2025 guidelines) |
| EMIR | OIS discounting throughout — collateralised derivatives |
| IFRS 9 / IFRS 13 | Fair value levels, OIS discounting, Level 2 inputs |

---

## Running things

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Create .env and add your FRED API key:
echo "FRED_API_KEY=your_key_here" > .env

pytest tests/ -v
jupyter lab
```