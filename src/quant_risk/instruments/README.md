# quant_risk.instruments

Instrument pricing library for the Quant Risk Engine. Implements fixed
income, derivative, and fund instruments used across FRTB, IRRBB, XVA,
and fund risk modules.

---

## Design

All instruments inherit from the abstract base class `Instrument` defined
in `base.py`. Instrument pricers depend only on the `DiscountCurve`
abstraction from `quant_risk.curves` -- never on a concrete curve
implementation. This means any instrument can be priced against an OIS
curve, an NSS curve, a shocked curve, or a simulated Monte Carlo path
without changing the instrument code.

```

Instrument (abstract base)
├── Bond
│   ├── FixedRateBond
│   └── FloatingRateBond      
├── Swap
│   └── IRSwap                
├── FXForward                 
└── Option                    
    ├── VanillaOption
    └── ExoticOption          
```


---

## Abstract Base -- `Instrument`

Defined in `base.py`. Every concrete instrument must implement:

| Method | Returns | Description |
|--------|---------|-------------|
| `price(curve)` | `float` | Full revaluation price given a discount curve |
| `dv01(curve)` | `float` | Price change for 1bp parallel shift in rates |
| `duration(curve)` | `float` | Modified duration in years |
| `cash_flows()` | `pd.DataFrame` | Scheduled cash flows with dates and amounts |

Optional methods with default implementations:

| Method | Returns | Description |
|--------|---------|-------------|
| `key_rate_dv01(curve, tenors)` | `dict` | DV01 per tenor bucket -- FRTB delta sensitivity |
| `pv01(curve)` | `float` | Alias for dv01 -- EUR convention |
| `describe()` | `str` | Human readable summary |

---

## Implemented Instruments

### `Bond` -- `bond.py`
Fixed rate government or corporate bond priced via OIS discounting.

```python
from quant_risk.instruments.bond import Bond
from quant_risk.curves.ois import OISCurve

curve = OISCurve.from_processed()
bond  = Bond(
    isin           = "DE0001102580",
    face_value     = 1_000_000,
    coupon_rate    = 2.60,
    issue_date     = "2023-08-15",
    maturity_date  = "2034-08-15",
    currency       = "EUR",
)

print(bond.price(curve))
print(bond.dv01(curve))
print(bond.duration(curve))
print(bond.key_rate_dv01(curve))
```

**Regulatory use:**
- FRTB SA -- delta GIRR sensitivity per tenor vertex
- IRRBB -- key rate DV01 feeds EVE bucket calculation
- IFRS 9 -- Level 2 fair value, OIS discounting

---

### `IRSwap` -- `swap.py` _(planned)_
Multi-curve interest rate swap. Fixed leg discounted at OIS, floating
leg projected at EURIBOR and discounted at OIS.

```python
from quant_risk.instruments.swap import IRSwap
from quant_risk.curves.ois import OISCurvecurve = OISCurve.from_processed()
swap  = IRSwap(
notional       = 10_000_000,
maturity_years = 5,
fixed_rate     = 2.50,
valuation_date = "2026-03-24",
)

print(swap.price(curve))
print(swap.par_rate(curve))
print(swap.dv01(curve))
```

**Regulatory use:**
- FRTB SA -- GIRR delta per tenor vertex
- EMIR -- OIS discounting mandatory for collateralised derivatives
- IFRS 13 -- Level 2 fair value

---

### `FXForward` -- `fx_forward.py`
FX forward priced via covered interest rate parity. EUR and USD OIS
discount curves, cross-currency basis adjustment.

```python
from quant_risk.instruments.fx_forward import FXForward
import QuantLib as ql

fwd = FXForward.seasoned(
    notional_foreign = 1_000_000,
    spot_rate        = 1.1578,
    f0               = 1.1792,
    maturity_date    = ql.Date(24, 3, 2027),
    valuation_date   = ql.Date(24, 3, 2026),
    usd_disc_handle  = usd_handle,
)

print(fwd.price(curve))
print(fwd.delta_fx(curve))
print(fwd.delta_ir_usd(curve))
print(fwd.hedge_effectiveness(10_000_000, curve))
```

**Regulatory use:**
- FRTB SA -- delta FX (FX risk bucket), delta IR (GIRR bucket)
- EMIR -- FX forwards > 3 days are reportable OTC derivatives
- AIFMD II -- hedge effectiveness monitoring, monthly rebalancing
- IFRS 13 -- Level 2 fair value

---

### `VanillaOption` -- `option.py`
European vanilla option priced via Black-Scholes-Merton. Full Greeks,
implied volatility inversion, FRTB curvature risk.

```python
from quant_risk.instruments.option import VanillaOption
import QuantLib as ql

opt = VanillaOption(
    spot           = 5250.0,
    strike         = 5250.0,
    expiry_date    = ql.Date(24, 6, 2026),
    valuation_date = ql.Date(24, 3, 2026),
    sigma          = 0.165,
    option_type    = 'call',
    notional_      = 1000.0,
    div_yield      = 0.030,
)

print(opt.price(curve))
print(opt.delta(curve))
print(opt.vega(curve))
print(opt.frtb_curvature(curve))
```

**Regulatory use:**
- FRTB SA -- delta, vega, curvature risk (CRR3 Article 325)
- EMIR -- OTC options reporting and margining
- IFRS 13 -- Level 2 (observable vol), Level 3 (exotic)

---

### `CreditDefaultSwap` -- `cds.py`
CDS priced via QuantLib MidPointCdsEngine. Hazard rate bootstrapping,
CS01, jump-to-default, FRTB CSR-NS capital.

```python
from quant_risk.instruments.cds import CreditDefaultSwap
import QuantLib as ql

cds = CreditDefaultSwap.from_flat_spread(
    valuation_date = ql.Date(24, 3, 2026),
    maturity       = ql.Date(20, 3, 2031),
    notional_      = 10_000_000,
    par_spread     = 0.0150,
    coupon         = 0.0100,
)

print(cds.price(curve))
print(cds.cs01(curve))
print(cds.par_spread(curve))
print(cds.jump_to_default(curve))
```

**Regulatory use:**
- FRTB SA -- CSR-NS delta (CS01 per tenor vertex), Default Risk Charge
- EMIR -- CDS reporting and clearing obligations
- IFRS 9/13 -- Level 2 fair value, OIS discounting

---

## Regulatory Coverage

| Instrument | FRTB SA | IRRBB | XVA | IFRS 9 |
|------------|---------|-------|-----|--------|
| Bond | Delta GIRR | EVE bucket | CVA collateral | Level 2 FV |
| IR Swap | Delta GIRR, Vega | EVE, NII | CVA, DVA | Level 2 FV |
| FX Forward | Delta FX | -- | CVA | Level 2 FV |
| Vanilla Option | Delta, Vega | -- | CVA | Level 3 FV |

---

## Dependencies
quant_risk.curves.base.DiscountCurve  -- curve abstraction
quant_risk.curves.ois.OISCurve        -- OIS discounting
quant_risk.curves.nss.NSSCurve        -- government benchmark
QuantLib                               -- schedule, day count, analytics