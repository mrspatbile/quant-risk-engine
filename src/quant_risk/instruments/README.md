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

---

### `FXForward` -- `fx_forward.py` _(planned)_
FX forward priced via covered interest rate parity using two OIS curves.

---

### `VanillaOption` -- `option.py` _(planned)_
European vanilla option priced via Black-Scholes-Merton with implied
volatility surface.

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