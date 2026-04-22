# Fund Regulation Reference -- Quant Risk Engine

A concise reference covering the main regulatory frameworks applicable to
EU alternative investment funds and UCITS, with focus on risk measurement
and quantitative implementation. Includes IFRS 13 fair value hierarchy as
it applies to fund valuation.

---

## 1 The Luxembourg Fund Landscape

Luxembourg is the largest fund domicile in the EU and the second largest
in the world after the US. The regulatory authority is the
**Commission de Surveillance du Secteur Financier (CSSF)**.

| Vehicle | Directive | Supervisor | Typical investor |
|---------|-----------|------------|-----------------|
| UCITS | UCITS V (EU 2014/91) | CSSF | Retail |
| AIF -- RAIF | AIFMD | CSSF | Professional |
| AIF -- SIF | AIFMD | CSSF | Well-informed |
| AIF -- SICAR | AIFMD | CSSF | Professional (PE/VC) |
| AIF -- Part II UCI | AIFMD | CSSF | Professional |

**Key distinction:** UCITS is a product-level directive -- the fund itself is
authorised and can passport across the EU. AIFMD is a manager-level directive --
the Alternative Investment Fund Manager (AIFM) is authorised, not the fund.

---

## 2 AIFMD -- Alternative Investment Fund Managers Directive

### 2.1 Legislative History

| Instrument | Reference | Status | Content |
|------------|-----------|--------|---------|
| AIFMD | Directive 2011/61/EU | Transposed into Luxembourg law | Original directive -- manager authorisation, depositary, leverage, reporting |
| AIFMD Level 2 | Commission Delegated Regulation 231/2013/EU | **Directly applicable** -- no transposition needed | Risk management, leverage, liquidity, remuneration, transparency detail |
| AIFMD II | Directive 2024/927/EU | Transposition deadline: April 2026 | Loan-originating AIFs, LMT - liquidity management tools, delegation rules |
| AIFMD II Level 2 | Delegated acts pending (ESMA mandates) | TBD | LMT calibration, loan-AIF RTS (reg techl stds) |

**Luxembourg transposition of AIFMD:**
- Law of 12 July 2013 on AIFMs (as amended) -- transposed Directive 2011/61/EU
- CSSF Regulation 13-02 of 15 October 2013 -- CSSF-specific AIFM requirements
- CSSF Circular 18/698 -- internal governance and risk management organisation

**Delegated Regulation 231/2013 is directly applicable EU law** -- it was adopted
as a Regulation, not a Directive, so Luxembourg did not need to transpose it.
It applies identically across all EU member states.

### 2.2 AIFM Authorisation Thresholds

AIFMs below the thresholds are exempt from full authorisation (registered AIFMs only):

| Threshold | AUM limit | Condition |
|-----------|-----------|-----------|
| Sub-threshold leveraged | EUR 100 million | AUM including leverage |
| Sub-threshold unleveraged | EUR 500 million | AUM, no redemption rights for 5 years |

Full authorisation required above thresholds (Article 6, Directive 2011/61/EU).

### 2.3 Risk Management Framework

**Source: Articles 38-45, Delegated Regulation 231/2013**

The AIFM must maintain a permanent risk management function that is functionally
and hierarchically separate from portfolio management.

**Minimum risk measures to monitor (Article 40):**

| Risk category | Metric | Frequency |
|---------------|--------|-----------|
| Market risk | VaR, stress tests, sensitivity | At least monthly |
| Liquidity risk | Liquidity coverage, redemption modelling | At least monthly |
| Counterparty risk | Exposures by counterparty, netting | At least quarterly |
| Operational risk | Incident tracking, scenario analysis | Continuous |
| Concentration risk | Issuer, sector, geography | At least monthly |

**Stress testing requirements (Article 48):**

AIFMs must conduct stress tests that:
- Cover liquidity and market risk jointly
- Use both historical and hypothetical scenarios
- Are performed at least annually (more frequently for leveraged or illiquid funds)
- Feed into the AIFM's liquidity management policy

### 2.4 Leverage

**Source: Articles 6-11, Delegated Regulation 231/2013**

AIFMD defines two leverage calculation methods. AIFMs must calculate and
report both:

#### Gross Method

$$Leverage_{gross} = \frac{\sum |Exposure_i|}{NAV}$$

All positions at market value, absolute value. Derivatives converted to their
delta-adjusted notional. No netting.

#### Commitment Method

$$Leverage_{commitment} = \frac{\sum |NetExposure_i|}{NAV}$$

Netting of offsetting positions is allowed if they relate to the same underlying
and are held with the same purpose. Hedging arrangements reduce gross exposure.
Efficient portfolio management techniques (repos, securities lending) are
included at their gross cash value.

**Regulatory leverage limits** are set at the fund level in the fund's
constitutional documents and must be disclosed to investors. There is no
fixed EU-wide cap -- the AIFM sets limits and the CSSF can impose additional
restrictions under Article 25 of Directive 2011/61/EU.

### 2.5 Depositary Requirements

**Source: Articles 21-23, Directive 2011/61/EU**

Every AIF must appoint a depositary:

| Function | Requirement |
|----------|-------------|
| Cash monitoring | All cash flows monitored -- subscriptions, redemptions, fees |
| Asset safekeeping | Financial instruments held in custody; other assets verified and recorded |
| Oversight | Verify NAV calculation, ensure AIF rules and law are complied with |
| Liability | Depositary liable for loss of financial instruments held in custody |

Depositary must be established in the home member state of the AIF (Luxembourg
depositaries for Luxembourg AIFs). Entities eligible: credit institutions,
MiFID investment firms with own funds above EUR 730k.

### 2.6 Transparency and Reporting

**Source: Articles 22-24, Directive 2011/61/EU; Articles 104-111, Delegated Regulation 231/2013**

| Report | Recipient | Frequency |
|--------|-----------|-----------|
| Annual report | Investors + CSSF | Annual (within 6 months of fiscal year end) |
| Investor disclosure (pre-investment) | Investors | Before investment |
| Investor disclosure (material changes) | Investors | Without undue delay |
| AIFMD Annex IV reporting | CSSF (then ESMA) | Quarterly (AUM > EUR 1bn) or Semi-annual |

**Annex IV key data fields reported:**

- Fund identification (LEI, domicile, strategy)
- AUM, NAV, number of investors
- Leverage (gross and commitment)
- Risk measures: VaR, stress test results
- Liquidity profile (proportion by bucket)
- Portfolio composition by asset class and geography
- Top 5 counterparties and concentrations

### 2.7 AIFMD II -- Key Changes (Directive 2024/927/EU)

Transposition deadline: April 2026. Key additions over original AIFMD:

| Area | Change |
|------|--------|
| Loan-originating AIFs | New sub-regime -- capital requirements, concentration limits, leverage cap 300% |
| Liquidity Management Tools (LMTs) | Mandatory availability of at least two LMTs (redemption gates, notice periods, swing pricing, anti-dilution levies, side pockets) |
| Delegation | AIFM cannot delegate more than it retains -- letter-box entity prohibition reinforced |
| ELTIF alignment | Coordinated changes with ELTIF 2.0 Regulation (EU 2023/606) |
| Third-country passporting | Scope narrowed -- stricter conditions for non-EU AIFMs |

---

## 3 UCITS -- Undertakings for Collective Investment in Transferable Securities

### 3.1 Legislative History

| Instrument | Reference | Status | Content |
|------------|-----------|--------|---------|
| UCITS I | Directive 85/611/EEC | Replaced | Original framework |
| UCITS IV | Directive 2009/65/EC | Transposed | KIID, management company passport, master-feeder |
| UCITS V | Directive 2014/91/EU | Transposed | Depositary, remuneration, sanctions -- aligned with AIFMD |
| UCITS Level 2 | Commission Directive 2010/43/EU | Transposed | Organisational requirements, conflicts, risk management |
| UCITS Level 2 | Commission Regulation 583/2010 | **Directly applicable** | KIID format and content |

**Luxembourg transposition:**
- Law of 17 December 2010 relating to undertakings for collective investment
  (as amended) -- transposed UCITS IV and V into Luxembourg law
- CSSF Circular 14/592 -- UCITS risk management
- CSSF Circular 08/380 -- risk management processes for UCITS

### 3.2 Eligible Assets and Diversification

**Source: Articles 50-57, Directive 2009/65/EC**

UCITS may only invest in transferable securities and money market instruments
admitted to regulated markets, plus a limited list of other assets (eligible
units of UCITS/UCIs, deposits, financial derivatives for hedging or EPM).

**Key diversification limits (5/10/40 rule):**

| Rule | Limit |
|------|-------|
| Single issuer (general) | Max 5% of NAV |
| Single issuer (elevated) | Up to 10% if total of 10%+ positions does not exceed 40% |
| Government securities | Up to 35% in a single government issuer |
| Single UCITS/UCI | Max 10% of NAV |
| Deposits with single credit institution | Max 20% of NAV |
| OTC derivative counterparty exposure | Max 5% of NAV (10% for credit institutions) |

### 3.3 Global Exposure and Risk Measurement

**Source: CSSF Circular 11/512; CESR/10-788; Articles 40-42, Directive 2010/43/EU**

UCITS must measure global exposure to limit derivative risk. Two methods:

#### Commitment Approach

For non-sophisticated funds with limited derivative use.

$$GlobalExposure \leq 100\% \text{ of NAV}$$

Derivatives converted to their equivalent underlying position. Simple netting
rules apply for positions in the same underlying.

#### Value at Risk (VaR) Approach

For sophisticated funds or those with complex derivative strategies.

Two variants:

| Variant | Description | Limit |
|---------|-------------|-------|
| Absolute VaR | VaR of the fund portfolio | Max 20% of NAV (99% confidence, 20 business days) |
| Relative VaR | VaR of fund / VaR of reference portfolio | Max ratio of 2:1 |

**VaR calculation requirements (CESR/10-788):**

| Parameter | Requirement |
|-----------|-------------|
| Confidence level | 99% one-tailed |
| Holding period | 20 business days (or 1 day scaled) |
| Historical observation period | Minimum 1 year (250 business days) |
| Data updates | At least monthly (daily recommended) |
| Backtesting | Daily, minimum 250 days, report exceptions |

**Stress testing** required alongside VaR:
- Standard scenarios (parallel shifts, slope changes, spread widening)
- Specific scenarios relevant to portfolio strategy
- Results reported to senior management at least monthly

### 3.4 Counterparty Risk from Derivatives

OTC derivative counterparty exposure under commitment approach:
- Max 5% of NAV per counterparty (general)
- Max 10% of NAV for credit institution counterparties

For collateralised positions, net exposure after haircuts counts against the limit.

### 3.5 Leverage in UCITS

UCITS do not use "leverage" as a formal regulatory metric in the same way as
AIFMD. Global exposure (commitment or VaR) is the primary metric. However:

- Under the commitment approach, total commitment cannot exceed 100% of NAV
- This implies a maximum gross leverage of 200% (assets at 200% of NAV if fully deployed)
- UCITS using the VaR approach may have higher notional leverage but are constrained
  by absolute or relative VaR limits

### 3.6 KIID and PRIIPs KID

| Document | Regulation | Applicable to |
|----------|------------|--------------|
| KIID | Commission Regulation 583/2010 (directly applicable) | UCITS (retail investors) |
| PRIIPs KID | Regulation 1286/2014/EU (directly applicable) | AIFs marketed to retail + UCITS from 2026 |

**PRIIPs KID (Regulation 1286/2014/EU, as amended by Regulation 2021/2259/EU):**

- UCITS exemption from PRIIPs expired 31 December 2022 -- UCITS now required to
  produce PRIIPs KID for retail investors
- Contains: risk indicator (SRI 1-7), performance scenarios, costs over time
- **SRI calculation** uses historical or simulated volatility and is prescribed
  by the PRIIPs RTS (Commission Delegated Regulation 653/2017)

KIID -- Key Investor Information Document
Regulation 583/2010, directly applicable. Two-page standardised pre-investment disclosure for retail investors.

Investment objectives and policy
SRRI (1-7, based on historical volatility)
Past performance
Costs and charges
Practical information (depositary, tax, complaints)

PRIIPs KID -- Key Information Document for Packaged Retail and Insurance-based Investment Products
Regulation 1286/2014, directly applicable. Three-page maximum pre-investment disclosure, mandatory for retail investors from January 2023, replacing the KIID.

SRI (1-7, VEV-based -- see metrics table)
Prescribed performance scenarios (stress, unfavourable, moderate, favourable)
Granular cost breakdown over holding period
Applies uniformly to funds, structured products, and insurance wrappers

KIID vs PRIIPs KID today: PRIIPs KID mandatory for all UCITS share classes marketed to retail. KIID remains valid only for professional-only share classes where PRIIPs does not apply.


---

## 4 Liquidity Risk Management -- Fund Level

### 4.1 AIFMD Liquidity Requirements

**Source: Articles 46-49, Delegated Regulation 231/2013**

AIFMs of open-ended funds must:

- Maintain a liquidity management system appropriate to the investment strategy
- Ensure the liquidity profile of the AIF is consistent with its redemption policy
- Monitor the liquidity of assets and model investor redemption behaviour
- Conduct liquidity stress tests at least annually

**Liquidity bucketing (Annex IV reporting):**

| Bucket | Liquidation horizon |
|--------|---------------------|
| 1 day | Cash, overnight deposits |
| 2-7 days | Government bonds, large-cap equities |
| 8-30 days | Corporate bonds, small/mid-cap equities |
| 31-90 days | Private credit, less liquid bonds |
| 91-180 days | Real estate, private equity (listed) |
| 181-365 days | Illiquid alternatives |
| > 1 year | Private equity, infrastructure, direct real estate |

### 4.2 AIFMD II Liquidity Management Tools (LMTs)

**Source: Annex V, Directive 2024/927/EU**

Open-ended AIFs must have at least two LMTs available in their constitutional
documents (unless the AIF's investment strategy makes them inappropriate):

| LMT | Description | Use case |
|-----|-------------|---------|
| Redemption gates | Cap on redemptions per period | High redemption stress |
| Notice period extension | Extend redemption notice beyond normal | Illiquid markets |
| Suspension | Full suspension of subscriptions and redemptions | Exceptional circumstances |
| Side pockets | Segregate illiquid assets | Asset illiquidity event |
| Swing pricing | Adjust NAV to protect remaining investors | Dilution from large flows |
| Anti-dilution levy (ADL) | Fee charged to redeeming/subscribing investors | Transaction cost recovery |
| Redemption in kind | Deliver securities instead of cash | Extreme illiquidity |

### 4.3 UCITS Liquidity Requirements

**Source: Article 40(3), Directive 2010/43/EU; CESR/09-178**

UCITS must be capable of honouring redemption requests at all times under
normal market conditions. Minimum requirements:

- At least 10% of NAV must be redeemable within 1 business day (practical guideline)
- Liquidity of portfolio must match redemption frequency
- Temporary suspension permitted only in exceptional circumstances (Article 45,
  Directive 2009/65/EC), with immediate notification to CSSF and investors

---

## 5 Valuation -- IFRS 13

### 5.1 Overview

**IFRS 13 Fair Value Measurement** defines fair value and establishes a
hierarchy for inputs used in fair value measurement. It applies to all
IFRS-reporting entities where another standard requires or permits fair
value measurement.

IFRS 13 is an IASB standard adopted by the EU via:
- **Commission Regulation (EU) 1255/2012** -- endorsed IFRS 13 for EU use
- **Directly applicable** in Luxembourg for IFRS-reporting entities (listed
  companies, banks, and any fund that elects IFRS reporting)
- Luxembourg funds using Lux GAAP use analogous fair value principles under
  the Law of 19 December 2002 on the register of commerce

### 5.2 Fair Value Definition

**IFRS 13.9:**

> Fair value is the price that would be received to sell an asset or paid to
> transfer a liability in an orderly transaction between market participants
> at the measurement date.

Key elements:
- Exit price (not entry price / transaction price)
- Orderly transaction (not a forced sale)
- Market participant perspective (not entity-specific)
- Measurement date (point-in-time)

### 5.3 The Fair Value Hierarchy

IFRS 13 classifies inputs to valuation techniques into three levels:

| Level | Inputs | Examples |
|-------|--------|---------|
| Level 1 | Quoted prices in active markets for identical assets | Exchange-traded equities, on-the-run government bonds, listed futures |
| Level 2 | Observable inputs other than Level 1 prices | OTC derivative prices from dealer quotes, bonds priced from yield curves, FX forwards from observable spot and interest rates |
| Level 3 | Unobservable inputs | DCF with internally estimated cash flows, private equity NAV, illiquid structured products, model-based prices where significant inputs are unobservable |

**Classification rules:**

- The level is determined by the lowest level input that is significant to the
  overall fair value measurement
- A derivative can be Level 2 even if the model is complex, provided all
  significant inputs are observable
- A bond priced off a yield curve is Level 2 if the curve is constructed from
  observable market data

### 5.4 Valuation Techniques

**IFRS 13.61-13.66:**

Three approaches are recognised:

| Approach | Description | Fund application |
|----------|-------------|-----------------|
| Market approach | Uses prices from identical or comparable transactions | Comparable company multiples, recent transaction prices for PE |
| Income approach | Discounts future cash flows or earnings | DCF for bonds, private credit, real estate |
| Cost approach | Replacement cost | Rarely used for financial instruments |

Entities must use the technique(s) that maximise the use of observable inputs
and minimise the use of unobservable inputs.

### 5.5 Valuation of Financial Instruments by Asset Class

| Instrument | Typical Level | Technique |
|------------|--------------|-----------|
| Listed equities (liquid market) | Level 1 | Market price |
| Exchange-traded futures/options | Level 1 | Market price |
| OTC interest rate swaps | Level 2 | Discount cash flows using OIS/EURIBOR curves |
| Government bonds (on-the-run) | Level 1/2 | Market price or yield curve pricing |
| Corporate bonds (IG, liquid) | Level 2 | Yield curve + credit spread |
| Corporate bonds (HY, illiquid) | Level 2/3 | Dealer quotes or model |
| FX spot | Level 1 | Observable FX rate |
| FX forwards | Level 2 | CIP: spot + OIS discount factors |
| OTC options (vanilla) | Level 2 | BSM with observable implied vol |
| OTC options (exotic) | Level 2/3 | Model with partially unobservable inputs |
| Private equity fund interests | Level 3 | NAV of underlying fund |
| Direct PE / VC investments | Level 3 | DCF or comparable transaction |
| Real estate (direct) | Level 3 | Appraised value, income capitalisation |
| CLO tranches (senior) | Level 2 | Observable spread matrix |
| CLO tranches (equity) | Level 3 | Cash flow model |

### 5.6 Day 1 P&L (Transaction Price vs Fair Value)

**IFRS 13.57-13.60:**

When the transaction price differs from fair value at initial recognition
(e.g., an OTC derivative with an upfront premium), the difference (Day 1 P&L)
is only recognised immediately if fair value is evidenced by observable market
data (Level 1 or Level 2 with all significant inputs observable).

If the valuation relies on unobservable inputs (Level 3), Day 1 P&L is
deferred and amortised over the life of the instrument.

### 5.7 Disclosure Requirements

**IFRS 13.91-13.99:**

For each class of assets and liabilities measured at fair value, entities
must disclose:

| Disclosure | Level 1 | Level 2 | Level 3 |
|------------|---------|---------|---------|
| Fair value at period end | Yes | Yes | Yes |
| Transfers between levels | Yes | Yes | Yes |
| Valuation technique | -- | Yes | Yes |
| Significant unobservable inputs | -- | -- | Yes |
| Sensitivity to unobservable inputs | -- | -- | Yes |
| Reconciliation of opening to closing balance | -- | -- | Yes |

---

## 6 Luxembourg-Specific Fund Vehicles

### 6.1 RAIF -- Reserved Alternative Investment Fund

**Law of 23 July 2016 on RAIFs (as amended by Law of 21 July 2023)**

The RAIF is a Luxembourg-specific vehicle that sits outside direct CSSF
product-level supervision. Instead:

- The RAIF itself is not authorised by CSSF -- no product approval required
- The RAIF must be managed by a fully authorised Luxembourg or EU AIFM
- All AIFMD investor protections apply via the AIFM (depositary, reporting, leverage limits)
- Time to market: 2-4 weeks (no regulatory approval of the fund itself)

Available legal forms: SICAV, SCS, SCSp, SA, Sarl, SCA, cooperative.
Can adopt: umbrella structure, compartments, multiple share classes.

**Key condition:** RAIF investors must be well-informed investors (minimum
EUR 125,000 investment, or professional investors by MiFID II definition).

### 6.2 SIF -- Specialised Investment Fund

**Law of 13 February 2007 on SIFs (as amended)**

SIF is a CSSF-supervised fund vehicle -- lighter touch than UCITS but with
CSSF authorisation at fund level:

- CSSF authorises the SIF and approves material changes
- No eligible asset restrictions (vs UCITS) -- can hold any asset class
- Diversification: 30% limit per single issuer (less strict than UCITS)
- Minimum investor: well-informed investor (EUR 125,000 or professional)
- Minimum subscribed capital: EUR 1,250,000 (within 12 months of authorisation)

### 6.3 SICAR -- Investment Company in Risk Capital

**Law of 15 June 2004 on SICARs (as amended)**

SICAR is designed for private equity and venture capital:

- Invests exclusively in risk capital (equity or quasi-equity in companies
  in their development phase)
- No diversification requirements
- No investment restrictions on instruments or structures
- Minimum investor: professional investor (no EUR 125,000 threshold)
- CSSF-supervised at fund level

---

## 7 Key Metrics Reference -- Fund Risk

| Metric | Formula | Regulatory use |
|--------|---------|---------------|
| Commitment leverage | Net exposure / NAV | AIFMD Annex IV, UCITS |
| Gross leverage | Sum(|exposures|) / NAV | AIFMD Annex IV |
| Absolute VaR | VaR(99%, 20d) | UCITS global exposure |
| Relative VaR | VaR(fund) / VaR(benchmark) | UCITS global exposure |
| Tracking error | $\sigma(R_{fund} - R_{benchmark})$ | UCITS disclosure |
| Max drawdown | Peak to trough NAV decline | Fund reporting |
| Liquidity coverage | Liquid assets / redemption liability | AIFMD liquidity |
| SRI (PRIIPs) | Scaled 1-7 from VEV | PRIIPs KID |
| VEV (PRIIPs) | $\sqrt{Var_H} \times \sqrt{52/H}$ annualised | PRIIPs KID |

**VEV (Value at Risk Equivalent Volatility) for PRIIPs SRI:**

$$VEV = \sqrt{\frac{-2 \ln(1 - VaR_{0.975})}{T}}$$

Where $T$ is the recommended holding period in years. The SRI band is then
determined from the VEV table in Annex II of Delegated Regulation 653/2017.

---

## 8 ESMA Guidelines and CSSF Circulars -- Quick Reference

| Reference | Content |
|-----------|---------|
| ESMA/2013/232 | Reporting obligations under AIFMD |
| ESMA/2014/869 | Key concepts of AIFMD (leverage, AUM calculation) |
| CESR/10-788 | Risk measurement and global exposure for UCITS |
| CESR/09-178 | UCITS liquidity risk management |
| CSSF Circular 18/698 | Internal governance and risk for AIFMs |
| CSSF Circular 14/592 | Risk management process for UCITS |
| CSSF Circular 11/512 | Risk measurement and UCITS derivative exposure |
| CSSF Circular 08/380 | Risk management processes for UCITS |

---

*Reference document for the Quant Risk Engine project.*
*Regulation references: Directive 2011/61/EU (AIFMD), Delegated Regulation 231/2013,*
*Directive 2024/927/EU (AIFMD II), Directive 2009/65/EC (UCITS IV),*
*Directive 2014/91/EU (UCITS V), Regulation 1286/2014/EU (PRIIPs),*
*IFRS 13 as endorsed by Commission Regulation 1255/2012,*
*Luxembourg Law of 12 July 2013, Law of 23 July 2016 (RAIF), Law of 13 February 2007 (SIF).*