YFINANCE_SERIES = {
    # Commodities
    "GOLD_FUTURES":        "GC=F",        # Gold futures
    "SILVER_FUTURES":      "SI=F",        # Silver futures
    "OIL_WTI_FUTURES":     "CL=F",        # WTI crude oil futures
    "OIL_BRENT_FUTURES":   "BZ=F",        # Brent crude oil futures

    # Safe havens
    "GLD":         "GLD",        # SPDR Gold ETF -- spot proxy, cleaner than GC=F
    "TLT":         "TLT",        # iShares 20Y+ Treasury ETF -- long bond safe haven

    # Broader market context
    "EEM":         "EEM",        # iShares MSCI Emerging Markets ETF -- broad EM
    "IEF":         "IEF",        # iShares 7-10Y Treasury ETF -- rates proxy
    "LQD":         "LQD",        # iShares IG Corporate Bond ETF -- credit proxy
    "HYG":         "HYG",        # iShares HY Corporate Bond ETF -- risk appetite

    # European specific
    "EWG":         "EWG",        # iShares MSCI Germany ETF -- Ukraine proximity
    "EWI":         "EWI",        # iShares MSCI Italy ETF -- peripheral Europe risk


    # Equities
    "SP500":       "^GSPC",       # S&P 500
    "NASDAQ":      "^IXIC",       # Nasdaq Composite
    "STOXX50":     "^STOXX50E",   # Euro Stoxx 50 -- large cap eurozone
    "NIKKEI":      "^N225",       # Nikkei 225 -- Japan

    # FX
    "USDJPY":      "JPY=X",       # USD/JPY spot
    "USDGBP":      "GBP=X",       # USD/GBP spot
    "USDCHF":      "CHF=X",       # USD/CHF spot -- safe haven proxy

    # Volatility
    "VIX_YF":      "^VIX",        # CBOE VIX -- implied vol, fear gauge

    # Sectors
    "ENERGY_SECTOR":  "XLE",      # SPDR Energy Select -- oil majors, gas
    "DEFENSE_SECTOR": "ITA",      # iShares US Aerospace and Defense ETF
    "EUROPE_EQ":      "FEZ",      # SPDR Euro Stoxx 50 ETF -- eurozone equity
    "EM_ASIA":        "GMF",      # SPDR S&P Emerging Asia Pacific ETF
    "INSURANCE":      "IAK",      # iShares US Insurance ETF
    "TOURISM":        "PEJ",      # Invesco Dynamic Leisure and Entertainment -- hotels, airlines, restaurants
}