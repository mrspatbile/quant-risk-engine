"""
Calibration utilities — estimate instrument inputs from historical market data.

These are pure computation functions: they take price series as input and return
statistics. No API calls, no caching, no QuantLib dependencies.

Callers fetch price data via ExternalStore (yfinance) or any other source
and pass Series objects directly.

Rate convention : outputs in decimal (e.g. 0.20 = 20% volatility).
"""

import numpy as np
import pandas as pd


def realized_volatility(
    prices: pd.Series,
    window: int | None = None,
    annualisation: int = 252,
) -> float:
    """
    Annualised realised volatility from a price series.

    Computed as the standard deviation of log returns, scaled to annual frequency.

    Parameters
    ----------
    prices : pd.Series
        Historical price series, indexed by date. Must have at least 2 observations.
    window : int or None
        Rolling window in trading days. If None, uses the full series.
    annualisation : int
        Trading days per year for annualisation. Default 252.

    Returns
    -------
    float
        Annualised volatility in decimal (e.g. 0.20 = 20%).

    Examples
    --------
    >>> vol = realized_volatility(prices, window=252)
    >>> sigma = vol   # pass directly to VanillaOption(sigma=vol, ...)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 2:
        raise ValueError("Need at least 2 price observations to compute volatility")
    if window is not None:
        log_returns = log_returns.iloc[-window:]
    return float(log_returns.std() * np.sqrt(annualisation))


def realized_correlation(
    prices1: pd.Series,
    prices2: pd.Series,
    window: int | None = None,
) -> float:
    """
    Pearson correlation of log returns between two price series.

    Parameters
    ----------
    prices1 : pd.Series
        Historical price series for asset 1, indexed by date.
    prices2 : pd.Series
        Historical price series for asset 2, indexed by date.
    window : int or None
        Rolling window in trading days applied to aligned log returns.
        If None, uses the full common history.

    Returns
    -------
    float
        Correlation coefficient in (-1, 1).

    Examples
    --------
    >>> rho = realized_correlation(prices_spx, prices_eurostoxx, window=252)
    >>> opt = WorstOfOption.from_prices(..., prices1=prices_spx, prices2=prices_eurostoxx)
    """
    r1 = np.log(prices1 / prices1.shift(1)).dropna()
    r2 = np.log(prices2 / prices2.shift(1)).dropna()

    common = r1.index.intersection(r2.index)
    if len(common) < 2:
        raise ValueError(
            f"Need at least 2 common observations, got {len(common)}"
        )
    r1, r2 = r1[common], r2[common]

    if window is not None:
        r1, r2 = r1.iloc[-window:], r2.iloc[-window:]
        if len(r1) < 2:
            raise ValueError(
                f"Window={window} results in fewer than 2 observations"
            )

    return float(r1.corr(r2))
