import requests
import pandas as pd
from quant_risk.data.base import CentralBankClient


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED series codes for US rates
FRED_SERIES = {
    "SOFR":    "SOFR",
    "3M":  "DGS3MO",
    "6M":  "DGS6MO",
    "1Y":  "DGS1",
    "2Y":  "DGS2",
    "5Y":  "DGS5",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}


class FedClient(CentralBankClient):
    """
    Client for the St. Louis Fed FRED API.

    Available data:
    - SOFR daily fixings (overnight OIS rate)
    - US Treasury constant maturity rates (3M to 30Y)
    - Full yield curve across standard maturities

    Requires a free API key from fred.stlouisfed.org
    No rate data requires payment -- all used here is public.

    Day count : ACT/360
    Overnight : SOFR
    Currency  : USD
    """

    def __init__(self, api_key: str):
        """
        Parameters
        ----------
        api_key : str
            Free FRED API key from fred.stlouisfed.org
        """
        self.api_key = api_key

    @property
    def day_count_convention(self) -> str:
        return "ACT/360"

    @property
    def currency(self) -> str:
        return "USD"

    def _get_series(self, series_id: str, last_n: int) -> pd.Series:
        """Fetch a single FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": last_n,
        }
        response = requests.get(FRED_BASE, params=params)
        if response.status_code != 200:
            raise ConnectionError(
                f"FRED API returned {response.status_code} "
                f"for series {series_id}"
            )
        observations = response.json()["observations"]
        records = {
            obs["date"]: float(obs["value"])
            for obs in observations
            if obs["value"] != "."
        }
        return pd.Series(records, name=series_id).sort_index()

    def get_overnight_rate(self, last_n: int = 252) -> pd.Series:
        """
        SOFR daily fixings. The USD OIS reference rate.

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        return self._get_series("SOFR", last_n)

    def get_spot_rate(self, maturity: str = "10Y",
                      last_n: int = 252) -> pd.Series:
        """
        US Treasury constant maturity rate.

        Parameters
        ----------
        maturity : str
            One of 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        series_id = FRED_SERIES.get(maturity)
        if series_id is None:
            raise ValueError(
                f"Maturity '{maturity}' not available. "
                f"Choose from {[k for k in FRED_SERIES if k != 'SOFR']}"
            )
        return self._get_series(series_id, last_n)

    def get_full_curve(self, last_n: int = 5) -> pd.DataFrame:
        """
        Full US Treasury curve across standard maturities.

        Returns
        -------
        pd.DataFrame indexed by date, columns are maturity labels,
        values in percent.
        """
        maturities = [k for k in FRED_SERIES if k != "SOFR"]
        series = {m: self.get_spot_rate(m, last_n=last_n) for m in maturities}
        return pd.DataFrame(series)
        
    def get_fx_spot(self, pair: str = "EURUSD", last_n: int = 5) -> pd.Series:
        """
        FX spot rate from FRED.

        Parameters
        ----------
        pair : str
            Currency pair. Supported: 'EURUSD', 'GBPUSD', 'USDJPY',
            'USDCHF', 'USDBRL'
        last_n : int
            Number of observations to fetch.

        Returns
        -------
        pd.Series indexed by date string, values as FX rate.
        """
        fx_series = {
            "EURUSD": "DEXUSEU",   # USD per EUR
            "GBPUSD": "DEXUSUK",   # USD per GBP
            "USDJPY": "DEXJPUS",   # JPY per USD
            "USDCHF": "DEXSZUS",   # CHF per USD
            "USDBRL": "DEXBZUS",   # BRL per USD
        }
        series_id = fx_series.get(pair.upper())
        if series_id is None:
            raise ValueError(
                f"Pair '{pair}' not supported. "
                f"Choose from {list(fx_series.keys())}"
            )
        return self._get_series(series_id, last_n)
    
    def get_series(self, name: str):
        import time

        if name not in FRED_SERIES:
            raise ValueError(f"{name} not in registry")

        try:
            series_id = FRED_SERIES[name]
            return self.client._get_series(series_id)

        except Exception as e:
            print(f"[WARN] {name} failed: {e}")
            return pd.Series(dtype=float)