import requests
import pandas as pd
from quant_risk.data.base import CentralBankClient
from quant_risk.data.cache_mixin import CacheMixin
from quant_risk.logging import get_logger

logger = get_logger(__name__)


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


class FedClient(CacheMixin, CentralBankClient):
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

    def __init__(self, api_key: str, cache_dir=None) -> None:
        """
        Parameters
        ----------
        api_key : str
            Free FRED API key from fred.stlouisfed.org
        cache_dir : Path or str, optional
            Override the default cache directory (data/cache/fred/).
        """
        if not api_key:
            raise ValueError("FRED API key must be a non-empty string")
        self.api_key = api_key
        from pathlib import Path
        from quant_risk.config import CACHE_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "fred"
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    @property
    def day_count_convention(self) -> str:
        return "ACT/360"

    @property
    def currency(self) -> str:
        return "USD"

    def _get_series(self, series_id: str, last_n: int,
                    date: str | None = None) -> pd.Series:
        """
        Fetch a single FRED series with local history store.

        On a cache hit (any stored observation on or before date) the API is
        not called. On a miss, fetches last_n observations up to date from
        FRED, merges into the local store, and returns the full stored series.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. 'DGS10').
        last_n : int
            Observations to fetch on a cache miss.
        date : str or None
            Target date as ISO string. If None, always fetches fresh.
        """
        cache_name = f"fred_{series_id}"

        # Cache lookup when a specific date is requested
        if date is not None:
            cached = self._load_cache(cache_name)
            if cached is not None:
                cached.index = pd.to_datetime(cached.index)
                if not cached.index[cached.index <= pd.Timestamp(date)].empty:
                    logger.debug("FRED cache hit: %s date=%s", series_id, date)
                    return cached

        # Cache miss or date=None — fetch fresh from FRED
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": last_n,
        }
        if date is not None:
            params["observation_end"] = date

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
        fresh = pd.Series(records, name=series_id).sort_index()
        logger.info("FRED %s — %d observations fetched", series_id, len(fresh))

        # Merge into local store and return full series
        merged = self._merge_cache(cache_name, fresh.to_frame())
        return merged.iloc[:, 0].rename(series_id)

    def get_overnight_rate(self, last_n: int = 252,
                           date: str | None = None) -> pd.Series:
        """
        SOFR daily fixings. The USD OIS reference rate.

        Parameters
        ----------
        last_n : int
            Observations to fetch on a cache miss.
        date : str or None
            Target date as ISO string. Uses local store if available.

        Returns
        -------
        pd.Series indexed by date, values in percent.
        """
        return self._get_series("SOFR", last_n, date=date)

    def get_spot_rate(self, maturity: str = "10Y", last_n: int = 252,
                      date: str | None = None) -> pd.Series:
        """
        US Treasury constant maturity rate.

        Parameters
        ----------
        maturity : str
            One of 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y.
        last_n : int
            Observations to fetch on a cache miss.
        date : str or None
            Target date as ISO string. Uses local store if available.

        Returns
        -------
        pd.Series indexed by date, values in percent.
        """
        series_id = FRED_SERIES.get(maturity)
        if series_id is None:
            raise ValueError(
                f"Maturity '{maturity}' not available. "
                f"Choose from {[k for k in FRED_SERIES if k != 'SOFR']}"
            )
        return self._get_series(series_id, last_n, date=date)

    def get_full_curve(self, last_n: int = 5,
                       date: str | None = None) -> pd.DataFrame:
        """
        Full US Treasury curve across standard maturities.

        Parameters
        ----------
        last_n : int
            Observations to fetch per series on a cache miss.
        date : str or None
            Target date as ISO string. Uses local store if available.

        Returns
        -------
        pd.DataFrame indexed by date, columns are maturity labels,
        values in percent.
        """
        maturities = [k for k in FRED_SERIES if k != "SOFR"]
        series = {m: self.get_spot_rate(m, last_n=last_n, date=date)
                  for m in maturities}
        return pd.DataFrame(series)

    def get_fx_spot(self, pair: str = "EURUSD", last_n: int = 5,
                    date: str | None = None) -> pd.Series:
        """
        FX spot rate from FRED.

        Parameters
        ----------
        pair : str
            Currency pair. Supported: 'EURUSD', 'GBPUSD', 'USDJPY',
            'USDCHF', 'USDBRL'.
        last_n : int
            Observations to fetch on a cache miss.
        date : str or None
            Target date as ISO string. Uses local store if available.

        Returns
        -------
        pd.Series indexed by date, values as FX rate.
        """
        fx_series = {
            "EURUSD": "DEXUSEU",
            "GBPUSD": "DEXUSUK",
            "USDJPY": "DEXJPUS",
            "USDCHF": "DEXSZUS",
            "USDBRL": "DEXBZUS",
        }
        series_id = fx_series.get(pair.upper())
        if series_id is None:
            raise ValueError(
                f"Pair '{pair}' not supported. "
                f"Choose from {list(fx_series.keys())}"
            )
        return self._get_series(series_id, last_n, date=date)
    
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