import os
import pandas as pd
from pathlib import Path

from quant_risk.data.fed import FedClient
from quant_risk.data.fed_registry import FRED_SERIES


class FREDStore:
    """
    High-level macro data layer:
    - full history
    - caching
    - aligned panel construction
    """

    def __init__(self, api_key: str, cache_dir: str = "data/cache/fred"):
        self.client = FedClient(api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # caching layer
    # ----------------------------
    def _cache_path(self, name: str):
        return self.cache_dir / f"{name}.parquet"

    def _load_cache(self, name: str):
        path = self._cache_path(name)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def _save_cache(self, name: str, df: pd.Series):
        df.to_frame(name=name).to_parquet(self._cache_path(name))

    # ----------------------------
    # public API
    # ----------------------------
    def get_series(self, name: str) -> pd.Series:
        """
        name = logical name (not FRED code)
        """
        if name not in FRED_SERIES:
            raise ValueError(f"{name} not in registry")

        cached = self._load_cache(name)
        if cached is not None:
            return cached[name]

        series_id = FRED_SERIES[name]
        s = self.client._get_series(series_id, last_n=None)

        self._save_cache(name, s)
        return s

    # ----------------------------
    # panel builder
    # ----------------------------
    def build_panel(self, series_list: list[str]):
        data = {}

        for name in series_list:
            s = self.get_series(name)

            if len(s) == 0:
                continue  # skip broken series

            data[name] = s

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    # ----------------------------
    # alignment layer
    # ----------------------------
    def align(self, df: pd.DataFrame, freq: str = "B") -> pd.DataFrame:
        """
        Handles mixed frequency:
        - daily + monthly + irregular series
        """
        df = df.sort_index()

        # reindex to business days
        idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)

        df = df.reindex(idx)

        # forward fill macro series (standard in macro finance)
        df = df.ffill()

        return df