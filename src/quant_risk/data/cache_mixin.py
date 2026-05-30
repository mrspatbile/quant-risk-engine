from pathlib import Path
import pandas as pd
from quant_risk.logging import get_logger

logger = get_logger(__name__)


class CacheMixin:
    """
    Mixin providing a complete parquet cache layer.
    Requires self.cache_dir (Path) to be set by the inheriting class.

    Methods
    -------
    _cache_path  -- resolve parquet file path for a given series name
    _load_cache  -- load from parquet, returns Series or DataFrame
    _save_cache  -- save Series or DataFrame to parquet
    clear_cache  -- delete one, several, or all cached files

    Usage
    -----
    class MyStore(CacheMixin):
        def __init__(self):
            self.cache_dir = Path("some/cache/dir")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    """

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.parquet"

    def _load_cache(self, name: str) -> pd.Series | pd.DataFrame | None:
        path = self._cache_path(name)
        if not path.exists():
            logger.debug("cache miss: %s", name)
            return None
        df = pd.read_parquet(path)
        logger.debug("cache hit: %s (%d rows)", name, len(df))
        if df.shape[1] == 1:
            return df.iloc[:, 0].rename(df.columns[0])
        return df

    def _save_cache(self, name: str, s: pd.Series | pd.DataFrame) -> None:
        path = self._cache_path(name)
        if isinstance(s, pd.Series):
            s.to_frame(name=name).to_parquet(path)
        else:
            s.to_parquet(path)
        logger.debug("cache write: %s (%d rows)", name, len(s))

    def _merge_cache(self, name: str, new_data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Append new rows to the local history store, deduplicate, sort by date.

        Implements the append-only pattern for static historical facts (e.g. ECB
        NSS parameters): once a row for a given date is stored it is correct
        forever. New observations are appended, existing dates are overwritten
        with the fresh value (keep='last'), and the index is always sorted.

        Parameters
        ----------
        name : str
            Cache key (parquet filename without extension).
        new_data : pd.DataFrame or pd.Series
            Rows to merge in. Must have a DatetimeIndex or date-parseable index.

        Returns
        -------
        pd.DataFrame or pd.Series
            The full merged, sorted store as returned by _load_cache.
        """
        existing = self._load_cache(name)
        if existing is not None:
            combined = pd.concat([existing, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = new_data.copy() if isinstance(new_data, pd.DataFrame) else new_data.to_frame()
            combined = combined.sort_index()
        self._save_cache(name, combined)
        logger.debug("cache merge: %s now %d rows", name, len(combined))
        return self._load_cache(name)

    def clear_cache(self, names: str | list[str] | None = None) -> None:
        """
        Delete cached parquet files.

        Parameters
        ----------
        names : str, list of str, or None
            If None, clears entire cache.
            If str, deletes that single series.
            If list, deletes each series in the list.

        Examples
        --------
        store.clear_cache()
        store.clear_cache("HY_OAS")
        store.clear_cache(["HY_OAS", "IG_OAS", "DXY"])
        """
        if names is None:
            files = list(self.cache_dir.glob("*.parquet"))
            for f in files:
                f.unlink()
            print(f"cleared {len(files)} files from cache")
        else:
            if isinstance(names, str):
                names = [names]
            for name in names:
                path = self._cache_path(name)
                if path.exists():
                    path.unlink()
                    print(f"cleared: {name}")
                else:
                    print(f"not found in cache: {name}")