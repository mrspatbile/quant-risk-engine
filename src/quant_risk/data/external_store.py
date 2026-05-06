import pandas as pd
import yfinance as yf
from pathlib import Path
from quant_risk.config import CACHE_DIR
from quant_risk.data.external_registry import YFINANCE_SERIES

# for fama-french factors dowwnload fiels and unzip on the fly, then cache as parquet for future use
import urllib.request
import zipfile
import io

from quant_risk.data.cache_mixin import CacheMixin

class ExternalStore(CacheMixin):

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "external"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # # ----------------------------
    # # caching layer
    # # ----------------------------
    # def _cache_path(self, name: str):
    #     return self.cache_dir / f"{name}.parquet"

    # # def _load_cache(self, name: str):
    # #     path = self._cache_path(name)
    # #     if path.exists():
    # #         return pd.read_parquet(path)
    # #     return None


    # def _load_cache(self, name: str) -> pd.Series | pd.DataFrame | None:
    #     path = self._cache_path(name)
    #     if path.exists():
    #         df = pd.read_parquet(path)
    #         if df.shape[1] == 1:
    #             return df.iloc[:, 0].rename(df.columns[0])
    #         return df
    #     return None


    # # def _save_cache(self, name: str, s: pd.Series):
    # #     s.to_frame(name=name).to_parquet(self._cache_path(name))


    # def _save_cache(self, name: str, s: pd.Series | pd.DataFrame):
    #     path = self._cache_path(name)
    #     if isinstance(s, pd.Series):
    #         s.to_frame(name=name).to_parquet(path)
    #     else:
    #         s.to_parquet(path)


    # def clear_cache(self, names: str | list[str] | None = None) -> None:
    #     """
    #     Delete cached parquet files.
    #     If names is a string, deletes that series.
    #     If names is a list, deletes each series in the list.
    #     If names is None, clears the entire cache.

    #     Usage:

    #     store.clear_cache()                            # nuke all
    #     store.clear_cache("HY_OAS")                    # single
    #     store.clear_cache(["HY_OAS", "IG_OAS", "DXY"]) # list
    #     """
    #     if names is None:
    #         files = list(self.cache_dir.glob("*.parquet"))
    #         for f in files:
    #             f.unlink()
    #         print(f"cleared {len(files)} files from cache")
    #     else:
    #         if isinstance(names, str):
    #             names = [names]
    #         for name in names:
    #             path = self._cache_path(name)
    #             if path.exists():
    #                 path.unlink()
    #                 print(f"cleared: {name}")
    #             else:
    #                 print(f"not found in cache: {name}")

    # ----------------------------
    # public API
    # ----------------------------
    def get_gpr(self) -> pd.Series:
        cached = self._load_cache("GPR")
        if cached is not None:
            return cached

        url = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"
        gpr = pd.read_csv(url)
        gpr["Date"] = pd.to_datetime(gpr["Date"])
        gpr = gpr.set_index("Date")
        s = gpr["GPR_AI"].rename("GPR")

        self._save_cache("GPR", s)
        return s

    def get_yfinance(self, name: str, start: str = "2000-01-01") -> pd.Series:
        if name not in YFINANCE_SERIES:
            raise ValueError(f"{name} not in YFINANCE_SERIES registry")

        cached = self._load_cache(name)
        if cached is not None:
            return cached

        ticker = YFINANCE_SERIES[name]
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        s = raw["Close"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.rename(name).dropna()

        self._save_cache(name, s)
        return s


    def _fetch_french_zip(self, url: str, header_starts_with: str) -> pd.DataFrame:
        """Parse a Kenneth French zip CSV, skipping the text header block."""
        import urllib.request
        import zipfile
        import io

        with urllib.request.urlopen(url) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
            with zf.open(csv_name) as f:
                raw = f.read().decode('utf-8')

        lines = raw.splitlines()
        start = next(
            i for i, line in enumerate(lines)
            if line.strip().startswith(header_starts_with)
        )

        df = pd.read_csv(
            io.StringIO('\n'.join(lines[start:])),
            index_col=0,
            header=0,
        )
        df = df.dropna()
        df.columns = df.columns.str.strip()
        df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m%d')
        df = df.apply(pd.to_numeric, errors='coerce') / 100
        df = df[df.index >= '2000-01-01']
        return df.dropna(how='all')


    def get_fama_french(self, frequency: str = "daily") -> pd.DataFrame:
        cache_key = f"FF_{frequency}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        if frequency == "daily":
            url_ff5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
            url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        else:
            url_ff5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
            url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

        ff5 = self._fetch_french_zip(url_ff5, header_starts_with=',Mkt-RF')
        mom = self._fetch_french_zip(url_mom, header_starts_with=',Mom')

        df = ff5.join(mom[['Mom']], how='left')

        self._save_cache(cache_key, df)
        return df

    # ----------------------------
    # panel builder
    # ----------------------------
    def build_panel(self, series_list: list[str], start: str = "2000-01-01") -> pd.DataFrame:
        data = {}
        for name in series_list:
            try:
                data[name] = self.get_yfinance(name, start=start)
            except Exception as e:
                print(f"failed {name}: {e}")

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()