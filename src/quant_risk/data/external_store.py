import pandas as pd
import yfinance as yf
from pathlib import Path
from quant_risk.config import CACHE_DIR
from quant_risk.data.external_registry import YFINANCE_SERIES

class ExternalStore:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR / "external"
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

    def _save_cache(self, name: str, s: pd.Series):
        s.to_frame(name=name).to_parquet(self._cache_path(name))

    # ----------------------------
    # public API
    # ----------------------------
    def get_gpr(self) -> pd.Series:
        cached = self._load_cache("GPR")
        if cached is not None:
            return cached["GPR"]

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
            return cached[name]

        ticker = YFINANCE_SERIES[name]
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        s = raw["Close"].squeeze().rename(name).dropna()

        self._save_cache(name, s)
        return s

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