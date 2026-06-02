import pandas as pd
import yfinance as yf
from quant_risk.data.external_registry import YFINANCE_SERIES

# for fama-french factors download fields and unzip on the fly
import urllib.request
import zipfile
import io


class ExternalDataClient:
    """Stateless fetcher for external market data via yfinance and Kenneth French."""

    def __init__(self):
        pass

    # ----------------------------
    # public API
    # ----------------------------
    def get_gpr(self) -> pd.Series:
        """Fetch Geopolitical Risk Index from Matteo Iacoviello's website."""
        url = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"
        gpr = pd.read_csv(url)
        gpr["Date"] = pd.to_datetime(gpr["Date"])
        gpr = gpr.set_index("Date")
        s = gpr["GPR_AI"].rename("GPR")
        return s

    def get_yfinance(self, name: str, start: str = "2000-01-01") -> pd.Series:
        """Fetch price series from yfinance (equity, ETF, index)."""
        if name not in YFINANCE_SERIES:
            raise ValueError(f"{name} not in YFINANCE_SERIES registry")

        ticker = YFINANCE_SERIES[name]
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        s = raw["Close"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.rename(name).dropna()
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
        """Fetch Fama-French 5 factor model + momentum (daily or monthly)."""
        if frequency == "daily":
            url_ff5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
            url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
        else:
            url_ff5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
            url_mom = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

        ff5 = self._fetch_french_zip(url_ff5, header_starts_with=',Mkt-RF')
        mom = self._fetch_french_zip(url_mom, header_starts_with=',Mom')

        df = ff5.join(mom[['Mom']], how='left')
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
