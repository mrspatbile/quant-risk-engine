import pandas as pd
import yfinance as yf

class ExternalStore:
    def get_gpr(self):
        url = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_data_daily.csv"

        gpr = pd.read_csv(url)
        gpr["Date"] = pd.to_datetime(gpr["Date"])
        gpr = gpr.set_index("Date")

        return gpr[["GPR_AI"]].rename(columns={"GPR_AI": "GPR"})
    
    def get_yfinance(self, ticker: str, start: str = "2000-01-01", column: str = "Close") -> pd.Series:
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        return (
            raw[column].squeeze()
            .rename(ticker)
            .dropna()
        )