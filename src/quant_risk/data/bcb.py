import requests
import pandas as pd
from quant_risk.data.base import CentralBankClient


BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# BCB SGS series codes
BCB_SERIES = {
    "SELIC": 11,    # Selic overnight target rate
    "CDI":   12,    # CDI interbank rate (DI futures settle here)
    "IPCA":  433,   # Inflation index -- useful for real rate curve
}

# ANBIMA government bond maturities (indicative, from ANBIMA API)
ANBIMA_MATURITIES = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]


class BCBClient(CentralBankClient):
    """
    Client for the Banco Central do Brasil (BCB) public API.

    Available data:
    - CDI daily rate (the BRL OIS-equivalent, DI futures settle here)
    - Selic overnight policy rate (set by COPOM, CDI tracks it closely)
    - IPCA monthly inflation (used for real rate curve, NTN-B pricing)

    Not yet implemented:
    - DI futures curve: available as manual CSV download from
      b3.com.br (Pesquisa por Pregão > Mercado de Derivativos -
      Taxas de Mercado para Swaps). Automation requires Selenium
      or a data vendor. In production this comes via Bloomberg.
    - NTN-B and LTN spot rates: free via ANBIMA API with CPF
      registration at anbima.com.br

    No API key required for BCB SGS series.

    Day count : BUS/252 -- critical difference from EUR/USD.
                Compounding is over business days, not calendar days.
    Overnight : CDI
    Currency  : BRL
    """

    @property
    def day_count_convention(self) -> str:
        return "BUS/252"

    @property
    def currency(self) -> str:
        return "BRL"

    def _get_series(self, code: int, last_n: int) -> pd.Series:
        """Fetch a single BCB SGS series."""
        url = BCB_BASE.format(code=code)
        params = {"formato": "json", "ultimos": last_n}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise ConnectionError(
                f"BCB API returned {response.status_code} "
                f"for series {code}"
            )
        data = response.json()
        records = {
            obs["data"]: float(obs["valor"].replace(",", "."))
            for obs in data
        }
        return pd.Series(records, name=str(code))

    def get_overnight_rate(self, last_n: int = 252) -> pd.Series:
        """
        CDI daily rate. The BRL OIS-equivalent reference rate.

        Note: CDI is quoted as an annual rate in BUS/252 convention.
        The daily compounding factor is (1 + CDI/100)^(1/252).

        Returns
        -------
        pd.Series indexed by date string (DD/MM/YYYY), values in percent.
        """
        return self._get_series(BCB_SERIES["CDI"], last_n)

    def get_selic(self, last_n: int = 252) -> pd.Series:
        """
        Selic overnight policy rate set by COPOM.
        CDI tracks Selic closely -- typically 10bps below.

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        return self._get_series(BCB_SERIES["SELIC"], last_n)

    def get_spot_rate(self, maturity: str = "10Y",
                      last_n: int = 252) -> pd.Series:
        """
        Brazilian government spot rate.

        Note: Full NTN-B and LTN curve data requires the ANBIMA
        API which needs registration. This method returns a
        placeholder until ANBIMA access is configured.

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        raise NotImplementedError(
            "Brazilian government spot rates require ANBIMA API access. "
            "Register at https://www.anbima.com.br/en_US/inform/apis.asp "
            "then implement using the ANBIMA market data endpoints. "
            "CDI curve via DI futures on B3 is the practical alternative."
        )

    def get_full_curve(self, last_n: int = 5) -> pd.DataFrame:
        """
        Placeholder -- see get_spot_rate note on ANBIMA.
        Returns CDI and Selic as a starting point.
        """
        return pd.DataFrame({
            "CDI":   self.get_overnight_rate(last_n),
            "SELIC": self.get_selic(last_n),
        })

    def get_ipca(self, last_n: int = 252) -> pd.Series:
        """
        IPCA inflation index monthly readings.
        Used for constructing the real rate curve (NTN-B pricing).

        Returns
        -------
        pd.Series indexed by date string, values in percent (monthly).
        """
        return self._get_series(BCB_SERIES["IPCA"], last_n)
    

    def load_di_curve(self, filepath: str) -> pd.DataFrame:
        """
        Load DI swap rates from a locally downloaded B3 CSV file.

        Download manually from b3.com.br:
        Pesquisa por Pregão > Mercado de Derivativos -
        Taxas de Mercado para Swaps > select date > download.

        Place the file in data/raw/ and pass the path here.

        Parameters
        ----------
        filepath : str
            Path to the B3 CSV file.

        Returns
        -------
        pd.DataFrame with columns [maturity, rate] where rate
        is in percent and maturity is in business days.
        """
        # B3 swap files are semicolon separated, latin-1 encoded
        df = pd.read_csv(
            filepath,
            sep=";",
            encoding="latin-1",
            decimal=",",
        )
        return df