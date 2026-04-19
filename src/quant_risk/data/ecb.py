# ECB WAF blocks requests without a realistic browser User-Agent.
# The headers below were tested April 2026 -- may need updating
# if the ECB rotates their WAF rules.

import requests
import pandas as pd
from quant_risk.data.base import CentralBankClient


BASE_URL = "https://data-api.ecb.europa.eu/service/data"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Standard maturities available from ECB yield curve dataset
GOV_MATURITIES = {
    "3M":  "SR_3M",
    "6M":  "SR_6M",
    "1Y":  "SR_1Y",
    "2Y":  "SR_2Y",
    "5Y":  "SR_5Y",
    "10Y": "SR_10Y",
    "20Y": "SR_20Y",
    "30Y": "SR_30Y",
}


class ECBClient(CentralBankClient):
    """
    Client for the ECB Statistical Data Warehouse REST API.

    Available data:
    - ESTR daily fixings (overnight OIS rate)
    - NSS curve parameters (beta0-3, tau1-2)
    - Euro area government spot rates (3M to 30Y)
    - EURIBOR panel rates (via FM dataset)

    Note: ECB WAF blocks requests without a realistic browser
    User-Agent. Headers below were tested April 2026.

    Day count : ACT/360
    Overnight : ESTR
    Currency  : EUR
    """

    @property
    def day_count_convention(self) -> str:
        return "ACT/360"

    @property
    def currency(self) -> str:
        return "EUR"

    def _get(self, dataset: str, series_key: str, params: dict) -> dict:
        """Raw GET request. Raises on non-200 status."""
        url = f"{BASE_URL}/{dataset}/{series_key}"
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            raise ConnectionError(
                f"ECB API returned {response.status_code} for {url}"
            )
        return response.json()

    def _parse_observations(self, data: dict) -> pd.Series:
        """
        Extract observations from ECB JSON response into a pandas Series
        indexed by date string.
        """
        series = data["dataSets"][0]["series"]
        first_series = next(iter(series.values()))
        observations = first_series["observations"]
        time_periods = (
            data["structure"]["dimensions"]["observation"][0]["values"]
        )
        records = {}
        for idx, period in enumerate(time_periods):
            obs = observations.get(str(idx))
            if obs and obs[0] is not None:
                records[period["id"]] = obs[0]
        return pd.Series(records, name="value")

    def get_overnight_rate(self, last_n: int = 252) -> pd.Series:
        """
        ESTR daily fixings. The EUR OIS reference rate.

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        data = self._get(
            dataset="FM",
            series_key="B.U2.EUR.SF.B.B.TC.EUR.IV.Z.Z.A",
            params={"format": "jsondata", "lastNObservations": last_n},
        )
        return self._parse_observations(data)

    def get_spot_rate(self, maturity: str = "10Y",
                      last_n: int = 252) -> pd.Series:
        """
        ECB euro area government bond spot rate at a given maturity.

        Parameters
        ----------
        maturity : str
            One of 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y
        last_n : int
            Number of observations to fetch.

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        code = GOV_MATURITIES.get(maturity)
        if code is None:
            raise ValueError(
                f"Maturity '{maturity}' not available. "
                f"Choose from {list(GOV_MATURITIES.keys())}"
            )
        data = self._get(
            dataset="YC",
            series_key=f"B.U2.EUR.4F.G_N_A.SV_C_YM.{code}",
            params={"format": "jsondata", "lastNObservations": last_n},
        )
        return self._parse_observations(data)

    def get_full_curve(self, last_n: int = 5) -> pd.DataFrame:
        """
        Full euro area government spot curve across all
        standard maturities.

        Returns
        -------
        pd.DataFrame
            Indexed by date, columns are maturity labels (3M..30Y),
            values in percent.
        """
        series = {}
        for maturity in GOV_MATURITIES:
            series[maturity] = self.get_spot_rate(maturity, last_n=last_n)
        return pd.DataFrame(series)

    def get_nss_parameters(self, last_n: int = 5) -> pd.DataFrame:
        """
        ECB published Nelson-Siegel-Svensson parameters.

        Returns
        -------
        pd.DataFrame with columns [beta0, beta1, beta2, beta3, tau1, tau2]
        indexed by date.
        """
        param_codes = {
            "beta0": "B.U2.EUR.4F.G_N_A.SV_C_YM.BETA0",
            "beta1": "B.U2.EUR.4F.G_N_A.SV_C_YM.BETA1",
            "beta2": "B.U2.EUR.4F.G_N_A.SV_C_YM.BETA2",
            "beta3": "B.U2.EUR.4F.G_N_A.SV_C_YM.BETA3",
            "tau1":  "B.U2.EUR.4F.G_N_A.SV_C_YM.TAU1",
            "tau2":  "B.U2.EUR.4F.G_N_A.SV_C_YM.TAU2",
        }
        series = {}
        for name, code in param_codes.items():
            data = self._get(
                dataset="YC",
                series_key=code,
                params={"format": "jsondata", "lastNObservations": last_n},
            )
            series[name] = self._parse_observations(data)
        return pd.DataFrame(series)
