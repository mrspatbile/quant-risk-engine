"""
ECB Statistical Data Warehouse client.

Available data:
- ESTR daily fixings (overnight OIS rate)
- NSS curve parameters (beta0-3, tau1-2)
- Euro area government spot rates (3M to 30Y)
- MMSR OIS weighted average rates (1M to 10Y)

Note: ECB WAF blocks requests without a realistic browser
User-Agent. Headers below were tested April 2026.

Day count : ACT/360
Overnight : ESTR
Currency  : EUR
"""

# standard library
from dataclasses import dataclass

# third party
import pandas as pd
import requests

# internal
from quant_risk.data.base import CentralBankClient


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

BASE_URL = "https://data-api.ecb.europa.eu/service/data"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

@dataclass
class RateBucket:
    """Rate curve bucket -- API series code and representative maturity in years."""
    code: str
    maturity: float


GOV_MATURITIES = {
    "3M":  RateBucket(code="SR_3M",  maturity=3/12),
    "6M":  RateBucket(code="SR_6M",  maturity=6/12),
    "1Y":  RateBucket(code="SR_1Y",  maturity=1.0),
    "2Y":  RateBucket(code="SR_2Y",  maturity=2.0),
    "5Y":  RateBucket(code="SR_5Y",  maturity=5.0),
    "10Y": RateBucket(code="SR_10Y", maturity=10.0),
    "20Y": RateBucket(code="SR_20Y", maturity=20.0),
    "30Y": RateBucket(code="SR_30Y", maturity=30.0),
}

MMSR_OIS_BUCKETS = {
    "1M"  : RateBucket(code="FC", maturity=1/12),
    "2M"  : RateBucket(code="FD", maturity=2/12),
    "3M"  : RateBucket(code="FE", maturity=3/12),
    "6M"  : RateBucket(code="FF", maturity=6/12),
    "9M"  : RateBucket(code="FG", maturity=9/12),
    "2Y"  : RateBucket(code="FI", maturity=2.0),
    "3Y"  : RateBucket(code="FJ", maturity=3.0),
    "5Y"  : RateBucket(code="FK", maturity=5.0),
    "10Y" : RateBucket(code="FM", maturity=10.0),
}


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------

class ECBClient(CentralBankClient):
    """
    ECB Statistical Data Warehouse REST API client.

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

        Series: EST/B.EU000A2X2A25.WT
        Source: ECB Statistical Data Warehouse, EST dataset.

        Returns
        -------
        pd.Series indexed by date string, values in percent.
        """
        data = self._get(
            dataset="EST",
            series_key="B.EU000A2X2A25.WT",
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
        if maturity not in GOV_MATURITIES:
            raise ValueError(
                f"Maturity '{maturity}' not available. "
                f"Choose from {list(GOV_MATURITIES.keys())}"
            )
        data = self._get(
            dataset="YC",
            series_key=f"B.U2.EUR.4F.G_N_A.SV_C_YM.{GOV_MATURITIES[maturity].code}",
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
    
    def get_ois_rates(self, last_n: int = 5) -> pd.DataFrame:
        """
        ECB MMSR OIS weighted average rates by maturity bucket.

        Source: ECB Money Market Statistical Reporting (MMSR),
        regulation EU/2016/867. Weighted average fixed rates on
        actual executed ESTR OIS wholesale transactions, aggregated
        by maturity bucket. Published with approximately 3-week lag.

        These are genuine executed transactions -- not indicative
        quotes -- covering tenors from 1 month to beyond 10 years.

        Returns
        -------
        pd.DataFrame
            Indexed by date, columns are tenor labels (1M..10Y),
            values in percent.
        """
        series = {}
        for tenor, bucket in MMSR_OIS_BUCKETS.items():
            series_key = (
                f"B.U2._X._Z.S1ZV._Z.O._X.WR._X.{bucket.code}._Z._Z.EUR._Z"
            )
            try:
                data = self._get(
                    dataset="MMSR",
                    series_key=series_key,
                    params={
                        "format": "jsondata",
                        "lastNObservations": last_n,
                    },
                )
                series[tenor] = self._parse_observations(data)
            except ConnectionError:
                series[tenor] = pd.Series(dtype=float)

        return pd.DataFrame(series)
