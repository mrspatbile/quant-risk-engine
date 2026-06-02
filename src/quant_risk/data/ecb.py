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
from quant_risk.logging import get_logger

logger = get_logger(__name__)


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
    "1M"   : RateBucket(code="FC", maturity=1/12),
    "2M"   : RateBucket(code="FD", maturity=2/12),
    "3M"   : RateBucket(code="FE", maturity=3/12),
    "6M"   : RateBucket(code="FF", maturity=6/12),
    "9M"   : RateBucket(code="FG", maturity=9/12),
    "12M"  : RateBucket(code="FH", maturity=1.0),
    "2Y"   : RateBucket(code="FI", maturity=2.0),
    "3Y"   : RateBucket(code="FJ", maturity=3.0),
    "5Y"   : RateBucket(code="FK", maturity=5.0),
    "10Y"  : RateBucket(code="FL", maturity=10.0),
    "10Y+" : RateBucket(code="FM", maturity=15.0),
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

    def __init__(self) -> None:
        pass

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
        data = response.json()
        try:
            n_obs = len(
                next(iter(data["dataSets"][0]["series"].values()))["observations"]
            )
        except (KeyError, StopIteration):
            n_obs = 0
        logger.info("ECB %s/%s — %d observations", dataset, series_key, n_obs)
        return data

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
        pd.Series indexed by date string, values decimal (e.g. 0.025 for 2.5%).
        """
        data = self._get(
            dataset="EST",
            series_key="B.EU000A2X2A25.WT",
            params={"format": "jsondata", "lastNObservations": last_n},
        )
        return self._parse_observations(data) / 100
    
    def get_overnight_rate_single_date(self, date: str) -> float:
        """
        Get ESTR rate for a specific date.
        
        Parameters
        ----------
        date : str
            ISO date string, e.g. '2025-12-31'
        
        Returns
        -------
        float
            ESTR rate in decimal (e.g. 0.025 for 2.5%)
        """
        data = self._get(
            dataset="EST",
            series_key="B.EU000A2X2A25.WT",
            params={"format": "jsondata", "lastNObservations": 365},
        )
        series = self._parse_observations(data) / 100
        
        return series.loc[date]

    def get_spot_rate(self, maturity: str = "10Y",
                      last_n: int = 252,
                      rating: str = "AAA") -> pd.Series:
        """
        ECB euro area government bond spot rate at a given maturity.

        Parameters
        ----------
        maturity : str
            One of 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y
        last_n : int
            Number of observations to fetch.
        rating : str
            'AAA' for triple-A rated issuers only (G_N_A),
            'ALL' for all issuers all ratings (G_N_C).

        Returns
        -------
        pd.Series indexed by date string, values decimal (e.g. 0.025 for 2.5%).
        """
        if maturity not in GOV_MATURITIES:
            raise ValueError(
                f"Maturity '{maturity}' not available. "
                f"Choose from {list(GOV_MATURITIES.keys())}"
            )
        instrument = "G_N_A" if rating == "AAA" else "G_N_C"
        data = self._get(
            dataset="YC",
            series_key=f"B.U2.EUR.4F.{instrument}.SV_C_YM.{GOV_MATURITIES[maturity].code}",
            params={"format": "jsondata", "lastNObservations": last_n},
        )
        return self._parse_observations(data) / 100

    def get_full_curve(self, last_n: int = 5,
                       rating: str = "AAA") -> pd.DataFrame:
        """
        Full euro area government spot curve across all
        standard maturities.

        Parameters
        ----------
        last_n : int
            Number of observations to fetch.
        rating : str
            'AAA' for triple-A rated issuers only (G_N_A),
            'ALL' for all issuers all ratings (G_N_C).

        Returns
        -------
        pd.DataFrame
            Indexed by date, columns are maturity labels (3M..30Y),
            values decimal (e.g. 0.025 for 2.5%).
        """
        series = {}
        for maturity in GOV_MATURITIES:
            series[maturity] = self.get_spot_rate(
                maturity, last_n=last_n, rating=rating
            )
        return pd.DataFrame(series)

    def get_nss_parameters(self, last_n: int = 5,
                           rating: str = "AAA",
                           date: str | None = None) -> pd.DataFrame:
        """
        ECB published Nelson-Siegel-Svensson parameters.

        Parameters
        ----------
        last_n : int
            Number of observations to fetch. Default 5 — sufficient to cover
            weekends and ECB public holidays.
        rating : str
            'AAA' for triple-A rated issuers only (G_N_A),
            'ALL' for all issuers all ratings (G_N_C).
        date : str or None
            Target date as ISO string ('YYYY-MM-DD'). If given, the ECB API
            is called with endPeriod=date so results are anchored to that date.

        Returns
        -------
        pd.DataFrame with columns [beta0, beta1, beta2, beta3, tau1, tau2]
        indexed by date.
        """
        instrument = "G_N_A" if rating == "AAA" else "G_N_C"
        param_codes = {
            "beta0": f"B.U2.EUR.4F.{instrument}.SV_C_YM.BETA0",
            "beta1": f"B.U2.EUR.4F.{instrument}.SV_C_YM.BETA1",
            "beta2": f"B.U2.EUR.4F.{instrument}.SV_C_YM.BETA2",
            "beta3": f"B.U2.EUR.4F.{instrument}.SV_C_YM.BETA3",
            "tau1":  f"B.U2.EUR.4F.{instrument}.SV_C_YM.TAU1",
            "tau2":  f"B.U2.EUR.4F.{instrument}.SV_C_YM.TAU2",
        }
        query_params: dict = {"format": "jsondata", "lastNObservations": last_n}
        if date is not None:
            query_params["endPeriod"] = date

        series = {}
        for name, code in param_codes.items():
            data = self._get(dataset="YC", series_key=code, params=query_params)
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
            values decimal (e.g. 0.025 for 2.5%).
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
                series[tenor] = self._parse_observations(data) / 100
            except ConnectionError:
                series[tenor] = pd.Series(dtype=float)

        return pd.DataFrame(series)

    def get_overnight_rate_by_date(self, date: str = None) -> float | None:
        """
        Get ESTR overnight rate for a specific date.

        If the exact date is unavailable, returns None and informs the caller
        of the most recent available date. Callers decide whether to retry
        or use a different date.

        Parameters
        ----------
        date : str, optional
            ISO date string (e.g. '2026-06-01'). If None, returns most recent.

        Returns
        -------
        float | None
            ESTR rate in decimal (e.g. 0.025 for 2.5%), or None if exact date
            unavailable and caller was given the most recent available date.
        """
        data = self._get(
            dataset="EST",
            series_key="B.EU000A2X2A25.WT",
            params={
                "format": "jsondata",
                "lastNObservations": 365,
            } if date is None else {
                "format": "jsondata",
                "endPeriod": date,
            },
        )
        series = self._parse_observations(data) / 100

        if date is None:
            # No specific date requested, return most recent
            return float(series.iloc[-1])

        # Date was requested; check if we got it
        if date in series.index:
            return float(series.loc[date])

        # Exact date not available; inform caller
        most_recent_date = series.index[-1]
        print(
            f"Date {date} not available in ECB data. "
            f"Most recent available: {most_recent_date}"
        )
        return None

    def get_ois_par_rates_by_date(self, date: str = None) -> "OISParRates | None":
        """
        Get OIS par rates (1M–10Y+) for a specific date.

        All tenors are fetched with the same endPeriod, ensuring they align
        to a single valuation date. This fixes the legacy issue where
        different tenors could have different dates.

        If the exact date is unavailable, returns None and informs the caller
        of the most recent available date.

        Parameters
        ----------
        date : str, optional
            ISO date string (e.g. '2026-06-01'). If None, uses most recent.

        Returns
        -------
        OISParRates | None
            Object with rates dict and valuation_date, or None if exact date
            unavailable (caller informed of most recent available).

        Raises
        ------
        ValueError
            If any tenor fails to return a value for the resolved date.
        ConnectionError
            If ECB API is unreachable.
        """
        from quant_risk.curves.models import OISParRates

        series_dict = {}
        resolved_date = None

        for tenor, bucket in MMSR_OIS_BUCKETS.items():
            series_key = (
                f"B.U2._X._Z.S1ZV._Z.O._X.WR._X.{bucket.code}._Z._Z.EUR._Z"
            )

            data = self._get(
                dataset="MMSR",
                series_key=series_key,
                params={
                    "format": "jsondata",
                    "lastNObservations": 1 if date is None else None,
                    "endPeriod": date,
                } if date is not None else {
                    "format": "jsondata",
                    "lastNObservations": 1,
                },
            )

            series = self._parse_observations(data) / 100

            if len(series) == 0:
                raise ValueError(
                    f"Tenor {tenor} returned no data for date {date}"
                )

            # Track the resolved date from the first tenor
            if resolved_date is None:
                resolved_date = series.index[-1]
                # If caller requested a date and we got a different one, inform them
                if date is not None and resolved_date != date:
                    print(
                        f"Date {date} not available in ECB data. "
                        f"Using most recent: {resolved_date}"
                    )

            # Verify all tenors resolved to the same date
            actual_date = series.index[-1]
            if actual_date != resolved_date:
                raise ValueError(
                    f"Tenor {tenor} resolved to {actual_date} "
                    f"but other tenors resolved to {resolved_date}. "
                    f"Cannot construct curve with misaligned dates."
                )

            series_dict[tenor] = float(series.iloc[-1])

        return OISParRates(rates=series_dict, valuation_date=resolved_date)
