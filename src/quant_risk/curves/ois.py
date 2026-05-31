"""
OIS discount curve -- bootstrapped from ESTR market instruments via QuantLib.

Loads pre-bootstrapped curve data from data/processed/ or constructs
from raw MMSR OIS rates fetched via ECBClient.
"""

import os
import pandas as pd
import numpy as np
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve


class OISCurve(DiscountCurve):
    """
    ESTR OIS discount curve bootstrapped from ECB MMSR market data.

    Wraps a QuantLib PiecewiseNaturalLogCubicDiscount curve built
    from deposit and OIS swap rate helpers. Exposes the standard
    DiscountCurve interface for use by instrument pricers and
    risk calculators.

    Construction
    ------------
    OISCurve.from_processed()   -- load latest saved curve from data/processed/
    OISCurve.from_ecb()         -- fetch live from ECB API and bootstrap
    OISCurve(ois_data)          -- construct from a pre-loaded DataFrame

    Day count : ACT/360
    Index     : ESTR (ql.Estr())
    Calendar  : TARGET
    """

    _calendar   = ql.TARGET()
    _day_count  = ql.Actual360()

    def __init__(self, ois_data: pd.DataFrame):
        """
        Bootstrap OIS curve from MMSR rate data.

        Parameters
        ----------
        ois_data : pd.DataFrame
            DataFrame with columns [years, zero_rate_pct, discount_factor,
            valuation_date] indexed by maturity label.
            Produced by ECBClient.get_ois_rates() or loaded from
            data/processed/ois_curve_YYYY-MM-DD.csv.
        """
        self._ois_data       = ois_data
        self._valuation_date = ois_data["valuation_date"].iloc[0]
        self._ql_curve       = self._bootstrap(ois_data)

    # ------------------------------------------------------------------
    # DiscountCurve interface
    # ------------------------------------------------------------------

    @property
    def currency(self) -> str:
        return "EUR"

    @property
    def valuation_date(self) -> str:
        return self._valuation_date

    def discount(self, T: float) -> float:
        dt = self._settlement() + ql.Period(int(T * 365), ql.Days)
        return self._ql_curve.discount(dt)

    def zero_rate(self, T: float) -> float:
        """
        Continuously compounded zero rate at T.

        Returns
        -------
        float
            Zero rate, decimal (e.g. 0.025 for 2.5%).
        """
        dt = self._settlement() + ql.Period(int(T * 365), ql.Days)
        return self._ql_curve.zeroRate(dt, self._day_count, ql.Continuous).rate()

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Continuously compounded forward rate for [T1, T2].

        Returns
        -------
        float
            Forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        dt1 = self._settlement() + ql.Period(int(T1 * 365), ql.Days)
        dt2 = self._settlement() + ql.Period(int(T2 * 365), ql.Days)
        return self._ql_curve.forwardRate(dt1, dt2, self._day_count, ql.Continuous).rate()

    def instantaneous_forward(self, T: float) -> float:
        """
        Instantaneous forward rate at T via finite differences.
        f(0,T) = -d/dT ln P(0,T). Required for Hull-White calibration.

        Returns
        -------
        float
            Instantaneous forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        dT = 1 / 365
        settlement = self._settlement()
        dt1 = settlement + ql.Period(max(int((T - dT) * 365), 1), ql.Days)
        dt2 = settlement + ql.Period(int((T + dT) * 365), ql.Days)
        p1  = self._ql_curve.discount(dt1)
        p2  = self._ql_curve.discount(dt2)
        return -(np.log(p2) - np.log(p1)) / (2 * dT)

    # ------------------------------------------------------------------
    # class methods -- alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_processed(cls, processed_dir: str = None) -> "OISCurve":
        """
        Load the most recent OIS curve from data/processed/.

        Parameters
        ----------
        processed_dir : str, optional
            Path to data/processed/. If None, resolved from project root.

        Returns
        -------
        OISCurve
        """
        if processed_dir is None:
            processed_dir = cls._find_processed_dir()

        # Prefer parquet (written by notebook 02); fall back to legacy CSV.
        parquet_files = sorted([
            f for f in os.listdir(processed_dir)
            if f.startswith("ois_curve_") and f.endswith(".parquet")
        ])
        csv_files = sorted([
            f for f in os.listdir(processed_dir)
            if f.startswith("ois_curve_") and f.endswith(".csv")
        ])

        if parquet_files:
            path = os.path.join(processed_dir, parquet_files[-1])
            data = pd.read_parquet(path)
        elif csv_files:
            path = os.path.join(processed_dir, csv_files[-1])
            data = pd.read_csv(path, index_col="maturity")
        else:
            raise FileNotFoundError(
                "No OIS curve file found in data/processed/. "
                "Run notebook 02_bootstrapping_ois.ipynb first."
            )
        return cls(data)

    @classmethod
    def from_ecb(cls) -> "OISCurve":
        """
        Fetch live MMSR OIS rates from ECB API and bootstrap curve.

        Returns
        -------
        OISCurve
        """
        from quant_risk.data.ecb import ECBClient
        client   = ECBClient()
        ois_data = client.get_ois_rates(last_n=5)
        latest   = ois_data.iloc[[-1]].T
        latest.columns = ["zero_rate_pct"]
        latest.index.name = "maturity"

        from quant_risk.data.ecb import MMSR_OIS_BUCKETS
        latest["years"] = [
            MMSR_OIS_BUCKETS[m].maturity
            for m in latest.index
        ]
        latest["valuation_date"] = ois_data.index[-1]
        return cls(latest)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _settlement(self) -> ql.Date:
        parts = self._valuation_date.split("-")
        return ql.Date(int(parts[2]), int(parts[1]), int(parts[0]))

    def _bootstrap(self, ois_data: pd.DataFrame) -> ql.YieldTermStructure:
        """Bootstrap QuantLib curve from MMSR OIS rates."""
        from quant_risk.data.ecb import MMSR_OIS_BUCKETS

        settlement  = self._settlement()
        estr_index  = ql.Estr()
        short_nodes = ["ON", "1M", "2M", "3M", "6M", "9M", "12M"]
        long_nodes  = ["2Y", "3Y", "5Y", "10Y", "10Y+"]

        ql.Settings.instance().evaluationDate = settlement

        helpers = []

        for label in ois_data.index:
            if label not in MMSR_OIS_BUCKETS:
                continue
            bucket  = MMSR_OIS_BUCKETS[label]
            rate    = ois_data.loc[label, "zero_rate_pct"]   # already decimal
            tenor   = ql.Period(int(bucket.maturity * 12), ql.Months)
            quote   = ql.QuoteHandle(ql.SimpleQuote(rate))

            if label in short_nodes:
                helpers.append(ql.DepositRateHelper(
                    quote, tenor, 0,
                    self._calendar, ql.ModifiedFollowing,
                    False, self._day_count
                ))
            elif label in long_nodes:
                helpers.append(ql.OISRateHelper(
                    2, tenor, quote, estr_index
                ))

        curve = ql.PiecewiseNaturalLogCubicDiscount(
            settlement, helpers, self._day_count
        )
        curve.enableExtrapolation()
        return curve

    @staticmethod
    def _find_processed_dir() -> str:
        """Resolve data/processed/ relative to project root."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "../../.."))
        return os.path.join(root, "data", "processed")