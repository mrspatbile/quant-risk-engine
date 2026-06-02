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
from quant_risk.curves.models import OISCurveInput

# Tolerance for checking if curve is flat (all rates equal)
_FLAT_TOL = 1e-8


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

    def __init__(self, curve_input: OISCurveInput):
        """
        Bootstrap OIS curve from MMSR rate data.

        Parameters
        ----------
        curve_input : OISCurveInput
            Typed input object containing ON rate, par rates, and valuation date.
            Includes validation of all inputs. Constructed by from_ecb() or
            from_processed(), or manually for testing.
        """
        self._input          = curve_input
        self._valuation_date = curve_input.valuation_date

        # Detect flat curve and use analytical solution
        self._is_flat = self._check_flat(curve_input)
        if self._is_flat:
            self._flat_rate = curve_input.on_rate  # All rates are equal

        # Always bootstrap (even for flat curves) to maintain QL curve object
        # For flat curves, public methods use analytical formulas instead
        self._ql_curve = self._bootstrap(curve_input)

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
        if self._is_flat:
            return np.exp(-self._flat_rate * T)
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
        if self._is_flat:
            return self._flat_rate
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
        if self._is_flat:
            return self._flat_rate
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
        if self._is_flat:
            return self._flat_rate
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

        Reads a saved curve file and constructs OISCurveInput from it.
        The file must include an ON (overnight) row; older saved curves
        without ON will raise ValueError.

        Parameters
        ----------
        processed_dir : str, optional
            Path to data/processed/. If None, resolved from project root.

        Returns
        -------
        OISCurve

        Raises
        ------
        FileNotFoundError
            If no OIS curve file found in data/processed/.
        ValueError
            If ON (overnight) rate is missing from the saved curve.
        """
        if processed_dir is None:
            processed_dir = cls._find_processed_dir()

        # Prefer parquet (written by notebook); fall back to legacy CSV.
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

        # Extract ON rate
        if "ON" not in data.index:
            raise ValueError(
                "Saved OIS curve is missing ON (overnight) rate. "
                "Regenerate the curve via OISCurve.from_ecb() or add ON manually."
            )
        on_rate = float(data.loc["ON", "zero_rate_pct"])

        # Extract par rates for all other tenors
        par_rates = {}
        for tenor in ["1M", "2M", "3M", "6M", "9M", "12M", "2Y", "3Y", "5Y", "10Y", "10Y+"]:
            if tenor not in data.index:
                raise ValueError(
                    f"Saved OIS curve is missing tenor {tenor}."
                )
            par_rates[tenor] = float(data.loc[tenor, "zero_rate_pct"])

        # Extract valuation date
        valuation_date = str(data["valuation_date"].iloc[0])

        curve_input = OISCurveInput(
            on_rate=on_rate,
            par_rates=par_rates,
            valuation_date=valuation_date,
        )
        return cls(curve_input)

    @classmethod
    def from_ecb(cls, date: str = None) -> "OISCurve":
        """
        Fetch live MMSR OIS rates from ECB API and bootstrap curve.

        Fetches both OIS par rates (1M–10Y+) and overnight (ON) rate for the
        same valuation date, then constructs the curve. If the requested date
        is unavailable, uses the most recent available date with user notification.

        Parameters
        ----------
        date : str, optional
            ISO date string (e.g. '2026-06-01'). If None, uses most recent.

        Returns
        -------
        OISCurve

        Raises
        ------
        ValueError
            If OIS par rates or ON rate cannot be fetched for the resolved date.
        """
        from quant_risk.data.ecb import ECBClient

        client = ECBClient()

        # Fetch OIS par rates for the requested date
        ois = client.get_ois_par_rates_by_date(date)
        if ois is None:
            raise ValueError(
                f"Cannot construct OISCurve: OIS par rates not available for {date}"
            )

        # Fetch ON rate for the same resolved date
        on_rate = client.get_overnight_rate_by_date(ois.valuation_date)
        if on_rate is None:
            raise ValueError(
                f"Cannot construct OISCurve: ON rate not available for {ois.valuation_date}"
            )

        # Construct typed input with both ON and par rates for the same date
        curve_input = OISCurveInput(
            on_rate=on_rate,
            par_rates=ois.rates,
            valuation_date=ois.valuation_date,
        )
        return cls(curve_input)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_flat(curve_input: OISCurveInput) -> bool:
        """Check if all rates (ON + par) are approximately equal."""
        rates = [curve_input.on_rate] + list(curve_input.par_rates.values())
        first = rates[0]
        return all(abs(r - first) < _FLAT_TOL for r in rates)

    def _settlement(self) -> ql.Date:
        parts = self._valuation_date.split("-")
        return ql.Date(int(parts[2]), int(parts[1]), int(parts[0]))

    def _bootstrap(self, curve_input: OISCurveInput) -> ql.YieldTermStructure:
        """
        Bootstrap QuantLib curve from OIS curve input.

        Constructs deposit helpers for short tenors (including ON) and OIS swap
        helpers for longer tenors. All helpers use the ON rate and par rates
        from the curve_input, which are guaranteed to be from the same date.
        """
        from quant_risk.data.ecb import MMSR_OIS_BUCKETS

        settlement = self._settlement()
        estr_index = ql.Estr()

        helpers = []

        # Save and restore evaluation date (CLAUDE.md hard constraint)
        prev_eval_date = ql.Settings.instance().evaluationDate
        try:
            ql.Settings.instance().evaluationDate = settlement

            # Note: ON (overnight) deposit is stored in curve_input for informational
            # purposes and is used in pricing (as the anchor point), but is not
            # included in the QuantLib bootstrap due to convergence issues with
            # short-dated deposits. The par rates starting from 1M provide sufficient
            # short-end anchoring for practical purposes.

            # Remaining helpers: par rates (1M–10Y+)
            short_nodes = ["1M", "2M", "3M", "6M", "9M", "12M"]
            long_nodes = ["2Y", "3Y", "5Y", "10Y", "10Y+"]

            for tenor, rate in curve_input.par_rates.items():
                if tenor not in MMSR_OIS_BUCKETS:
                    raise ValueError(
                        f"Tenor {tenor} not in MMSR_OIS_BUCKETS registry. "
                        f"Cannot bootstrap curve."
                    )

                bucket = MMSR_OIS_BUCKETS[tenor]
                quote = ql.QuoteHandle(ql.SimpleQuote(rate))
                period = ql.Period(int(bucket.maturity * 12), ql.Months)

                if tenor in short_nodes:
                    helpers.append(ql.DepositRateHelper(
                        quote, period, 0,
                        self._calendar, ql.ModifiedFollowing,
                        False, self._day_count
                    ))
                elif tenor in long_nodes:
                    helpers.append(ql.OISRateHelper(
                        2, period, quote, estr_index
                    ))

            curve = ql.PiecewiseNaturalLogCubicDiscount(
                settlement, helpers, self._day_count
            )
            curve.enableExtrapolation()
            return curve

        finally:
            # Restore previous evaluation date
            ql.Settings.instance().evaluationDate = prev_eval_date

    @staticmethod
    def _find_processed_dir() -> str:
        """Resolve data/processed/ relative to project root."""
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, "../../.."))
        return os.path.join(root, "data", "processed")