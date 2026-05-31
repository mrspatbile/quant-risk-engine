# src/quant_risk/curves/array_curve.py

"""
ArrayCurve — DiscountCurve built from arbitrary maturity/rate numpy arrays.

Intended uses
-------------
1. Downstream applications (banking-risk) that receive a maturity grid and
   zero rates from an external source and need a proper DiscountCurve.

2. Sampling an analytical curve (NSSCurve) onto a QuantLib-backed grid for
   use in QuantLib pricers that require a YieldTermStructureHandle:

       T = np.linspace(0.25, 30, 120)
       r = np.array([nss.zero_rate(t) / 100 for t in T])
       ql_curve = ArrayCurve(T, r).ql_curve

Rate convention
---------------
zero_rates must be in **decimal** (e.g. 0.025 for 2.5%), consistent with
QuantLib internals. This differs from the rest of the library where rates
are in percent at public boundaries. The docstring documents this explicitly.
"""

from __future__ import annotations

import numpy as np
import QuantLib as ql

from quant_risk.curves.base import DiscountCurve

_VALID_INTERPOLATIONS = frozenset({"log_linear", "linear", "cubic"})


class ArrayCurve(DiscountCurve):
    """
    QuantLib-backed discount curve built from maturity/rate arrays.

    Wraps ql.DiscountCurve (log_linear) or ql.ZeroCurve (linear, cubic)
    and exposes the full DiscountCurve interface.

    Construction
    ------------
    ArrayCurve(maturities, zero_rates)
    curve_from_arrays(maturities, zero_rates)   # module-level convenience

    Parameters
    ----------
    maturities : np.ndarray
        Vertex maturities in years, shape (n,). Must be strictly increasing.
    zero_rates : np.ndarray
        Continuously compounded zero rates in **decimal** (0.025 = 2.5%),
        shape (n,). Same length as maturities.
    interpolation : str
        "log_linear" (default) — linear on log P(0,T), preserves positive
        forward rates. Equivalent to PiecewiseLogLinearDiscount.
        "linear"               — linear on zero rates.
        "cubic"                — monotone cubic spline on zero rates.
    reference_date : ql.Date or None
        Valuation date. Defaults to ql.Settings.instance().evaluationDate.
    valuation_date : str
        ISO string for the DiscountCurve.valuation_date property.
        Derived from reference_date if not supplied.
    """

    _day_count = ql.Actual365Fixed()
    _calendar  = ql.NullCalendar()

    def __init__(
        self,
        maturities: np.ndarray,
        zero_rates: np.ndarray,
        interpolation: str = "log_linear",
        reference_date: ql.Date | None = None,
        valuation_date: str = "",
    ) -> None:
        maturities = np.asarray(maturities, dtype=float)
        zero_rates  = np.asarray(zero_rates,  dtype=float)

        self._validate(maturities, zero_rates, interpolation)

        if reference_date is None:
            reference_date = ql.Settings.instance().evaluationDate

        self._maturities    = maturities
        self._zero_rates    = zero_rates
        self._interpolation = interpolation
        self._ref_date      = reference_date
        self._valuation_date = (
            valuation_date or
            f"{reference_date.year()}-{reference_date.month():02d}-{reference_date.dayOfMonth():02d}"
        )

        self._ql_curve = self._build(maturities, zero_rates, interpolation, reference_date)

    # ── DiscountCurve interface ───────────────────────────────────────────

    @property
    def currency(self) -> str:
        return ""

    @property
    def valuation_date(self) -> str:
        return self._valuation_date

    def discount(self, T: float) -> float:
        """
        Discount factor P(0,T). T in years.

        Delegates to the internal QuantLib curve via a date offset from
        the reference date.
        """
        dt = self._ref_date + ql.Period(max(int(round(T * 365)), 1), ql.Days)
        return self._ql_curve.discount(dt)

    def zero_rate(self, T: float) -> float:
        """
        Continuously compounded zero rate at T in years.

        Returns
        -------
        float
            Zero rate, decimal (e.g. 0.025 for 2.5%).
        """
        T = max(T, 1e-6)
        dt = self._ref_date + ql.Period(max(int(round(T * 365)), 1), ql.Days)
        return self._ql_curve.zeroRate(dt, self._day_count, ql.Continuous).rate()

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Continuously compounded forward rate for [T1, T2].

        Returns
        -------
        float
            Forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        T1 = max(T1, 1e-6)
        T2 = max(T2, T1 + 1e-6)
        dt1 = self._ref_date + ql.Period(max(int(round(T1 * 365)), 1), ql.Days)
        dt2 = self._ref_date + ql.Period(max(int(round(T2 * 365)), 1), ql.Days)
        return self._ql_curve.forwardRate(dt1, dt2, self._day_count, ql.Continuous).rate()

    def instantaneous_forward(self, T: float) -> float:
        """
        Instantaneous forward rate at T via central finite differences.

        Returns
        -------
        float
            Instantaneous forward rate, decimal (e.g. 0.025 for 2.5%).
        """
        dT = 1e-4
        p1 = self.discount(max(T - dT, 1e-6))
        p2 = self.discount(T + dT)
        return -(np.log(p2) - np.log(p1)) / (2 * dT)

    # ── Raw QuantLib access ───────────────────────────────────────────────

    @property
    def ql_curve(self) -> ql.YieldTermStructure:
        """
        The underlying QuantLib YieldTermStructure.

        Wrap in ql.YieldTermStructureHandle for use in QuantLib pricers:
            handle = ql.YieldTermStructureHandle(array_curve.ql_curve)
        """
        return self._ql_curve

    # ── Private ──────────────────────────────────────────────────────────

    @staticmethod
    def _validate(
        maturities: np.ndarray,
        zero_rates: np.ndarray,
        interpolation: str,
    ) -> None:
        if maturities.ndim != 1 or zero_rates.ndim != 1:
            raise ValueError("maturities and zero_rates must be 1-dimensional arrays")
        if len(maturities) == 0:
            raise ValueError("maturities must contain at least one element")
        if len(maturities) != len(zero_rates):
            raise ValueError(
                f"maturities and zero_rates must have the same length: "
                f"got {len(maturities)} and {len(zero_rates)}"
            )
        if len(maturities) > 1 and not np.all(np.diff(maturities) > 0):
            bad = int(np.where(np.diff(maturities) <= 0)[0][0])
            raise ValueError(
                f"maturities must be strictly increasing: "
                f"found non-increasing step at index {bad} "
                f"({maturities[bad]:.4f}Y → {maturities[bad + 1]:.4f}Y)"
            )
        if interpolation not in _VALID_INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {sorted(_VALID_INTERPOLATIONS)}: "
                f"got {interpolation!r}"
            )

    def _build(
        self,
        maturities: np.ndarray,
        zero_rates: np.ndarray,
        interpolation: str,
        ref: ql.Date,
    ) -> ql.YieldTermStructure:
        """Build the internal QuantLib curve."""
        vertex_dates = [
            ref + ql.Period(max(int(round(T * 365)), 1), ql.Days)
            for T in maturities
        ]

        if interpolation == "log_linear":
            # DiscountCurve anchors at (ref_date, 1.0) then each vertex
            dates = [ref] + vertex_dates
            dfs   = [1.0] + [
                float(np.exp(-r * T))
                for r, T in zip(zero_rates, maturities)
            ]
            curve = ql.DiscountCurve(dates, dfs, self._day_count, self._calendar)
        else:
            # Include ref_date with first rate for flat back-extension to T=0
            dates = [ref] + vertex_dates
            rates = [float(zero_rates[0])] + list(zero_rates.astype(float))
            if interpolation == "linear":
                curve = ql.ZeroCurve(
                    dates, rates, self._day_count, self._calendar,
                    ql.Linear(), ql.Continuous, ql.Annual,
                )
            else:  # cubic — monotone cubic spline on zero rates
                curve = ql.MonotonicCubicZeroCurve(
                    dates, rates, self._day_count, self._calendar,
                    ql.MonotonicCubic(), ql.Continuous, ql.Annual,
                )

        curve.enableExtrapolation()
        return curve


# ── Module-level convenience ──────────────────────────────────────────────────

def curve_from_arrays(
    maturities: np.ndarray,
    zero_rates: np.ndarray,
    interpolation: str = "log_linear",
    reference_date: ql.Date | None = None,
) -> ArrayCurve:
    """
    Build an ArrayCurve from maturity/rate arrays.

    Thin wrapper around ArrayCurve(). Returns a full DiscountCurve subclass
    with .discount(), .zero_rate(), .forward_rate() and a .ql_curve property
    for QuantLib pricer integration.

    Parameters
    ----------
    maturities    : years to each vertex, strictly increasing
    zero_rates    : continuously compounded zero rates in decimal (0.025 = 2.5%)
    interpolation : "log_linear" | "linear" | "cubic"
    reference_date: ql.Date — defaults to ql.Settings evaluationDate

    Returns
    -------
    ArrayCurve
    """
    return ArrayCurve(maturities, zero_rates, interpolation, reference_date)
