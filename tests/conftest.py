import numpy as np
import pytest
import pandas as pd
from quant_risk.curves.ois import OISCurve
from quant_risk.curves.models import OISCurveInput

_TENORS = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
_LABELS = ["1M", "2M", "3M", "6M", "9M", "12M", "2Y", "3Y", "5Y", "10Y", "10Y+"]


def _make_ois_input(rate: float = 0.025, valuation_date: str = "2026-03-24") -> OISCurveInput:
    """Create OISCurveInput with flat rate across all tenors."""
    return OISCurveInput(
        on_rate=rate,
        par_rates={
            "1M": rate,
            "2M": rate,
            "3M": rate,
            "6M": rate,
            "9M": rate,
            "12M": rate,
            "2Y": rate,
            "3Y": rate,
            "5Y": rate,
            "10Y": rate,
            "10Y+": rate,
        },
        valuation_date=valuation_date,
    )


def _flat_ois(rate: float = 0.025) -> OISCurve:
    """Flat OIS curve at rate (decimal). Shared by test_models and test_xva."""
    return OISCurve(_make_ois_input(rate))


def _synthetic_ois_input() -> OISCurveInput:
    """OIS input matching original synthetic curve (flat rate)."""
    # Use flat rate structure that QL bootstrap can converge on.
    # Original tests used flat 2.5%, which converges well.
    rate = 0.025
    return OISCurveInput(
        on_rate=rate,
        par_rates={
            "1M": rate,
            "2M": rate,
            "3M": rate,
            "6M": rate,
            "9M": rate,
            "12M": rate,
            "2Y": rate,
            "3Y": rate,
            "5Y": rate,
            "10Y": rate,
            "10Y+": rate,
        },
        valuation_date="2026-03-24",
    )


@pytest.fixture
def curve():
    return OISCurve(_synthetic_ois_input())


@pytest.fixture
def ois_curve():
    return OISCurve(_synthetic_ois_input())
