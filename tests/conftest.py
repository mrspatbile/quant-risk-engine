import pytest
import pandas as pd
from quant_risk.curves.ois import OISCurve


def _synthetic_ois_data() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "years":           [1/12, 2/12, 3/12, 6/12, 9/12,  1.0,  2.0,  3.0,  5.0, 10.0, 15.0],
            "zero_rate_pct":   [2.63, 2.61, 2.58, 2.48, 2.38, 2.30, 2.20, 2.15, 2.20, 2.40, 2.50],
            "discount_factor": [0.998, 0.996, 0.994, 0.988, 0.982, 0.977, 0.957, 0.937, 0.895, 0.786, 0.690],
            "valuation_date":  ["2026-03-24"] * 11,
        },
        index=["1M", "2M", "3M", "6M", "9M", "12M", "2Y", "3Y", "5Y", "10Y", "10Y+"],
    )
    data.index.name = "maturity"
    return data


@pytest.fixture
def curve():
    return OISCurve(_synthetic_ois_data())


@pytest.fixture
def ois_curve():
    return OISCurve(_synthetic_ois_data())
