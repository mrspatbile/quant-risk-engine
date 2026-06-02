"""
Tests for quant_risk.data.ecb -- ECBClient.

All tests use mocked API responses -- no live ECB API calls.
Run with: pytest tests/test_ecb_client.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from quant_risk.data.ecb import ECBClient


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _mock_ecb_response():
    """
    Mock ECB API response for ESTR rates.
    Matches the structure returned by ECB SDW API for EST/B.EU000A2X2A25.WT
    """
    return {
        "dataSets": [
            {
                "series": {
                    "0": {
                        "observations": {
                            "0": [2.45],   # 2025-12-31
                            "1": [2.40],   # 2026-01-02
                            "2": [2.38],   # 2026-01-05
                            "3": [2.35],   # 2026-01-06
                        }
                    }
                }
            }
        ],
        "structure": {
            "dimensions": {
                "observation": [
                    {
                        "values": [
                            {"id": "2025-12-31"},
                            {"id": "2026-01-02"},
                            {"id": "2026-01-05"},
                            {"id": "2026-01-06"},
                        ]
                    }
                ]
            }
        }
    }


class TestECBClientGetOvernightRateSingleDate:
    """Test get_overnight_rate_single_date method."""

    def test_returns_float(self):
        """Should return a float rate in decimal format."""
        client = ECBClient()
        with patch.object(client, '_get', return_value=_mock_ecb_response()):
            rate = client.get_overnight_rate_single_date('2025-12-31')
            assert isinstance(rate, (float, np.floating))

    def test_converts_percent_to_decimal(self):
        """Should convert rate from percent (2.45) to decimal (0.0245)."""
        client = ECBClient()
        with patch.object(client, '_get', return_value=_mock_ecb_response()):
            rate = client.get_overnight_rate_single_date('2025-12-31')
            assert rate == pytest.approx(0.0245)

    def test_retrieves_correct_date(self):
        """Should return the rate for the requested date."""
        client = ECBClient()
        with patch.object(client, '_get', return_value=_mock_ecb_response()):
            rate_1 = client.get_overnight_rate_single_date('2025-12-31')
            rate_2 = client.get_overnight_rate_single_date('2026-01-02')

            assert rate_1 == pytest.approx(0.0245)
            assert rate_2 == pytest.approx(0.0240)
            assert rate_1 != rate_2

    def test_date_not_found_raises_keyerror(self):
        """Should raise KeyError if date is not in response."""
        client = ECBClient()
        with patch.object(client, '_get', return_value=_mock_ecb_response()):
            with pytest.raises(KeyError):
                client.get_overnight_rate_single_date('2025-01-01')

    def test_calls_get_with_correct_params(self):
        """Should call _get with EST dataset and ESTR series key."""
        client = ECBClient()
        mock_get = Mock(return_value=_mock_ecb_response())

        with patch.object(client, '_get', mock_get):
            client.get_overnight_rate_single_date('2025-12-31')

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs['dataset'] == 'EST'
        assert call_kwargs['series_key'] == 'B.EU000A2X2A25.WT'
        assert call_kwargs['params']['format'] == 'jsondata'
        assert call_kwargs['params']['lastNObservations'] == 365

    def test_handles_multiple_sequential_calls(self):
        """Should work correctly across multiple calls."""
        client = ECBClient()
        with patch.object(client, '_get', return_value=_mock_ecb_response()):
            rates = [
                client.get_overnight_rate_single_date('2025-12-31'),
                client.get_overnight_rate_single_date('2026-01-02'),
                client.get_overnight_rate_single_date('2026-01-05'),
            ]

            assert len(rates) == 3
            assert rates[0] == pytest.approx(0.0245)
            assert rates[1] == pytest.approx(0.0240)
            assert rates[2] == pytest.approx(0.0238)
