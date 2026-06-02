from quant_risk.data.ecb import ECBClient
from quant_risk.data.fed import FedClient
from quant_risk.data.external_data_client import ExternalDataClient

__all__ = [
    "ECBClient",
    "FedClient",
    "ExternalDataClient",
]


def __getattr__(name):
    """Provide helpful error message for deprecated ExternalStore."""
    if name == "ExternalStore":
        raise ImportError(
            "\nExternalStore was renamed to ExternalDataClient in v0.3.0.\n"
            "\nMigration:\n"
            "  from quant_risk import ExternalDataClient\n"
            "  client = ExternalDataClient()\n"
            "  prices = client.get_yfinance('SPY')\n"
            "  factors = client.get_fama_french()\n"
            "\nThis helpful message will be removed in v0.4.0.\n"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
