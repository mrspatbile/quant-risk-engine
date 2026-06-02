from quant_risk.instruments import (
    Bond,
    IRSwap,
    FXForward,
    VanillaOption,
    CreditDefaultSwap,
    STANDARD_RECOVERY,
    STANDARD_COUPON_IG,
    STANDARD_COUPON_HY,
    EquityForward,
    DigitalOption,
    BarrierOption,
    ChooserOption,
    CompoundOption,
    AsianOption,
    LookbackOption,
    CliquetOption,
    ShoutOption,
    NapoleonOption,
    Accumulator,
    Decumulator,
    WorstOfOption,
    BestOfOption,
    TotalReturnSwap,
)
from quant_risk.curves import OISCurve, NSSCurve, ArrayCurve
from quant_risk.models import (
    VasicekProcess,
    HullWhiteProcess,
    CIRProcess,
    GBMProcess,
    LocalVolProcess,
    MCSimulator,
)
from quant_risk.data import ECBClient, FedClient, ExternalDataClient
from quant_risk.utils import realized_volatility, realized_correlation
from quant_risk.risk import Trade, XVAEngine

__all__ = [
    # instruments
    "Bond",
    "IRSwap",
    "FXForward",
    "VanillaOption",
    "CreditDefaultSwap",
    "STANDARD_RECOVERY",
    "STANDARD_COUPON_IG",
    "STANDARD_COUPON_HY",
    "EquityForward",
    "DigitalOption",
    "BarrierOption",
    "ChooserOption",
    "CompoundOption",
    "AsianOption",
    "LookbackOption",
    "CliquetOption",
    "ShoutOption",
    "NapoleonOption",
    "Accumulator",
    "Decumulator",
    "WorstOfOption",
    "BestOfOption",
    "TotalReturnSwap",
    # curves
    "OISCurve",
    "NSSCurve",
    "ArrayCurve",
    # models
    "VasicekProcess",
    "HullWhiteProcess",
    "CIRProcess",
    "GBMProcess",
    "LocalVolProcess",
    "MCSimulator",
    # calibration utilities
    "realized_volatility",
    "realized_correlation",
    # data
    "ECBClient",
    "FedClient",
    "ExternalDataClient",
    # risk
    "Trade",
    "XVAEngine",
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
