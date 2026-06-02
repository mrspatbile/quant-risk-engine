"""
Type-safe input objects for discount curve construction.

These dataclasses enforce validation at the data→curve boundary and replace
untyped DataFrames, dicts, and Series with explicit, immutable structures.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class OISParRates:
    """
    Raw OIS par rates returned by ECBClient.

    This is an intermediate transport type — the output of
    get_ois_par_rates_by_date() and input to OISCurve.from_ecb().

    Parameters
    ----------
    rates : dict[str, float]
        OIS par rates indexed by MMSR tenor label (e.g. '1M', '2M', ..., '10Y+').
        Values are in decimal (e.g. 0.025 for 2.5%).
    valuation_date : str
        ISO date string YYYY-MM-DD. The date these rates resolved to
        (may be earlier than the requested date if exact date unavailable).
    """
    rates: dict[str, float]
    valuation_date: str

    def __post_init__(self):
        # Validate date format
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.valuation_date):
            raise ValueError(
                f"valuation_date must be YYYY-MM-DD, got {self.valuation_date}"
            )

        # Validate rates are in plausible range
        for tenor, rate in self.rates.items():
            if not isinstance(rate, (int, float)):
                raise TypeError(
                    f"Rate for {tenor} must be numeric, got {type(rate).__name__}"
                )
            if rate < -0.02 or rate > 0.20:
                raise ValueError(
                    f"Rate for {tenor} is {rate}, outside plausible range [-0.02, 0.20]"
                )


@dataclass(frozen=True)
class OISCurveInput:
    """
    Complete input to OISCurve constructor.

    Combines overnight rate and par rates for the same valuation date.
    All validation happens in __post_init__ — callers cannot construct
    an invalid OISCurveInput and pass it to OISCurve().

    Parameters
    ----------
    on_rate : float
        ESTR overnight rate in decimal (e.g. 0.025 for 2.5%).
    par_rates : dict[str, float]
        OIS par rates indexed by MMSR tenor label ('1M', '2M', ..., '10Y+').
        Values in decimal. Must include all standard MMSR tenors.
    valuation_date : str
        ISO date string YYYY-MM-DD. The single date all rates are from.
    """
    on_rate: float
    par_rates: dict[str, float]
    valuation_date: str

    def __post_init__(self):
        # Validate date format
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", self.valuation_date):
            raise ValueError(
                f"valuation_date must be YYYY-MM-DD, got {self.valuation_date}"
            )

        # Validate ON rate
        if not isinstance(self.on_rate, (int, float)):
            raise TypeError(
                f"on_rate must be numeric, got {type(self.on_rate).__name__}"
            )
        if self.on_rate < -0.02 or self.on_rate > 0.20:
            raise ValueError(
                f"on_rate is {self.on_rate}, outside plausible range [-0.02, 0.20]"
            )

        # Validate par rates dict
        expected_tenors = {
            "1M", "2M", "3M", "6M", "9M", "12M", "2Y", "3Y", "5Y", "10Y", "10Y+"
        }
        provided_tenors = set(self.par_rates.keys())

        if not provided_tenors.issubset(expected_tenors):
            extra = provided_tenors - expected_tenors
            raise ValueError(
                f"par_rates has unexpected tenors: {extra}"
            )

        if provided_tenors != expected_tenors:
            missing = expected_tenors - provided_tenors
            raise ValueError(
                f"par_rates missing required tenors: {missing}"
            )

        # Validate each par rate is numeric and in range
        for tenor, rate in self.par_rates.items():
            if not isinstance(rate, (int, float)):
                raise TypeError(
                    f"Par rate for {tenor} must be numeric, got {type(rate).__name__}"
                )
            if rate < -0.02 or rate > 0.20:
                raise ValueError(
                    f"Par rate for {tenor} is {rate}, outside plausible range [-0.02, 0.20]"
                )


@dataclass(frozen=True)
class NSSParameters:
    """
    Nelson-Siegel-Svensson curve parameters.

    These are the NSS parameters returned by ECBClient.get_nss_parameters()
    and passed to NSSCurve constructor. The parameters are the raw ECB values
    (betas in percent, not decimal).

    Parameters
    ----------
    beta0, beta1, beta2, beta3 : float
        NSS level, slope, and curvature parameters (ECB convention: in percent).
    tau1, tau2 : float
        NSS loading maturity parameters (in years).
    valuation_date : str
        ISO date string YYYY-MM-DD. Optional; empty string if not available.
    """
    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float
    valuation_date: str = ""

    def __post_init__(self):
        # Validate date format if provided
        if self.valuation_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", self.valuation_date):
            raise ValueError(
                f"valuation_date must be YYYY-MM-DD, got {self.valuation_date}"
            )

        # Validate tau parameters (must be positive)
        if self.tau1 <= 0:
            raise ValueError(f"tau1 must be positive, got {self.tau1}")
        if self.tau2 <= 0:
            raise ValueError(f"tau2 must be positive, got {self.tau2}")

        # Betas and taus can be any finite float — no range validation
        # (NSS parameters vary widely by curve and economic regime)
