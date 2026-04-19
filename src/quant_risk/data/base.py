from abc import ABC, abstractmethod
import pandas as pd


class CentralBankClient(ABC):
    """
    Abstract base class for central bank data clients.

    All concrete implementations must provide overnight rates,
    spot rates at standard maturities, and declare their
    day count convention. This allows the curve construction
    layer to work with any jurisdiction without modification.

    Implementations
    ---------------
    ECBClient  -- Euro area, ESTR, ACT/360
    FedClient  -- United States, SOFR, ACT/360
    BCBClient  -- Brazil, CDI/Selic, BUS/252
    """

    @property
    @abstractmethod
    def day_count_convention(self) -> str:
        """
        Day count convention for this market.

        Returns
        -------
        str
            One of 'ACT/360', 'ACT/365', 'BUS/252'
        """
        pass

    @property
    @abstractmethod
    def currency(self) -> str:
        """
        ISO 4217 currency code for this market.

        Returns
        -------
        str
            Examples: 'EUR', 'USD', 'BRL'
        """
        pass

    @abstractmethod
    def get_overnight_rate(self, last_n: int = 252) -> pd.Series:
        """
        Overnight risk-free rate fixing series.

        The reference rate for OIS discounting in this jurisdiction:
        ESTR for EUR, SOFR for USD, CDI for BRL.

        Parameters
        ----------
        last_n : int
            Number of most recent observations to fetch.

        Returns
        -------
        pd.Series
            Indexed by date string (YYYY-MM-DD), values in percent.
        """
        pass

    @abstractmethod
    def get_spot_rate(self, maturity: str, last_n: int = 252) -> pd.Series:
        """
        Government benchmark spot rate at a given maturity.

        Parameters
        ----------
        maturity : str
            Maturity label. Convention varies by implementation:
            ECB uses 'SR_10Y', Fed uses '10Y', BCB uses '10Y'.
        last_n : int
            Number of most recent observations to fetch.

        Returns
        -------
        pd.Series
            Indexed by date string (YYYY-MM-DD), values in percent.
        """
        pass

    @abstractmethod
    def get_full_curve(self, last_n: int = 5) -> pd.DataFrame:
        """
        Full term structure across standard maturities.

        Parameters
        ----------
        last_n : int
            Number of most recent dates to fetch per maturity.

        Returns
        -------
        pd.DataFrame
            Indexed by date, columns are maturity labels,
            values in percent.
        """
        pass

    def describe(self) -> str:
        """Human readable summary of this client's jurisdiction."""
        return (
            f"{self.__class__.__name__} | "
            f"currency={self.currency} | "
            f"day_count={self.day_count_convention}"
        )