from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import yfinance as yf
from datetime import datetime
import pandas as pd
import logging
import time
from .backoff import BackoffStrategy, ExponentialBackoff, with_backoff

logger = logging.getLogger(__name__)


class StockDataProvider(ABC):
    """Abstract base class for stock data providers."""

    @abstractmethod
    def get_info(self, symbol: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_market_cap(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_stock_price(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_business_summary(self, symbol: str) -> str:
        pass


class YFinanceProvider(StockDataProvider):
    """Yahoo Finance implementation with rate limiting & batch functions."""

    def __init__(
            self,
            backoff_strategy: Optional[BackoffStrategy] = None,
            batch_size: int = 10,
            rate_limit_pause: float = 2.0
    ):
        self.backoff = backoff_strategy or ExponentialBackoff()
        self.batch_size = batch_size
        self.rate_limit_pause = rate_limit_pause
        self._cache: Dict[str, Dict[str, Any]] = {}

    @with_backoff()
    def _fetch_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch the ticker info from yfinance."""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Log raw sector/industry data for debugging
        # logger.debug(f"YFinance raw data for {symbol}:")
        # logger.debug(f"  sector: {info.get('sector')}")
        # logger.debug(f"  industry: {info.get('industry')}")

        return info


    def _get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Get ticker info with caching and rate limiting."""
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_info(symbol)
            time.sleep(self.rate_limit_pause)  # Rate-limiting pause
        return self._cache[symbol]

    def get_sector_data(self, symbol: str) -> Dict[str, str]:
        """Get sector and industry data for a symbol."""
        info = self.get_info(symbol)
        sector = info.get('sector', '')
        industry = info.get('industry', '')

        # logger.info(f"[StockProvider] Raw YFinance data for {symbol}:")
        # logger.info(f"  Full info keys: {list(info.keys())}")
        # logger.info(f"  Sector: '{sector}'")
        # logger.info(f"  Industry: '{industry}'")

        return {
            'sector': sector,
            'industry': industry
        }

    def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get all available information for a stock."""
        try:
            return self._get_ticker_info(symbol)
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market capitalization for a stock."""
        try:
            info = self._get_ticker_info(symbol)
            return info.get('marketCap')
        except Exception as e:
            logger.error(f"Error fetching market cap for {symbol}: {str(e)}")
            return None

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Return a single 'price' for the symbol.
        Uses 'bid' or 'currentPrice' or 'previousClose' if necessary.
        """
        try:
            info = self._get_ticker_info(symbol)
            price = info.get('bid')
            if price is None:
                price = info.get('currentPrice')
                if price is None:
                    # logger.warning(
                    #     f"No 'currentPrice' found for {symbol}, "
                    #     f"trying fallback keys..."
                    # )
                    price = info.get('regularMarketPreviousClose') or info.get('previousClose')
            if price is None:
                logger.warning(
                    f"No price data found for {symbol}. Info keys: {list(info.keys())}"
                )
            return price
        except Exception as e:
            logger.error(f"Error fetching stock price for {symbol}: {str(e)}")
            return None

    def get_business_summary(self, symbol: str) -> str:
        """Get the business summary for a stock."""
        try:
            info = self._get_ticker_info(symbol)
            return info.get('longBusinessSummary', '')
        except Exception as e:
            logger.error(f"Error fetching summary for {symbol}: {str(e)}")
            return ""

    def get_historical_data(
            self,
            symbol: str,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.
        Returns DataFrame with OHLCV data or empty DataFrame if error.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start,
                end=end,
                interval=interval
            )
            if data.empty:
                logger.warning(f"No historical data for {symbol} "
                               f"({start} to {end}, interval={interval})")
            return data
        except Exception as e:
            logger.error(
                f"Error fetching historical data for {symbol}: {str(e)}"
            )
            return pd.DataFrame()

    # Add or update this method in the YFinanceProvider class
    def get_historical_data_multiple(
            self,
            symbols: List[str],
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for multiple tickers.
        Returns DataFrame with columns = ticker symbols.
        """
        if not symbols:
            return pd.DataFrame()

        df_all = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            progress=False,
            group_by="column",
            auto_adjust=auto_adjust
        )

        if df_all.empty:
            logger.warning("No data returned for requested symbols.")
            return pd.DataFrame()

        # Handle multi-index case
        if isinstance(df_all.columns, pd.MultiIndex):
            close_col = "Adj Close" if auto_adjust else "Close"
            if close_col in df_all.columns.levels[0]:
                df_close = df_all.xs(close_col, level=0, axis=1)
            else:
                logger.warning(f"'{close_col}' not found. Using 'Close'")
                df_close = df_all.xs("Close", level=0, axis=1)
        else:
            # Single-level columns
            df_close = df_all["Adj Close" if auto_adjust else "Close"]

        if isinstance(df_close, pd.Series):
            df_close = df_close.to_frame()

        # Forward fill missing values
        df_close.ffill(inplace=True)
        return df_close

    def get_historical_data_multiple_old(
            self,
            symbols: List[str],
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            auto_adjust: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for multiple tickers.
        Returns DataFrame with columns = ticker symbols.
        """
        # logger.info(
        #     f"Fetching multiple tickers: {symbols}, "
        #     f"from {start} to {end}, auto_adjust={auto_adjust}"
        # )

        if not symbols:
            return pd.DataFrame()

        df_all = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            progress=False,
            group_by="column",
            auto_adjust=auto_adjust
        )

        if df_all.empty:
            logger.warning("No data returned for requested symbols.")
            return pd.DataFrame()

        # Handle multi-index case
        if isinstance(df_all.columns, pd.MultiIndex):
            close_col = "Adj Close" if auto_adjust else "Close"
            if close_col in df_all.columns.levels[0]:
                df_close = df_all.xs(close_col, level=0, axis=1)
            else:
                logger.warning(f"'{close_col}' not found. Using 'Close'")
                df_close = df_all.xs("Close", level=0, axis=1)
        else:
            # Single-level columns
            df_close = df_all["Adj Close" if auto_adjust else "Close"]

        if isinstance(df_close, pd.Series):
            df_close = df_close.to_frame()

        df_close.ffill(inplace=True)
        return df_close


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    provider = YFinanceProvider()

    # Test sector data extraction
    test_symbols = ["AAPL", "MSFT", "JPM", "XOM", "GOOGL"]

    print("\nTesting sector data extraction:")
    for symbol in test_symbols:
        sector_data = provider.get_sector_data(symbol)
        print(f"\n{symbol}:")
        print(f"  Sector: {sector_data['sector']}")
        print(f"  Industry: {sector_data['industry']}")