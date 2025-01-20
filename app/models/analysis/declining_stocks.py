import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import logging
from app.models.portfolio import Portfolio
from utils.stock_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class DecliningStocksAnalyzer:
    """Analyzes portfolio for stocks with consistently declining prices."""

    def __init__(self):
        """Initialize the analyzer with a stock data provider."""
        self.stock_provider = YFinanceProvider()

    def analyze_portfolio(self, portfolio: Portfolio, months: int) -> Dict[str, Any]:
        """
        Analyze portfolio for stocks showing consistent price declines.

        Args:
            portfolio: Portfolio instance to analyze
            months: Number of months to analyze for decline

        Returns:
            Dict containing:
                - declining_stocks: Dict of stocks with consistent decline
                - price_table: DataFrame of monthly prices
        """
        if not portfolio or not portfolio.positions:
            logger.warning("Empty portfolio provided for analysis")
            return {"declining_stocks": {}, "price_table": None}

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * (months + 1))  # Extra month for comparison

            # Get historical data for all stocks
            symbols = list(portfolio.positions.keys())

            # Debug print
            print(f"Analyzing {len(symbols)} symbols over {months} months")
            print(f"Date range: {start_date} to {end_date}")

            price_data = self._get_monthly_prices(symbols, start_date, end_date)

            # Debug print
            print(f"Retrieved price data shape: {price_data.shape if price_data is not None else 'None'}")

            if price_data is None or price_data.empty:
                logger.warning("No price data retrieved for analysis")
                return {"declining_stocks": {}, "price_table": None}

            # Create price table
            price_table = self._create_price_table(portfolio, price_data)

            # Identify declining stocks
            declining_stocks = self._identify_declining_stocks(portfolio, price_data, months)

            return {
                "declining_stocks": declining_stocks,
                "price_table": price_table
            }

        except Exception as e:
            logger.error(f"Error in portfolio analysis: {str(e)}")
            return {"declining_stocks": {}, "price_table": None}

    def _get_monthly_prices(
            self,
            symbols: List[str],
            start_date: datetime,
            end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get monthly price data for all symbols.
        """
        try:
            # Get daily data first
            historical_data = self.stock_provider.get_historical_data_multiple(
                symbols,
                start=start_date,
                end=end_date
            )

            if historical_data.empty:
                return None

            # Resample to monthly, taking the last price of each month
            monthly_data = historical_data.resample('M').last()

            # Debug print
            print(f"Monthly data shape: {monthly_data.shape}")
            print(f"Monthly data columns: {monthly_data.columns}")

            return monthly_data

        except Exception as e:
            logger.error(f"Error getting monthly prices: {str(e)}")
            return None

    def _create_price_table(
            self,
            portfolio: Portfolio,
            price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a formatted price table with stock names and monthly prices.
        """
        try:
            # Initialize table with tickers and names
            records = []

            for symbol in portfolio.positions:
                if symbol in price_data.columns:
                    stock = portfolio.positions[symbol]
                    record = {
                        'Ticker': symbol,
                        'Name': stock.name or symbol
                    }

                    # Add monthly prices
                    monthly_prices = price_data[symbol].tail(7)  # Last 7 months
                    for i, (date, price) in enumerate(monthly_prices.items()):
                        record[f'Month{i}'] = round(price, 2) if pd.notnull(price) else None

                    records.append(record)

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error creating price table: {str(e)}")
            return pd.DataFrame()

    def _identify_declining_stocks(
            self,
            portfolio: Portfolio,
            price_data: pd.DataFrame,
            months: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify stocks with consistently declining prices.
        """
        declining_stocks = {}

        try:
            for symbol in portfolio.positions:
                if symbol not in price_data.columns:
                    continue

                stock = portfolio.positions[symbol]
                prices = price_data[symbol].tail(months + 1)  # Include extra month for comparison

                # Check for consistent decline
                is_declining = True
                for i in range(1, len(prices)):
                    if prices.iloc[i] >= prices.iloc[i - 1]:
                        is_declining = False
                        break

                if is_declining:
                    # Get detailed historical data for chart
                    historical_data = self.stock_provider.get_historical_data(
                        symbol,
                        start=datetime.now() - timedelta(days=180),  # 6 months
                        end=datetime.now()
                    )

                    declining_stocks[symbol] = {
                        "name": stock.name or symbol,
                        "historical_data": historical_data
                    }

            return declining_stocks

        except Exception as e:
            logger.error(f"Error identifying declining stocks: {str(e)}")
            return {}