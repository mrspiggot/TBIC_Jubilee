import os
import pandas as pd
import yfinance as yf
import logging
from typing import Dict, Optional

from app.models.portfolio import Portfolio


class SunburstController:
    """
    Controller for managing data processing and visualization
    for Sunburst charts across different market indices and portfolios.
    """

    SECTOR_COLORS = {
        'Technology': 'blue',
        'Communication Services': 'orange',
        'Healthcare': 'green',
        'Consumer Cyclical': 'gold',
        'Financial Services': 'lightblue',
        'Industrials': 'olive',
        'Consumer Defensive': 'crimson',
        'Energy': 'black',
        'Utilities': 'maroon',
        'Real Estate': 'pink',
        'Basic Materials': 'purple',
        'FUND': 'white',
        'Unknown': 'gray'
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.russell_path = "utils/data/index/data/russell_1000_tickers_components.xlsx"
        self.sp500_path = "utils/data/index/data/sp_500_tickers_components.xlsx"

    def load_index_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process index components data from Excel file.

        Args:
            file_path (str): Path to the index components Excel file

        Returns:
            pd.DataFrame: Processed index data with standardized columns
        """
        try:
            df = pd.read_excel(file_path)
            df['value'] = df['marketCap']
            df['industry'] = df['industry'].fillna('Unknown')
            return df[['stock', 'sector', 'industry', 'value']]
        except Exception as e:
            self.logger.error(f"Error loading index data from {file_path}: {e}")
            return pd.DataFrame(columns=['stock', 'sector', 'industry', 'value'])

    def get_sp500_data(self) -> pd.DataFrame:
        """Get S&P 500 index data."""
        return self.load_index_data(self.sp500_path)

    def get_russell_data(self) -> pd.DataFrame:
        """Get Russell 1000 index data."""
        return self.load_index_data(self.russell_path)

    def get_portfolio_data(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Process portfolio data for Sunburst visualization.
        Handles fund sector re-allocation and data preparation.

        Args:
            portfolio (Portfolio): User's investment portfolio

        Returns:
            pd.DataFrame: Processed portfolio data for visualization
        """
        if portfolio is None:
            return pd.DataFrame(columns=['stock', 'sector', 'industry', 'value'])

        df = portfolio.to_dataframe()
        processed_rows = []

        for _, row in df.iterrows():
            if row['sector'] == 'FUND':
                try:
                    ticker = yf.Ticker(row['ticker'])
                    sector_weightings = ticker.funds_data.sector_weightings

                    if sector_weightings and isinstance(sector_weightings, dict):
                        # Distribute fund value across sectors
                        for sector, weight in sector_weightings.items():
                            mapped_sector = self._map_yfinance_sector(sector)
                            processed_rows.append({
                                'stock': row['name'],
                                'sector': mapped_sector,
                                'industry': 'FUND',
                                'value': row['value_in_base'] * weight
                            })
                    else:
                        # If no sector weightings, keep as FUND
                        processed_rows.append({
                            'stock': row['name'],
                            'sector': 'FUND',
                            'industry': 'FUND',
                            'value': row['value_in_base']
                        })
                except Exception as e:
                    self.logger.error(f"Error processing fund {row['ticker']}: {e}")
                    processed_rows.append({
                        'stock': row['name'],
                        'sector': 'FUND',
                        'industry': 'FUND',
                        'value': row['value_in_base']
                    })
            else:
                processed_rows.append({
                    'stock': row['name'],
                    'sector': row['sector'],
                    'industry': row['sub_industry'] or 'Unknown',
                    'value': row['value_in_base']
                })

        return pd.DataFrame(processed_rows)

    def _map_yfinance_sector(self, yfinance_sector: str) -> str:
        """
        Map YFinance sector names to our standard sector classification.

        Args:
            yfinance_sector (str): Sector name from YFinance

        Returns:
            str: Mapped sector name
        """
        sector_map = {
            'technology': 'Technology',
            'consumer_cyclical': 'Consumer Cyclical',
            'communication_services': 'Communication Services',
            'consumer_defensive': 'Consumer Defensive',
            'financial_services': 'Financial Services',
            'industrials': 'Industrials',
            'healthcare': 'Healthcare',
            'basic_materials': 'Basic Materials',
            'real_estate': 'Real Estate',
            'energy': 'Energy',
            'utilities': 'Utilities'
        }
        return sector_map.get(yfinance_sector.lower(), 'Unknown')

    def get_sector_colors(self) -> Dict[str, str]:
        """
        Retrieve sector color mapping.

        Returns:
            Dict[str, str]: Mapping of sectors to colors
        """
        return self.SECTOR_COLORS