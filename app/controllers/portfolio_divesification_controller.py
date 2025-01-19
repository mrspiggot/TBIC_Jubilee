import os
import pandas as pd
import yfinance as yf
import logging
from typing import Dict, Optional

from app.models.portfolio import Portfolio

class PortfolioDiversificationController:
    """
    Controller for handling visualization data processing,
    specifically for sunburst and treemap charts.
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
        Load index components data from Excel file.
        """
        try:
            df = pd.read_excel(file_path)
            df['value'] = df['marketCap']
            return df[['stock', 'sector', 'industry', 'value']]
        except Exception as e:
            self.logger.error(f"Error loading index data from {file_path}: {e}")
            return pd.DataFrame()

    def get_sp500_data(self) -> pd.DataFrame:
        """Get S&P 500 index data."""
        return self.load_index_data(self.sp500_path)

    def get_russell_data(self) -> pd.DataFrame:
        """Get Russell 1000 index data."""
        return self.load_index_data(self.russell_path)

    def get_portfolio_data(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Process portfolio data for visualization.
        Handles fund re-allocation if possible.
        """
        df = portfolio.to_dataframe()

        # Process each row, handling funds separately
        processed_rows = []
        for _, row in df.iterrows():
            if row['sector'] == 'FUND':
                try:
                    ticker = yf.Ticker(row['ticker'])
                    sector_weightings = ticker.funds_data.sector_weightings

                    if sector_weightings:
                        # Distribute fund value across sectors
                        for sector, weight in sector_weightings.items():
                            processed_rows.append({
                                'stock': row['name'],
                                'sector': sector,
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
                    'industry': row['sub_industry'],
                    'value': row['value_in_base']
                })

        return pd.DataFrame(processed_rows)

    def get_sector_colors(self) -> Dict[str, str]:
        """Return sector color mapping."""
        return self.SECTOR_COLORS