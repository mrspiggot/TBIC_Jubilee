import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Optional, List, Dict
import logging

from app.models.portfolio import Portfolio
from app.models.stock import Stock
from utils.fx_manager import (
    FxManager, FxCache, FxRepository, ExchangeRateHostProvider
)
from utils.stock_provider import YFinanceProvider


class PortfolioController:
    """Controls portfolio operations and manages state between models and views."""

    def __init__(self):
        """Initialize controller with FX and Stock providers."""
        # Initialize providers if not in session state
        if 'fx_manager' not in st.session_state:
            st.session_state.fx_manager = self._initialize_fx_manager()

        if 'stock_provider' not in st.session_state:
            st.session_state.stock_provider = YFinanceProvider()

        # Initialize portfolio state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = None

        self.fx_manager: FxManager = st.session_state.fx_manager
        self.stock_provider: YFinanceProvider = st.session_state.stock_provider
        self.logger = logging.getLogger(__name__)

    def _initialize_fx_manager(self) -> FxManager:
        """Create and initialize FX manager with default settings."""
        provider = ExchangeRateHostProvider()
        cache = FxCache("utils/data/fx/fx_rates.pkl")
        repo = FxRepository(
            "utils/data/fx/fx_rates.xlsx",
            "utils/data/fx/fx_rates.db"
        )

        fx_manager = FxManager(
            provider=provider,
            cache=cache,
            repository=repo,
            stale_threshold_hours=48
        )

        # Initialize with default base currency
        fx_manager.ensure_fresh_data(base="GBP")
        return fx_manager

    def handle_file_upload(
            self,
            uploaded_file,
            base_currency: str,
            cash: float = 0.0
    ) -> None:
        """Process uploaded portfolio file and update state."""
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Validate required columns
            required_cols = {'ticker', 'quantity'}
            if not required_cols.issubset(df.columns):
                st.error(f"Upload must have columns: {required_cols}")
                return

            # Clean data
            df['ticker'] = df['ticker'].str.strip().str.upper()
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

            # Remove any rows with NaN quantities
            df = df.dropna(subset=['quantity'])

            # Create portfolio from DataFrame
            portfolio = Portfolio.from_dataframe(
                df,
                base_currency=base_currency,
                cash=cash,
                fx_manager=self.fx_manager,
                stock_provider=self.stock_provider
            )

            # Enrich with market data
            with st.spinner("Fetching market data..."):
                portfolio.enrich_positions()

            # Update session state
            st.session_state.portfolio = portfolio
            st.success("Portfolio processed successfully!")

        except Exception as e:
            st.error(f"Error processing portfolio: {str(e)}")

    def get_portfolio(self) -> Optional[Portfolio]:
        """Get current portfolio from session state."""
        return st.session_state.get('portfolio')

    def get_available_currencies(self) -> List[str]:
        """Get list of available currencies for base currency selection."""
        return [
            "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "HKD", "GBp"
        ]

    def clear_portfolio(self) -> None:
        """Clear current portfolio from session state."""
        st.session_state.portfolio = None

    def update_base_currency(self, new_currency: str) -> None:
        """Update portfolio base currency."""
        portfolio = self.get_portfolio()
        if portfolio:
            portfolio.base_currency = new_currency
            # Refresh FX data with new base
            self.fx_manager.ensure_fresh_data(base=new_currency)

    def update_cash(self, cash: float) -> None:
        """Update portfolio cash position."""
        portfolio = self.get_portfolio()
        if portfolio:
            portfolio.cash = cash

    def distribute_fund_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute fund sector weights across other sectors.

        Args:
            df (pd.DataFrame): DataFrame of portfolio positions

        Returns:
            pd.DataFrame: DataFrame with fund sectors distributed
        """
        # Create a copy to avoid modifying the original
        distributed_df = df.copy()

        # Identify fund rows
        fund_rows = distributed_df[distributed_df['sector'] == 'FUND']

        # Remove fund rows from the main dataframe
        distributed_df = distributed_df[distributed_df['sector'] != 'FUND']

        # Process each fund
        for _, fund_row in fund_rows.iterrows():
            try:
                # Fetch fund sector weightings
                ticker = yf.Ticker(fund_row['ticker'])
                fund_sector_weights = ticker.funds_data.sector_weightings

                # Distribute fund value across sectors
                for yf_sector, weight in fund_sector_weights.items():
                    # Map YFinance sector to our sector classification
                    mapped_sector = self._map_sector(yf_sector)

                    # Calculate distributed value
                    distributed_value = fund_row['value_in_base'] * weight

                    # Check if sector exists, if not add a new row
                    matching_rows = distributed_df[distributed_df['sector'] == mapped_sector]

                    if not matching_rows.empty:
                        # Add to existing sector's value
                        distributed_df.loc[
                            distributed_df['sector'] == mapped_sector, 'value_in_base'] += distributed_value
                    else:
                        # Create a new row for this sector
                        new_row = fund_row.copy()
                        new_row['sector'] = mapped_sector
                        new_row['value_in_base'] = distributed_value
                        distributed_df = distributed_df.append(new_row, ignore_index=True)

            except Exception as e:
                self.logger.error(f"Error processing fund {fund_row['ticker']}: {str(e)}")

        return distributed_df

    def _map_sector(self, yf_sector: str) -> str:
        """Map YFinance sector names to our sector classification."""
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
        return sector_map.get(yf_sector, 'Unknown')