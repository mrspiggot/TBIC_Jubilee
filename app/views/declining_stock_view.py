import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Optional
import logging
from app.controllers.declining_stocks_controller import DecliningStocksController
from app.models.portfolio import Portfolio

logger = logging.getLogger(__name__)


class DecliningStocksView:
    """Renders the declining stocks analysis view."""

    def __init__(self):
        self.controller = DecliningStocksController()
        self.months_options = {
            "1 Month": 1,
            "2 Months": 2,
            "3 Months": 3,
            "4 Months": 4,
            "5 Months": 5,
            "6 Months": 6
        }

    def render(self, portfolio: Portfolio) -> None:
        """Render the declining stocks analysis view."""
        st.title("Declining Stocks Analysis")

        if not portfolio:
            st.info("Please upload a portfolio to begin analysis.")
            return

        # Analysis controls
        col1, col2 = st.columns([2, 1])

        with col1:
            months_selection = st.selectbox(
                "Analysis Period",
                options=list(self.months_options.keys()),
                index=2  # Default to 3 months
            )
            months = self.months_options[months_selection]

        with col2:
            if st.button("Refresh Analysis"):
                self.controller.clear_cache()

        # Run analysis
        with st.spinner("Analyzing portfolio..."):
            results = self.controller.get_analysis_results(portfolio, months)

        if not results:
            st.error("Analysis failed. Please try again.")
            return

        # Display results
        self._render_price_table(results.get("price_table"))
        self._render_declining_stocks(results.get("declining_stocks", {}))

    def _render_price_table(self, price_table: Optional[pd.DataFrame]) -> None:
        """Render the price movement table."""
        if price_table is None or price_table.empty:
            return

        st.subheader("Price Movements")

        # Style the dataframe
        numeric_cols = price_table.columns[2:]
        price_table[numeric_cols] = price_table[numeric_cols].apply(pd.to_numeric, errors='coerce').round(2)
        def color_price_changes(row):
            styles = [''] * 2  # No color for Ticker and Name
            for i in range(2, len(row)):
                if i > 2:  # Skip first price column
                    curr_price = pd.to_numeric(row[i], errors='coerce')
                    prev_price = pd.to_numeric(row[i - 1], errors='coerce')
                    if pd.notna(curr_price) and pd.notna(prev_price):
                        color = 'red' if curr_price < prev_price else 'green'
                        styles.append(f'color: {color}')
                    else:
                        styles.append('')
                else:
                    styles.append('')
            return styles

        styled_table = price_table.style.apply(color_price_changes, axis=1)
        st.dataframe(styled_table, hide_index=True)

    def _render_declining_stocks(self, declining_stocks: Dict) -> None:
        """Render the declining stocks section."""
        if not declining_stocks:
            st.info("No stocks showing consistent price decline in the selected period.")
            return

        st.subheader("Stocks with Consistent Price Decline")

        for symbol, data in declining_stocks.items():
            with st.expander(f"{data['name']} ({symbol})"):
                if 'historical_data' in data and not data['historical_data'].empty:
                    fig = self._create_stock_chart(data['historical_data'], symbol)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No historical data available for this stock.")

    def _create_stock_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create a combined candlestick and line chart."""
        fig = go.Figure()

        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ))

        # Add moving averages
        ma20 = df['Close'].rolling(window=20).mean()
        ma50 = df['Close'].rolling(window=50).mean()

        fig.add_trace(go.Scatter(
            x=df.index, y=ma20,
            line=dict(color='orange', width=1),
            name='20-day MA'
        ))

        fig.add_trace(go.Scatter(
            x=df.index, y=ma50,
            line=dict(color='blue', width=1),
            name='50-day MA'
        ))

        fig.update_layout(
            title=f'{symbol} Price History',
            yaxis_title='Price',
            xaxis_title='Date',
            height=500,
            template='plotly_white'
        )

        return fig