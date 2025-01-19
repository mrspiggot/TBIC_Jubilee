import streamlit as st
import pandas as pd
from app.controllers.portfolio_controller import PortfolioController


class PortfolioView:
    """Handles the rendering of portfolio data and controls in the Streamlit UI."""

    def render_sidebar(self, controller: PortfolioController) -> None:
        """Render the sidebar with upload controls and portfolio settings."""
        st.sidebar.title("Portfolio Controls")

        # File upload section with format explanation
        st.sidebar.markdown("### Upload Portfolio")
        st.sidebar.markdown("""
        Upload an Excel file with your portfolio data. in the following format
        - Required columns: `ticker`, **must** be from Yahoo Finance; & `quantity`
        - Currency, Industry, Sector and type data will be automatically enriched

        """)

        uploaded_file = st.sidebar.file_uploader(
            "Choose Excel file",
            type=['xlsx']
        )

        base_currency = st.sidebar.selectbox(
            "Base Currency",
            options=controller.get_available_currencies(),
            index=1  # GBP default
        )

        cash = st.sidebar.number_input(
            f"Cash (in {base_currency})",
            min_value=0.0,
            value=0.0,
            step=1000.0
        )

        if st.sidebar.button("Process Portfolio"):
            if uploaded_file is None:
                st.sidebar.error("Please upload a portfolio file first")
            else:
                with st.spinner("Processing portfolio..."):
                    controller.handle_file_upload(uploaded_file, base_currency, cash)

    def render_portfolio_tab(self, controller: PortfolioController) -> None:
        """Render the main portfolio view with holdings and analysis."""
        portfolio = controller.get_portfolio()

        if portfolio is None:
            st.info("Please upload a portfolio file and click Process Portfolio")
            return

        st.title("Portfolio")

        # Portfolio summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"Total Value ({portfolio.base_currency})",
                value=f"{portfolio.get_total_value():,.2f}"
            )

        with col2:
            st.metric(
                label=f"Cash ({portfolio.base_currency})",
                value=f"{portfolio.cash:,.2f}"
            )

        with col3:
            df = portfolio.to_dataframe()
            positions_count = len(df)
            st.metric(
                label="Number of Positions",
                value=positions_count
            )

        # Portfolio breakdown
        st.subheader("Holdings")

        # Helper function for numeric formatting
        def format_value(x):
            if pd.isna(x):
                return "N/A"
            elif isinstance(x, (int, float)):
                return f"{x:,.2f}"
            return str(x)

        # Create a display DataFrame
        display_df = df.copy()
        numeric_columns = [
            'price', 'price_in_base', 'value_in_currency', 'value_in_base'
        ]

        for col in numeric_columns:
            display_df[col] = display_df[col].apply(format_value)

        # Configure column settings
        column_config = {
            'ticker': st.column_config.Column(
                'Symbol',
                help="Stock ticker symbol",
                width="small"
            ),
            'name': st.column_config.Column(
                'Name',
                help="Company name"
            ),
            'quantity': st.column_config.NumberColumn(
                'Quantity',
                help="Number of shares"
            ),
            'currency': st.column_config.Column(
                'Currency',
                help="Local currency",
                width="small"
            ),
            'type': st.column_config.Column(
                'Type',
                help="Investment type (Growth/Value/Defensive)",
                width="medium"
            ),
            'sector': st.column_config.Column(
                'Sector',
                help="Industry sector from YFinance"
            ),
            'sub_industry': st.column_config.Column(
                'Sub-Industry',
                help="Specific industry classification"
            ),
            'price': st.column_config.Column(
                'Price (Local)',
                help="Current price in local currency"
            ),
            'price_in_base': st.column_config.Column(
                f'Price ({portfolio.base_currency})',
                help="Current price in base currency"
            ),
            'value_in_currency': st.column_config.Column(
                'Value (Local)',
                help="Position value in local currency"
            ),
            'value_in_base': st.column_config.Column(
                f'Value ({portfolio.base_currency})',
                help="Position value in base currency"
            )
        }

        # Display the holdings table with tooltips
        st.dataframe(
            data=display_df,
            column_config=column_config,
            hide_index=True
        )

        # Sector Allocation
        if not df.empty:
            st.subheader("Sector Allocation")

            # Distribute fund sectors
            distributed_df = controller.distribute_fund_sectors(df)

            # Calculate sector values and percentages
            sector_values = distributed_df.groupby('sector')['value_in_base'].sum().sort_values(ascending=False)
            total_value = sector_values.sum()

            # Create allocation table
            alloc_data = []
            for sector, value in sector_values.items():
                percentage = (value / total_value) * 100
                alloc_data.append({
                    'Sector': sector,
                    'Value': f"{value:,.2f} {portfolio.base_currency}",
                    'Allocation': f"{percentage:.1f}%"
                })

            # Display sector allocation table
            st.table(pd.DataFrame(alloc_data))