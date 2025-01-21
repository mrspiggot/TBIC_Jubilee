# File: app/views/peer_analysis_view.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from app.controllers.peer_analysis_controller import PeerAnalysisController
from app.models.portfolio import Portfolio
import pandas as pd
from utils.formatters import format_number


# def format_number(value):
#     """Format numbers with K, M, Bn, Tn suffixes and 3 decimal places."""
#     if pd.isna(value) or value is None:
#         return "N/A"
#
#     try:
#         value = float(value)
#         if abs(value) >= 1e12:
#             return f"{value / 1e12:.3f}Tn"
#         if abs(value) >= 1e9:
#             return f"{value / 1e9:.3f}Bn"
#         if abs(value) >= 1e6:
#             return f"{value / 1e6:.3f}M"
#         if abs(value) >= 1e3:
#             return f"{value / 1e3:.3f}K"
#         return f"{value:.3f}"
#     except (ValueError, TypeError):
#         return str(value)

class PeerAnalysisView:
    """
    Renders UI elements for Peer Analysis:
      - Select a portfolio stock
      - Fetch & display peer metrics
      - Plot returns over chosen period
    """

    def __init__(self, sp500_data_path: str = "app/assets/sp_500_stock_ind.xlsx"):
        """
        sp500_data_path: path to the S&P 500 spreadsheet with columns:
                         ['stock', 'sector', 'industry', 'MarketCap', 'MarketCapBillions']
        """
        self.controller = PeerAnalysisController()
        self.sp500_data_path = sp500_data_path



    def render(self, portfolio: Portfolio) -> None:
        """
        Main entry point for rendering the Peer Analysis tab in Streamlit.
        """
        st.title("Peer Analysis")

        # Check we have a portfolio
        if not portfolio or not portfolio.positions:
            st.info("Please load a portfolio first.")
            return

        # 1) Load or get S&P 500 data
        if "sp500_peer_df" not in st.session_state:
            df_sp500 = self.controller.load_sp500_dataset(self.sp500_data_path)
            st.session_state["sp500_peer_df"] = df_sp500
        else:
            df_sp500 = st.session_state["sp500_peer_df"]

        # If S&P data is empty, show error
        if df_sp500.empty:
            st.error("Failed to load S&P 500 data or file is missing required columns.")
            return

        # 2) Stock dropdown
        stock_options = self.controller.get_portfolio_stock_options(portfolio)
        if not stock_options:
            st.warning("No valid stocks in the portfolio.")
            return

        selected_stock_str = st.selectbox("Select a Stock from Portfolio", stock_options)

        # 3) Slider for top N metrics
        top_n_metrics = st.slider("Number of metrics to display", min_value=1, max_value=15, value=5)

        # 4) Fetch Peers & Metrics
        if st.button("Fetch Peers & Metrics"):
            peer_table = self.controller.build_peer_table(
                chosen_display_str=selected_stock_str,
                portfolio=portfolio,
                sp500_df=df_sp500,
                top_n=top_n_metrics
            )
            st.session_state["peer_analysis_table"] = peer_table
            st.session_state["selected_peer_stock"] = selected_stock_str

        # 5) Display the peer table if we have it
        if "peer_analysis_table" in st.session_state and \
           st.session_state["selected_peer_stock"] == selected_stock_str:
            df_peers = st.session_state["peer_analysis_table"]

            if df_peers.empty:
                st.info("No peers found or no metrics returned.")
            else:
                st.subheader("Peer Comparison Table")
                chosen_symbol = self.controller.parse_stock_symbol(selected_stock_str)

                def style_dataframe(df):
                    # Create a copy to avoid modifying the original
                    styled_df = df.copy()

                    # Format each numeric column
                    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_columns:
                        styled_df[col] = styled_df[col].apply(format_number)

                    # Define the style function
                    def highlight_chosen(row):
                        return [
                            "background-color: yellow; color: black"
                            if row.get("symbol") == chosen_symbol else ""
                            for _ in row
                        ]

                    # Apply the styling
                    return styled_df.style.apply(highlight_chosen, axis=1)

                st.dataframe(
                    style_dataframe(df_peers),
                    use_container_width=True,
                    hide_index=True
                )

        # 6) Historical Performance Chart
        st.subheader("Peer Returns Over Time")
        period_choice = st.selectbox("Select Period", list(self.controller.PERIODS.keys()), index=1)

        if st.button("Get Peer Returns"):
            # Make sure we have a peer table
            if "peer_analysis_table" not in st.session_state:
                st.warning("Please fetch peers/metrics first.")
                return

            df_peers = st.session_state["peer_analysis_table"]
            if df_peers.empty or "symbol" not in df_peers.columns:
                st.warning("No valid peer symbols found to chart.")
                return

            chosen_symbol = self.controller.parse_stock_symbol(selected_stock_str)
            all_peer_symbols = df_peers["symbol"].unique().tolist()

            returns_df, norm_prices_df = self.controller.get_historical_returns_and_prices(
                chosen_symbol=chosen_symbol,
                peer_symbols=all_peer_symbols,
                period_label=period_choice
            )

            if returns_df.empty or norm_prices_df.empty:
                st.warning("No historical data returned.")
                return

            # Sort returns_df by return value in descending order
            returns_df = returns_df.sort_values("return", ascending=False)

            # Bar chart for total returns
            fig_bar = px.bar(
                returns_df,
                x="stock",
                y="return",
                color="is_selected",
                color_discrete_map={True: "yellow", False: "blue"},
                title=f"Total Return over {period_choice}",
                category_orders={"stock": returns_df["stock"].tolist()}  # Preserve the sorted order
            )
            fig_bar.update_layout(
                showlegend=False,
                xaxis_title="Stock",
                yaxis_title="Return (%)",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Line chart for normalized performance
            st.subheader("Relative Performance (Line Chart)")
            fig_line = go.Figure()

            # Chosen stock in thicker line
            fig_line.add_trace(
                go.Scatter(
                    x=norm_prices_df.index,
                    y=norm_prices_df[chosen_symbol],
                    name=chosen_symbol,
                    line=dict(color="yellow", width=3)
                )
            )

            # Plot peers
            palette = px.colors.qualitative.Set3
            idx = 0
            for col in norm_prices_df.columns:
                if col == chosen_symbol:
                    continue
                fig_line.add_trace(
                    go.Scatter(
                        x=norm_prices_df.index,
                        y=norm_prices_df[col],
                        name=col,
                        line=dict(color=palette[idx % len(palette)], width=1)
                    )
                )
                idx += 1

            fig_line.update_layout(
                title=f"Relative Performance over {period_choice}",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_line, use_container_width=True)