import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.controllers.benchmark_controller import BenchmarkController
from app.models.portfolio import Portfolio


class BenchmarkView:
    def __init__(self):
        self.controller = BenchmarkController()
        self.periods = {
            "1 Month": 1,
            "3 Months": 3,
            "6 Months": 6,
            "1 Year": 12,
            "3 Years": 36,
            "5 Years": 60
        }

    def render(self, portfolio: Portfolio):
        st.title("Benchmark Comparison")

        period = st.selectbox("Select Period", list(self.periods.keys()))
        months = self.periods[period]

        if st.button("Run Benchmark Analysis"):
            with st.spinner("Fetching benchmark data..."):
                equity_data = self.controller.get_equity_data(months)
                sector_data = self.controller.get_sector_data(months)
                crypto_data = self.controller.get_crypto_data(months)
                other_asset_data = self.controller.get_other_asset_data(months)
                portfolio_performance = self.controller.get_portfolio_performance(portfolio, months)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Equity Benchmarks")
                self.plot_benchmark(equity_data, portfolio_performance, "Equity Benchmarks")

                st.subheader("Sector ETFs")
                self.plot_benchmark(sector_data, portfolio_performance, "Sector ETFs")

            with col2:
                st.subheader("Cryptocurrencies")
                self.plot_benchmark(crypto_data, portfolio_performance, "Cryptocurrencies")

                st.subheader("Other Assets")
                self.plot_benchmark(other_asset_data, portfolio_performance, "Other Assets")

    def plot_benchmark(self, benchmark_data: pd.DataFrame, portfolio_performance: pd.Series, title: str):
        if benchmark_data.empty:
            st.warning(f"No data available for {title}")
            return

        fig = go.Figure()

        # Add benchmark data lines
        for column in benchmark_data.columns:
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data[column],
                mode='lines',
                name=column
            ))

        # Add portfolio performance line
        if not portfolio_performance.empty:
            fig.add_trace(go.Scatter(
                x=portfolio_performance.index,
                y=portfolio_performance,
                mode='lines',
                name="Portfolio",
                line=dict(color='red', width=2, dash='dash')
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Percentage Change",

            height=400,
            width=600
        )

        st.plotly_chart(fig, use_container_width=True)

