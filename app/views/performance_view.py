import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any

from app.controllers.performance_controller import PerformanceController
from app.models.portfolio import Portfolio


class PerformanceView:
    def __init__(self):
        self.periods = {
            "1 Month": 1,
            "2 Months": 2,
            "3 Months": 3,
            "6 Months": 6,
            "1 Year": 12,
            "2 Years": 24,
            "5 Years": 60
        }

    def render(self, portfolio: Portfolio) -> None:
        st.title("Performance")

        if not portfolio:
            st.info("Please upload and process a portfolio first on the Portfolio tab.")
            return

        selected_period = st.selectbox("Select Period", options=list(self.periods.keys()))
        months = self.periods[selected_period]

        controller = PerformanceController(portfolio)
        performance_data = controller.calculate_performance(months)

        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            self._render_portfolio_summary(performance_data["portfolio_summary"])
        with col2:
            self._render_performance_by_attribute(performance_data["performance_by_sector"], "Sector")

        # Row 2
        col1, col2 = st.columns(2)
        with col1:
            self._render_performers(performance_data["best_performers"], "Best Performers")
        with col2:
            self._render_performers(performance_data["worst_performers"], "Worst Performers")

        # Row 3
        col1, col2 = st.columns(2)
        with col1:
            self._render_performance_by_attribute(performance_data["performance_by_style"], "Style")
        with col2:
            self._render_performance_by_attribute(performance_data["performance_by_currency"], "Currency")

    def _render_portfolio_summary(self, summary_data: Dict[str, Any]) -> None:
        st.subheader("Portfolio Summary")
        subcol1, subcol2 = st.columns(2)

        with subcol1:
            st.metric("Initial Value", f"{summary_data['initial_value']:,.2f} {summary_data['base_currency']}")
            st.metric("Current Value", f"{summary_data['current_value']:,.2f} {summary_data['base_currency']}")
            st.metric("Number of Stocks", summary_data['num_stocks'])

        with subcol2:
            st.metric("Gain/Loss", f"{summary_data['gain_loss']:,.2f} {summary_data['base_currency']}")
            st.metric("Percentage Change", f"{summary_data['percentage_change']:.2f}%")
            st.metric("Cash Holdings", f"{summary_data['cash_holdings']:,.2f} {summary_data['base_currency']}")

    def _render_performers(self, performers: List[Dict[str, Any]], title: str) -> None:
        st.subheader(title)
        df = pd.DataFrame(performers)
        fig = px.bar(df, x='symbol', y='performance', color='sector', text='performance')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

    def _render_performance_by_attribute(self, attribute_data: List[Dict[str, Any]], attribute_name: str) -> None:
        st.subheader(f"Performance by {attribute_name}")
        df = pd.DataFrame(attribute_data)
        fig = px.bar(df, x='attribute', y='performance', color='attribute', text='performance')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)