import streamlit as st
import plotly.express as px
import pandas as pd

from app.controllers.portfolio_divesification_controller import PortfolioDiversificationController
from app.models.portfolio import Portfolio

class PortfolioDiversificationView:
    def __init__(self):
        self.controller = PortfolioDiversificationController()

    def build_sunburst_figure(self, df: pd.DataFrame, title: str):
        """Build a Plotly sunburst figure."""
        if df.empty:
            fig = px.sunburst(
                pd.DataFrame({"sector": [], "industry": [], "stock": [], "value": []}),
                path=["sector", "industry", "stock"],
                values="value",
            )
            fig.update_layout(title=f"{title} (No Data)")
            return fig

        # Create deep copy to prevent modifications
        df = df.copy(deep=True)

        fig = px.sunburst(
            df,
            path=["sector", "industry", "stock"],
            values="value",
            color="sector",
            color_discrete_map=self.controller.get_sector_colors(),
            title=title
        )

        # Customize chart layout
        fig.update_layout(
            height=800,
            width=800,
            margin=dict(t=50, l=0, r=0, b=0),
            title_x=0.5,
            title_font_size=20
        )

        return fig

    def render_visualization_tab(self, portfolio: Portfolio):
        """Render visualization tab with Sunburst and Treemap charts."""
        st.header("Portfolio Visualization")

        # Get visualization data
        sp500_data = self.controller.get_sp500_data()
        russell_data = self.controller.get_russell_data()
        portfolio_data = self.controller.get_portfolio_data(portfolio)

        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Sunburst Charts", "Treemap Charts"])

        with tab1:
            st.subheader("Sector Composition Sunburst")
            col1, col2, col3 = st.columns(3)

            with col1:
                fig_sp = self.build_sunburst_figure(
                    df=sp500_data,
                    title="S&P 500 Composition"
                )
                st.plotly_chart(fig_sp, use_container_width=False)

            with col2:
                fig_russell = self.build_sunburst_figure(
                    df=russell_data,
                    title="Russell 1000 Composition"
                )
                st.plotly_chart(fig_russell, use_container_width=False)

            with col3:
                fig_portfolio = self.build_sunburst_figure(
                    df=portfolio_data,
                    title=f"Portfolio Composition"
                )
                st.plotly_chart(fig_portfolio, use_container_width=False)

        with tab2:
            st.subheader("Sector Composition Treemap")
            st.write("Treemap implementation coming soon...")