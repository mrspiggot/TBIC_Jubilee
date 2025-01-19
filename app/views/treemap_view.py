import streamlit as st
import plotly.express as px
import pandas as pd

from app.controllers.treemap_controller import TreemapController
from app.models.portfolio import Portfolio


class TreemapView:
    def __init__(self):
        self.controller = TreemapController()

    def build_treemap_figure(self, df: pd.DataFrame, title: str):
        """Build a Plotly treemap figure."""
        if df.empty:
            fig = px.treemap(
                pd.DataFrame({"type": [], "sector": [], "industry": [], "stock": [], "value": []}),
                path=["type", "sector", "industry", "stock"],
                values="value",
            )
            fig.update_layout(title=f"{title} (No Data)")
            return fig

        # Create deep copy to prevent modifications
        df = df.copy(deep=True)

        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['type', 'sector', 'industry', 'stock', 'value'])

        # Replace any remaining NaNs with 'Unknown'
        df['type'] = df['type'].fillna('Unknown')
        df['sector'] = df['sector'].fillna('Unknown')
        df['industry'] = df['industry'].fillna('Unknown')
        df['stock'] = df['stock'].fillna('Unknown')

        # Ensure value is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)

        # Remove rows with zero or negative values
        df = df[df['value'] > 0]

        if df.empty:
            fig = px.treemap(
                pd.DataFrame({"type": [], "sector": [], "industry": [], "stock": [], "value": []}),
                path=["type", "sector", "industry", "stock"],
                values="value",
            )
            fig.update_layout(title=f"{title} (No Valid Data)")
            return fig

        fig = px.treemap(
            df,
            path=["type", "sector", "industry", "stock"],
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

    def render_treemap_tab(self, portfolio: Portfolio):
        """Render Treemap charts for different market indices and portfolio."""
        st.header("Portfolio Concentration")

        if portfolio is None:
            st.info("Please upload a portfolio file to view concentration")
            return

        # Get visualization data
        sp500_data = self.controller.get_sp500_data()
        russell_data = self.controller.get_russell_data()
        portfolio_data = self.controller.get_portfolio_data(portfolio)

        # Create columns for charts
        col1, col2, col3 = st.columns(3)

        with col1:
            fig_sp = self.build_treemap_figure(
                df=sp500_data,
                title="S&P 500 Concentration"
            )
            st.plotly_chart(fig_sp, use_container_width=False)

        with col2:
            fig_russell = self.build_treemap_figure(
                df=russell_data,
                title="Russell 1000 Concentration"
            )
            st.plotly_chart(fig_russell, use_container_width=False)

        with col3:
            fig_portfolio = self.build_treemap_figure(
                df=portfolio_data,
                title=f"Portfolio Concentration"
            )
            st.plotly_chart(fig_portfolio, use_container_width=False)