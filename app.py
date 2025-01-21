import streamlit as st
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.controllers.portfolio_controller import PortfolioController
from app.views.portfolio_view import PortfolioView
from app.views.sunburst_view import SunburstView
from app.views.treemap_view import TreemapView
from app.views.performance_view import PerformanceView
from app.views.benchmark_view import BenchmarkView
from app.views.peer_analysis_view import PeerAnalysisView

# Configure page
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="📈",
    layout="wide"
)

# Initialize controller
controller = PortfolioController()

# Initialize view
view = PortfolioView()

# Render sidebar
view.render_sidebar(controller)

# Create tabs
tab_portfolio, tab_composition, tab_concentration, tab_performance, tab_benchmark, tab_peer = st.tabs([
    "Portfolio", "Composition", "Concentration", "Performance", "Benchmark", "Peer Analysis"
])


with tab_portfolio:
    # Render main content
    view.render_portfolio_tab(controller)

with tab_composition:
    # Initialize Sunburst view
    sunburst_view = SunburstView()

    # Get current portfolio
    portfolio = controller.get_portfolio()

    # Render Sunburst tab
    sunburst_view.render_sunburst_tab(portfolio)

with tab_concentration:
    # Initialize Treemap view
    treemap_view = TreemapView()

    # Get current portfolio
    portfolio = controller.get_portfolio()

    # Render Treemap tab
    treemap_view.render_treemap_tab(portfolio)

with tab_performance:
    # Initialize Performance view
    performance_view = PerformanceView()

    # Get current portfolio
    portfolio = controller.get_portfolio()

    # Render Performance tab
    performance_view.render(portfolio)

with tab_benchmark:
    benchmark_view = BenchmarkView()
    benchmark_view.render(portfolio)


with tab_peer:
    peer_analysis_view = PeerAnalysisView()
    peer_analysis_view.render(portfolio)