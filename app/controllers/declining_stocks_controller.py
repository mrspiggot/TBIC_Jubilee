import streamlit as st
import pandas as pd
from typing import Dict, Optional
import logging
from app.models.analysis.declining_stocks import DecliningStocksAnalyzer
from app.models.portfolio import Portfolio

logger = logging.getLogger(__name__)


class DecliningStocksController:
    """Controls the analysis and data preparation for declining stocks view."""

    def __init__(self):
        self.analyzer = DecliningStocksAnalyzer()

    def get_analysis_results(self, portfolio: Portfolio, months: int) -> Dict:
        """
        Get declining stocks analysis with caching.

        Args:
            portfolio: Portfolio to analyze
            months: Number of months to analyze

        Returns:
            Dict containing analysis results
        """
        if not portfolio:
            logger.warning("No portfolio provided for analysis")
            return {}

        try:
            # Create cache key
            cache_key = f"declining_stocks_analysis_{months}"

            # Check if analysis is in cache
            if cache_key not in st.session_state:
                logger.info(f"Running new analysis for {months} months")
                results = self.analyzer.analyze_portfolio(portfolio, months)
                st.session_state[cache_key] = results
            else:
                logger.info("Using cached analysis results")
                results = st.session_state[cache_key]

            return results

        except Exception as e:
            logger.error(f"Error in declining stocks analysis: {str(e)}")
            return {}

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        cache_keys = [k for k in st.session_state.keys()
                      if k.startswith("declining_stocks_analysis_")]
        for key in cache_keys:
            del st.session_state[key]
        logger.info("Cleared declining stocks analysis cache")