import logging
import pandas as pd
from typing import List, Tuple, Optional, Dict
from datetime import date, timedelta
import streamlit as st

from utils.stock_provider import YFinanceProvider
from app.models.analysis.peer_analyzer import PeerAnalyzer
from app.models.enrichment.sector_metrics_lookup import SectorMetricsLookup
from app.models.portfolio import Portfolio

logger = logging.getLogger(__name__)

class PeerAnalysisController:
    """
    Coordinates the 'peer analysis' logic:
      - Reads S&P 500 data (app/assets/sp_500.xlsx) for peer filtering.
      - Always uses YFinance to get the chosen stock's 'industry' code, ignoring whether
        the portfolio stock is in S&P 500.
      - Looks up relevant metrics via SectorMetricsLookup (sector_valuation_metrics_methodology.xlsx).
      - Uses PeerAnalyzer for per-stock metrics from YFinance.
      - Returns DataFrames for the PeerAnalysisView to display or chart.
    """

    PERIODS = {
        "1 Month": 1,
        "3 Months": 3,
        "6 Months": 6,
        "1 Year": 12,
        "2 Years": 24,
        "5 Years": 60
    }

    def __init__(self):
        self.peer_analyzer = PeerAnalyzer()
        self.sector_metrics = SectorMetricsLookup()
        self.stock_provider = YFinanceProvider()

        # Cache references
        self._cached_sp500: Optional[pd.DataFrame] = None
        self._cached_metrics: Dict[str, List[str]] = {}

    def load_sp500_dataset(self, path: str = "app/assets/sp_500_stock_ind.xlsx") -> pd.DataFrame:
        """
        Loads (and caches) S&P 500 data from the given path.
        We expect columns: 'stock', 'sector', 'industry', etc.
        """
        if self._cached_sp500 is not None:
            return self._cached_sp500

        try:
            df = pd.read_excel(path)
            # Check minimal columns
            if "stock" not in df.columns or "industry" not in df.columns:
                logger.warning(f"[PeerAnalysisController] sp_500 missing 'stock' or 'industry' columns.")
                df = pd.DataFrame()

            self._cached_sp500 = df
            print(df.info())
            print(df.head(20))
            ind = df['industry'].unique().tolist()
            print(ind)

            logger.info(f"[PeerAnalysisController] Loaded S&P 500 from {path}, shape={df.shape}")
            return df

        except Exception as e:
            logger.error(f"[PeerAnalysisController] Error loading {path}: {e}")
            return pd.DataFrame()

    def get_portfolio_stock_options(self, portfolio: Portfolio) -> List[str]:
        """
        Return user-friendly strings, e.g. "Apple Inc (AAPL)", sorted by ticker.
        """
        if not portfolio or not portfolio.positions:
            return []
        options = []
        for symbol, stock_obj in portfolio.positions.items():
            display_name = stock_obj.name or symbol
            options.append(f"{display_name} ({symbol})")

        return sorted(options)

    def parse_stock_symbol(self, display_str: str) -> str:
        """
        Extract the ticker from "Apple Inc (AAPL)" -> "AAPL".
        """
        if "(" in display_str and display_str.endswith(")"):
            return display_str[display_str.rindex("(") + 1 : -1].strip()
        return display_str.strip()

    @staticmethod
    def normalize_string(s: str) -> str:
        """Normalize a string for comparison by removing extra whitespace and converting to lowercase."""
        return ' '.join(s.strip().lower().split())

    def build_peer_table(
            self,
            chosen_display_str: str,
            portfolio: Portfolio,
            sp500_df: pd.DataFrame,
            top_n: int = 5
    ) -> pd.DataFrame:

        SECTOR_MAPPING = {
            'Consumer Defensive': 'Consumer Staples',
            'Consumer Cyclical': 'Consumer Discretionary',
            'Financial Services': 'Financials',
            'Technology': 'Information Technology',
            'Healthcare': 'Health Care'
        }

        symbol = self.parse_stock_symbol(chosen_display_str)
        logger.info(f"[PeerAnalysisController] Processing symbol: {symbol}")

        # Get YFinance data
        yfin_info = self.stock_provider.get_info(symbol)
        target_sector = yfin_info.get("sector", "")
        target_industry = yfin_info.get("industry", "")


        # logger.info(f"[PeerAnalysisController] YFinance data for {symbol}:")
        # logger.info(f"  Sector: '{target_sector}'")
        # logger.info(f"  Industry: '{target_industry}'")

        # Map YFinance sector to S&P sector
        mapped_sector = SECTOR_MAPPING.get(target_sector, target_sector)
        # logger.info(f"[PeerAnalysisController] Mapped sector '{target_sector}' to '{mapped_sector}'")

        # Get sector matches
        peers_df = sp500_df[sp500_df['industry'].str.strip() == target_industry.strip()]
        # logger.info(f"[PeerAnalysisController] Found {len(peers_df)} companies in sector '{target_industry}'")

        if not peers_df.empty:
            # logger.info("[PeerAnalysisController] Peer companies found:")
            # for _, row in peers_df.iterrows():
            #     logger.info(f"  {row['stock']}: {row['industry']} ({row['sector']})")

            metrics_for_industry = self.sector_metrics.get_metrics_for_industry(target_industry, top_n=top_n)
            if not metrics_for_industry:
                logger.warning(f"[PeerAnalysisController] No metrics defined for industry='{target_industry}'")
                return pd.DataFrame()

            peer_symbols = peers_df["stock"].unique().tolist()
            # logger.info(f"[PeerAnalysisController] Will fetch metrics for {len(peer_symbols)} peers")

            result_df = self.peer_analyzer.get_peer_metrics(
                chosen_symbol=symbol,
                peer_symbols=peer_symbols,
                metrics=metrics_for_industry,
                portfolio=portfolio
            )

            return result_df
        else:
            logger.info(f"No peers found in sector {mapped_sector}")
            return pd.DataFrame()

    def get_historical_returns_and_prices(
        self,
        chosen_symbol: str,
        peer_symbols: List[str],
        period_label: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        For chosen_symbol + peer_symbols, fetch historical data (X months).
        Return:
         - returns_df: [stock, return, is_selected]
         - norm_prices_df: normalized performance
        """
        months = self.PERIODS.get(period_label, 3)
        end_date = date.today()
        start_date = end_date - timedelta(days=int(30.436875 * months))

        all_syms = [chosen_symbol] + [s for s in peer_symbols if s != chosen_symbol]
        if not all_syms:
            logger.warning("[PeerAnalysisController] No symbols to fetch historical data for.")
            return pd.DataFrame(), pd.DataFrame()

        df_prices = self.stock_provider.get_historical_data_multiple(
            symbols=all_syms,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )
        if df_prices.empty:
            logger.warning("[PeerAnalysisController] No historical data returned for peers.")
            return pd.DataFrame(), pd.DataFrame()

        first = df_prices.iloc[0]
        last = df_prices.iloc[-1]
        returns = ((last - first) / first) * 100.0

        returns_df = pd.DataFrame({
            "stock": returns.index,
            "return": returns.values,
            "is_selected": [True if x == chosen_symbol else False for x in returns.index]
        }).sort_values("return", ascending=False)

        norm_prices_df = (df_prices.div(df_prices.iloc[0]) - 1.0) * 100.0
        return returns_df, norm_prices_df
