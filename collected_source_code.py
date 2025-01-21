# Files extracted on: 2025-01-21 07:03:48

# Excluded files and directories: ['.git', '__pycache__', 'venv', 'env', '.env', 'build', 'dist', 'collect_source_code.pysetup_project_structure.py', 'collect_source_files.py', 'assets', '.venv', 'ddqpro.egg-info', '.gitignore', 'requirements.txt', 'data/utils', 'docs', 'main.py', 'create_structure.py']

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app.py
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

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/collect_source_code.py
# collect_source_files.py
from datetime import datetime
import os

# List of files and directories to exclude
EXCLUDES = ['.git', '__pycache__', 'venv', 'env', '.env', 'build', 'dist', "collect_source_code.py"
            'setup_project_structure.py', 'collect_source_files.py', "assets", ".venv", "ddqpro.egg-info",
            ".gitignore", "requirements.txt", "data/utils", "docs", "main.py", "create_structure.py"]

def should_exclude(path, excludes):
    for exclude in excludes:
        if exclude in path:
            return True
    return False

def collect_python_files(base_dir, excludes):
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d), excludes)]
        for file in files:
            if file.endswith('.py') and not should_exclude(os.path.join(root, file), excludes):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def write_collected_files(python_files, output_file):
    with open(output_file, 'w') as outfile:
        # Write the datetime as the first line
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        outfile.write(f"# Files extracted on: {current_datetime}\n\n")
        # Write the list of excluded files/directories at the top
        outfile.write(f"# Excluded files and directories: {EXCLUDES}\n\n")
        for file_path in python_files:
            outfile.write(f"# File: {file_path}\n")
            with open(file_path, 'r') as infile:
                outfile.write(infile.read())
                outfile.write('\n\n')  # Add some spacing between files

if __name__ == "__main__":
    # Set the base directory to the current working directory
    base_directory = os.getcwd()
    output_filename = 'collected_source_code.py'

    python_files = collect_python_files(base_directory, EXCLUDES)
    write_collected_files(python_files, output_filename)
    print(f"Collected {len(python_files)} Python files into {output_filename}")


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/stock.py
from dataclasses import dataclass
from typing import Optional
from utils.fx_manager import FxManager
from utils.stock_provider import YFinanceProvider

# Add a mapping of sectors to types
sector_to_type = {
    'Consumer Cyclical': 'Cyclical',
    'Financial Services': 'Cyclical',
    'Industrials': 'Cyclical',
    'Basic Materials': 'Cyclical',
    'Real Estate': 'Cyclical',
    'Consumer Defensive': 'Defensive',
    'Energy': 'Defensive',
    'Utilities': 'Defensive',
    'Communication Services': 'Growth',
    'Healthcare': 'Growth',
    'Technology': 'Growth',
    'FUND': "Diversified Fund"
}


@dataclass
class Stock:
    """Represents a single stock position with both basic and enriched data."""

    # Basic position data
    ticker: str
    quantity: float

    # Enriched data from YFinance
    name: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    sub_industry: Optional[str] = None
    type: Optional[str] = None
    price: Optional[float] = None

    def __post_init__(self):
        """Validate and set defaults for optional fields."""
        if self.quantity < 0:
            raise ValueError("Quantity cannot be negative")

        # Set defaults for optional fields
        self.name = self.name or self.ticker
        self.currency = self.currency or "USD"  # Default to USD if not specified

    def enrich_from_yfinance(self, provider: YFinanceProvider) -> None:
        """Fetch and update stock data from YFinance."""
        try:
            # Get basic info
            info = provider.get_info(self.ticker)
            self.name = info.get('longName', self.ticker)
            self.currency = info.get('currency', 'USD')

            # Get sector data
            sector_data = provider.get_sector_data(self.ticker)
            if sector_data['sector']=='':
                sector_data['sector'] = "FUND"
                sector_data['industry'] = "Diversified Fund"

            self.sector = sector_data.get('sector', 'Unknown')
            self.sub_industry = sector_data.get('industry', 'Unknown')

            # Determine type based on sector
            self.type = self._determine_type(self.sector)

            # Get latest price
            self.price = provider.get_stock_price(self.ticker)

        except Exception as e:
            print(f"Error enriching data for {self.ticker}: {str(e)}")

    def _determine_type(self, sector: str) -> str:
        """Map sector to type (Cyclical/Growth/Defensive)."""
        sector_to_type = {
            'Consumer Cyclical': 'Cyclical',
            'Financial Services': 'Cyclical',
            'Industrials': 'Cyclical',
            'Basic Materials': 'Cyclical',
            'Real Estate': 'Cyclical',
            'Consumer Defensive': 'Defensive',
            'Energy': 'Defensive',
            'Utilities': 'Defensive',
            'Communication Services': 'Growth',
            'Healthcare': 'Growth',
            'Technology': 'Growth',
            'FUND': "Diversified Fund"
        }
        return sector_to_type.get(sector, 'Unknown')

    def get_value_in_currency(self) -> float:
        """Calculate value in local currency."""
        if self.price is None:
            return 0.0
        return self.quantity * self.price

    def get_value_in_base(self, fx_manager: FxManager, base_currency: str) -> float:
        """Calculate value in base currency using FX rates."""
        value_in_currency = self.get_value_in_currency()
        if value_in_currency == 0.0 or self.currency == base_currency:
            return value_in_currency

        fx_rate = fx_manager.get_rate(self.currency, base_currency)
        return value_in_currency * fx_rate

    def get_price_in_base(self, fx_manager: FxManager, base_currency: str) -> float:
        """Get stock price in base currency."""
        if self.price is None:
            return 0.0
        if self.currency == base_currency:
            return self.price

        fx_rate = fx_manager.get_rate(self.currency, base_currency)
        return self.price * fx_rate

    def _determine_type(self, sector: str) -> str:
        """Map sector to type (Cyclical/Growth/Defensive)."""
        return sector_to_type[sector]

    # Add this method to enrich the stock with type information
    def enrich_type(self) -> None:
        self.type = self._determine_type(self.sector)


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/portfolio.py
from typing import List, Dict, Optional
import pandas as pd
from utils.fx_manager import FxManager
from utils.stock_provider import YFinanceProvider
from .stock import Stock
import datetime


class Portfolio:
    """Manages a collection of stock positions with associated calculations."""

    def __init__(
            self,
            base_currency: str = "GBP",
            cash: float = 0.0,
            fx_manager: Optional[FxManager] = None,
            stock_provider: Optional[YFinanceProvider] = None
    ):
        self.base_currency = base_currency
        self.cash = cash
        self.positions: Dict[str, Stock] = {}

        # Providers for data enrichment
        self.fx_manager = fx_manager
        self.stock_provider = stock_provider or YFinanceProvider()

    def add_position(self, stock: Stock) -> None:
        """Add or update a stock position."""
        if stock.ticker in self.positions:
            # Update quantity if position exists
            existing = self.positions[stock.ticker]
            existing.quantity += stock.quantity
            if existing.quantity <= 0:
                del self.positions[stock.ticker]
        else:
            # Add new position
            if stock.quantity > 0:
                self.positions[stock.ticker] = stock

    def remove_position(self, ticker: str) -> None:
        """Remove a position by ticker."""
        if ticker in self.positions:
            del self.positions[ticker]

    def enrich_positions(self) -> None:
        """Fetch additional data for all positions."""
        for stock in self.positions.values():
            stock.enrich_from_yfinance(self.stock_provider)
            stock.enrich_type()

    def get_total_value(self) -> float:
        """Calculate total portfolio value including cash."""
        if not self.fx_manager:
            raise ValueError("FX Manager not initialized")

        total = self.cash
        for stock in self.positions.values():
            total += stock.get_value_in_base(self.fx_manager, self.base_currency)
        return total

    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to DataFrame for display."""
        if not self.positions:
            return pd.DataFrame()

        records = []
        for stock in self.positions.values():
            value_in_currency = stock.get_value_in_currency()
            value_in_base = stock.get_value_in_base(self.fx_manager, self.base_currency)
            price_in_base = stock.get_price_in_base(self.fx_manager, self.base_currency)

            records.append({
                'ticker': stock.ticker,
                'name': stock.name,
                'quantity': stock.quantity,
                'currency': stock.currency,
                'type': stock.type,
                'sector': stock.sector,
                'sub_industry': stock.sub_industry,
                'price': stock.price,
                'price_in_base': price_in_base,
                'value_in_currency': value_in_currency,
                'value_in_base': value_in_base
            })

        df = pd.DataFrame(records)

        # Sort by value in base currency descending
        return df.sort_values('value_in_base', ascending=False)

    # Add this method to the Portfolio class
    # Add this method to the Portfolio class if it doesn't exist
    def get_historical_values(self, start_date: datetime, end_date: datetime) -> pd.Series:
        historical_data = self.stock_provider.get_historical_data_multiple(
            list(self.positions.keys()), start=start_date, end=end_date
        )
        portfolio_values = pd.Series(index=historical_data.index, dtype=float)
        for date, row in historical_data.iterrows():
            total_value = sum(
                row[symbol] * position.quantity * self.fx_manager.get_rate(position.currency, self.base_currency)
                for symbol, position in self.positions.items()
            )
            portfolio_values[date] = total_value
        return portfolio_values

    @classmethod
    def from_dataframe(
            cls,
            df: pd.DataFrame,
            base_currency: str,
            cash: float = 0.0,
            fx_manager: Optional[FxManager] = None,
            stock_provider: Optional[YFinanceProvider] = None
    ) -> 'Portfolio':
        """Create portfolio from DataFrame with minimal required columns."""
        portfolio = cls(
            base_currency=base_currency,
            cash=cash,
            fx_manager=fx_manager,
            stock_provider=stock_provider
        )

        # Handle both simple and full format
        required_cols = {'ticker', 'quantity'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must have columns: {required_cols}")

        for _, row in df.iterrows():
            stock = Stock(
                ticker=row['ticker'],
                quantity=float(row['quantity']),
                # Optional enriched data if present
                currency=row.get('currency'),
                sector=row.get('sector'),
                sub_industry=row.get('sub_industry'),
                type=row.get('type')
            )
            portfolio.add_position(stock)

        return portfolio

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/analysis/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/analysis/peer_analyzer.py
import logging
import pandas as pd
from typing import List, Dict
from app.models.portfolio import Portfolio
from utils.stock_provider import YFinanceProvider

logger = logging.getLogger(__name__)

class PeerAnalyzer:
    """
    Gathers metric data for a list of stocks in a given 'industry'.
    Possibly uses YFinance or other providers to retrieve specific metrics.
    """

    def __init__(self):
        self.stock_provider = YFinanceProvider()

    def get_peer_metrics(
        self,
        chosen_symbol: str,
        peer_symbols: List[str],
        metrics: List[str],
        portfolio: Portfolio
    ) -> pd.DataFrame:
        """
        For each symbol in peer_symbols, fetch the relevant metrics from YFinance info (or another source).
        Return a DataFrame with columns [symbol, name, <metrics>...].
        The chosen symbol can be included or highlight it if needed.
        """

        if not peer_symbols or not metrics:
            logger.warning("No peer symbols or no metrics to retrieve.")
            return pd.DataFrame()

        records = []
        for sym in peer_symbols:
            info = self.stock_provider.get_info(sym)
            row_data = {
                "symbol": sym,
                "name": info.get("shortName") or info.get("longName") or sym
            }
            # Add each metric if present in info
            for m in metrics:
                row_data[m] = info.get(m, None)
            records.append(row_data)

        df_peers = pd.DataFrame(records)

        # Optionally, sort the DataFrame by symbol or by some metric
        # For example, sort by one of the metrics if you want:
        # if metrics:
        #     df_peers.sort_values(by=metrics[0], ascending=False, inplace=True)

        return df_peers


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/enrichment/sector_metrics_lookup.py
# File: app/models/enrichment/sector_metrics_lookup.py

import logging
from typing import Dict, Any, List
import pandas as pd
import os

logger = logging.getLogger(__name__)

class SectorMetricsLookup:
    """
    Loads a spreadsheet of 'industry' -> up to 20 metrics, plus textual columns like 'Methodology' or
    'Ranking Justification'. Allows retrieving a list of metrics for an industry, plus any textual
    commentary if desired.
    """

    def __init__(self, excel_path: str = "app/assets/sector_valuation_metrics_methodology.xlsx"):
        """
        :param excel_path: Path to the Excel file containing columns like:
          - Industry
          - Metric Rank 1 ... Metric Rank 20
          - Methodology
          - Ranking Justification
        """
        self.excel_path = excel_path
        self.industry_data: Dict[str, Dict[str, Any]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """
        Reads the entire spreadsheet at `self.excel_path` and populates `self.industry_data`.
        Each row must have 'Industry'. We'll gather up to 20 metrics plus optional text fields.
        """
        if not os.path.isfile(self.excel_path):
            logger.error(f"[SectorMetricsLookup] File not found: {self.excel_path}")
            return

        try:
            df = pd.read_excel(self.excel_path)
            if "Industry" not in df.columns:
                logger.error(f"[SectorMetricsLookup] 'Industry' column not found in {self.excel_path}")
                return

            # We'll check each row for up to 20 metrics
            metric_columns = [f"Metric Rank {i}" for i in range(1, 21)]

            for idx, row in df.iterrows():
                industry_name = str(row["Industry"]).strip()

                # Gather all non-empty Metric Rank columns
                metrics_list: List[str] = []
                for mc in metric_columns:
                    if mc in df.columns:
                        cell_val = row.get(mc)
                        if pd.notna(cell_val):
                            metrics_list.append(str(cell_val).strip())

                # We also read these textual columns if present
                methodology = ""
                if "Methodology" in df.columns and pd.notna(row.get("Methodology")):
                    methodology = str(row["Methodology"]).strip()

                ranking_justification = ""
                if "Ranking Justification" in df.columns and pd.notna(row.get("Ranking Justification")):
                    ranking_justification = str(row["Ranking Justification"]).strip()

                self.industry_data[industry_name] = {
                    "metrics": metrics_list,
                    "methodology": methodology,
                    "justification": ranking_justification
                }

            print(f"[DEBUG] Loaded sector metrics for {len(self.industry_data)} industries from {self.excel_path}")
        except Exception as e:
            logger.error(f"[SectorMetricsLookup] Error reading {self.excel_path}: {e}")

    def get_metrics_for_industry(self, industry: str, top_n: int = 5) -> List[str]:
        """
        Return up to 'top_n' metrics for the specified industry.
        If not found, returns an empty list or fallback.
        """
        entry = self.industry_data.get(industry, {})
        all_metrics = entry.get("metrics", [])
        # If top_n is bigger than the length, slicing won't hurt
        return all_metrics[:top_n]

    def get_methodology_text(self, industry: str) -> str:
        """
        Return the 'Methodology' text for the specified industry, or "" if not found.
        """
        entry = self.industry_data.get(industry, {})
        return entry.get("methodology", "")

    def get_justification_text(self, industry: str) -> str:
        """
        Return the 'Ranking Justification' text for the specified industry, or "" if not found.
        """
        entry = self.industry_data.get(industry, {})
        return entry.get("justification", "")


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/models/enrichment/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/performance_controller.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import yfinance as yf

from app.models.portfolio import Portfolio
from utils.stock_provider import YFinanceProvider
from utils.fx_manager import FxManager, FxCache, FxRepository, ExchangeRateHostProvider

logger = logging.getLogger(__name__)


class PerformanceController:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.stock_provider = YFinanceProvider()

        provider = ExchangeRateHostProvider()
        cache = FxCache("utils/data/fx/fx_rates.pkl")
        repo = FxRepository("utils/data/fx/fx_rates.xlsx", "utils/data/fx/fx_rates.db")
        self.fx_manager = FxManager(provider=provider, cache=cache, repository=repo)

    def calculate_performance(self, months: int) -> Dict[str, Any]:
        logger.info(f"Calculating performance for {months} months")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        logger.info(f"Start date: {start_date}, End date: {end_date}")

        performance_data = {
            "portfolio_summary": self._calculate_portfolio_summary(start_date, end_date),
            "best_performers": self._get_best_performers(start_date, end_date),
            "worst_performers": self._get_worst_performers(start_date, end_date),
            "performance_by_currency": self._get_performance_by_attribute("currency", start_date, end_date),
            "performance_by_style": self._get_performance_by_attribute("type", start_date, end_date),
            "performance_by_sector": self._get_performance_by_attribute("sector", start_date, end_date)
        }

        return performance_data

    def _calculate_portfolio_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        initial_value = self._get_portfolio_value_at_date(start_date)
        current_value = self.portfolio.get_total_value()
        gain_loss = current_value - initial_value
        percentage_change = (gain_loss / initial_value) * 100 if initial_value != 0 else 0

        return {
            "initial_value": initial_value,
            "current_value": current_value,
            "gain_loss": gain_loss,
            "percentage_change": percentage_change,
            "base_currency": self.portfolio.base_currency,
            "num_stocks": len(self.portfolio.positions),
            "cash_holdings": self.portfolio.cash
        }

    def _get_portfolio_value_at_date(self, date: datetime) -> float:
        logger.info(f"Getting portfolio value at date: {date}")
        total_value = 0
        for symbol, position in self.portfolio.positions.items():
            historical_data = self.stock_provider.get_historical_data(symbol, start=date, end=date)
            if not historical_data.empty:
                price = historical_data.iloc[0]['Close']
                fx_rate = self.fx_manager.get_rate(position.currency, self.portfolio.base_currency)
                position_value = position.quantity * price * fx_rate
                logger.info(f"Position value for {symbol}: {position_value}")
                total_value += position_value
            else:
                logger.warning(f"No historical data found for {symbol} on {date}")
        logger.info(f"Total portfolio value: {total_value + self.portfolio.cash}")
        return total_value + self.portfolio.cash

    def _get_best_performers(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return self._get_performers(start_date, end_date, top=True)

    def _get_worst_performers(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return self._get_performers(start_date, end_date, top=False)

    def _get_performers(self, start_date: datetime, end_date: datetime, top: bool) -> List[Dict[str, Any]]:
        performances = []
        for symbol, position in self.portfolio.positions.items():
            historical_data = self.stock_provider.get_historical_data(symbol, start=start_date, end=end_date)
            if not historical_data.empty:
                initial_price = historical_data.iloc[0]['Close']
                final_price = historical_data.iloc[-1]['Close']
                performance = (final_price - initial_price) / initial_price * 100
                performances.append({
                    "symbol": symbol,
                    "name": position.name,
                    "sector": position.sector,
                    "performance": performance
                })

        performances.sort(key=lambda x: x["performance"], reverse=top)
        return performances[:5]

    def _get_performance_by_attribute(self, attribute: str, start_date: datetime, end_date: datetime) -> List[
        Dict[str, Any]]:
        performance_by_attribute = {}
        position_weights = self._calculate_position_weights()

        for symbol, position in self.portfolio.positions.items():
            historical_data = self.stock_provider.get_historical_data(symbol, start=start_date, end=end_date)
            if not historical_data.empty:
                initial_price = historical_data.iloc[0]['Close']
                final_price = historical_data.iloc[-1]['Close']
                performance = (final_price - initial_price) / initial_price * 100

                if position.sector == 'FUND':
                    distributed_performance = self._distribute_fund_performance(symbol, performance,
                                                                                position_weights[symbol])
                    for sector, sector_performance in distributed_performance.items():
                        if sector not in performance_by_attribute:
                            performance_by_attribute[sector] = []
                        performance_by_attribute[sector].append(sector_performance)
                else:
                    attr_value = getattr(position, attribute)
                    if attr_value not in performance_by_attribute:
                        performance_by_attribute[attr_value] = []
                    performance_by_attribute[attr_value].append(performance * position_weights[symbol])

        return [{"attribute": attr, "performance": sum(perfs)}
                for attr, perfs in performance_by_attribute.items()]

    def _get_fund_sector_weightings(self, symbol: str) -> Dict[str, float]:
        try:
            ticker = yf.Ticker(symbol)
            sector_weightings = ticker.info.get('sectorWeightings', {})
            return {sector: weight * 100 for sector, weight in sector_weightings.items()}
        except Exception as e:
            logger.error(f"Error fetching sector weightings for fund {symbol}: {str(e)}")
            return {}

    def _calculate_position_weights(self) -> Dict[str, float]:
        total_value = self.portfolio.get_total_value()
        return {symbol: position.get_value_in_base(self.fx_manager, self.portfolio.base_currency) / total_value
                for symbol, position in self.portfolio.positions.items()}

    def _distribute_fund_performance(self, fund_symbol: str, fund_performance: float, fund_weight: float) -> Dict[
        str, float]:
        sector_weightings = self._get_fund_sector_weightings(fund_symbol)
        return {sector: (weight / 100) * fund_performance * fund_weight
                for sector, weight in sector_weightings.items()}

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/peer_analysis_controller.py
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


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/sunburst_controller.py
import os
import pandas as pd
import yfinance as yf
import logging
from typing import Dict, Optional

from app.models.portfolio import Portfolio


class SunburstController:
    """
    Controller for managing data processing and visualization
    for Sunburst charts across different market indices and portfolios.
    """

    SECTOR_COLORS = {
        'Technology': 'blue',
        'Communication Services': 'orange',
        'Healthcare': 'green',
        'Consumer Cyclical': 'gold',
        'Financial Services': 'lightblue',
        'Industrials': 'olive',
        'Consumer Defensive': 'crimson',
        'Energy': 'black',
        'Utilities': 'maroon',
        'Real Estate': 'pink',
        'Basic Materials': 'purple',
        'FUND': 'white',
        'Unknown': 'gray'
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.russell_path = "utils/data/index/data/russell_1000_tickers_components.xlsx"
        self.sp500_path = "utils/data/index/data/sp_500_tickers_components.xlsx"

    def load_index_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process index components data from Excel file.

        Args:
            file_path (str): Path to the index components Excel file

        Returns:
            pd.DataFrame: Processed index data with standardized columns
        """
        try:
            df = pd.read_excel(file_path)
            df['value'] = df['marketCap']
            df['industry'] = df['industry'].fillna('Unknown')
            return df[['stock', 'sector', 'industry', 'value']]
        except Exception as e:
            self.logger.error(f"Error loading index data from {file_path}: {e}")
            return pd.DataFrame(columns=['stock', 'sector', 'industry', 'value'])

    def get_sp500_data(self) -> pd.DataFrame:
        """Get S&P 500 index data."""
        return self.load_index_data(self.sp500_path)

    def get_russell_data(self) -> pd.DataFrame:
        """Get Russell 1000 index data."""
        return self.load_index_data(self.russell_path)

    def get_portfolio_data(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Process portfolio data for Sunburst visualization.
        Handles fund sector re-allocation and data preparation.

        Args:
            portfolio (Portfolio): User's investment portfolio

        Returns:
            pd.DataFrame: Processed portfolio data for visualization
        """
        if portfolio is None:
            return pd.DataFrame(columns=['stock', 'sector', 'industry', 'value'])

        df = portfolio.to_dataframe()
        processed_rows = []

        for _, row in df.iterrows():
            if row['sector'] == 'FUND':
                try:
                    ticker = yf.Ticker(row['ticker'])
                    sector_weightings = ticker.funds_data.sector_weightings

                    if sector_weightings and isinstance(sector_weightings, dict):
                        # Distribute fund value across sectors
                        for sector, weight in sector_weightings.items():
                            mapped_sector = self._map_yfinance_sector(sector)
                            processed_rows.append({
                                'stock': row['name'],
                                'sector': mapped_sector,
                                'industry': 'FUND',
                                'value': row['value_in_base'] * weight
                            })
                    else:
                        # If no sector weightings, keep as FUND
                        processed_rows.append({
                            'stock': row['name'],
                            'sector': 'FUND',
                            'industry': 'FUND',
                            'value': row['value_in_base']
                        })
                except Exception as e:
                    self.logger.error(f"Error processing fund {row['ticker']}: {e}")
                    processed_rows.append({
                        'stock': row['name'],
                        'sector': 'FUND',
                        'industry': 'FUND',
                        'value': row['value_in_base']
                    })
            else:
                processed_rows.append({
                    'stock': row['name'],
                    'sector': row['sector'],
                    'industry': row['sub_industry'] or 'Unknown',
                    'value': row['value_in_base']
                })

        return pd.DataFrame(processed_rows)

    def _map_yfinance_sector(self, yfinance_sector: str) -> str:
        """
        Map YFinance sector names to our standard sector classification.

        Args:
            yfinance_sector (str): Sector name from YFinance

        Returns:
            str: Mapped sector name
        """
        sector_map = {
            'technology': 'Technology',
            'consumer_cyclical': 'Consumer Cyclical',
            'communication_services': 'Communication Services',
            'consumer_defensive': 'Consumer Defensive',
            'financial_services': 'Financial Services',
            'industrials': 'Industrials',
            'healthcare': 'Healthcare',
            'basic_materials': 'Basic Materials',
            'real_estate': 'Real Estate',
            'energy': 'Energy',
            'utilities': 'Utilities'
        }
        return sector_map.get(yfinance_sector.lower(), 'Unknown')

    def get_sector_colors(self) -> Dict[str, str]:
        """
        Retrieve sector color mapping.

        Returns:
            Dict[str, str]: Mapping of sectors to colors
        """
        return self.SECTOR_COLORS

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/portfolio_controller.py
import pandas as pd
import yfinance as yf
import streamlit as st
from typing import Optional, List, Dict
import logging

from app.models.portfolio import Portfolio
from app.models.stock import Stock
from utils.fx_manager import (
    FxManager, FxCache, FxRepository, ExchangeRateHostProvider
)
from utils.stock_provider import YFinanceProvider


class PortfolioController:
    """Controls portfolio operations and manages state between models and views."""

    def __init__(self):
        """Initialize controller with FX and Stock providers."""
        # Initialize providers if not in session state
        if 'fx_manager' not in st.session_state:
            st.session_state.fx_manager = self._initialize_fx_manager()

        if 'stock_provider' not in st.session_state:
            st.session_state.stock_provider = YFinanceProvider()

        # Initialize portfolio state
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = None

        self.fx_manager: FxManager = st.session_state.fx_manager
        self.stock_provider: YFinanceProvider = st.session_state.stock_provider
        self.logger = logging.getLogger(__name__)

    def _initialize_fx_manager(self) -> FxManager:
        """Create and initialize FX manager with default settings."""
        provider = ExchangeRateHostProvider()
        cache = FxCache("utils/data/fx/fx_rates.pkl")
        repo = FxRepository(
            "utils/data/fx/fx_rates.xlsx",
            "utils/data/fx/fx_rates.db"
        )

        fx_manager = FxManager(
            provider=provider,
            cache=cache,
            repository=repo,
            stale_threshold_hours=48
        )

        # Initialize with default base currency
        fx_manager.ensure_fresh_data(base="GBP")
        return fx_manager

    def handle_file_upload(
            self,
            uploaded_file,
            base_currency: str,
            cash: float = 0.0
    ) -> None:
        """Process uploaded portfolio file and update state."""
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)

            # Validate required columns
            required_cols = {'ticker', 'quantity'}
            if not required_cols.issubset(df.columns):
                st.error(f"Upload must have columns: {required_cols}")
                return

            # Clean data
            df['ticker'] = df['ticker'].str.strip().str.upper()
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

            # Remove any rows with NaN quantities
            df = df.dropna(subset=['quantity'])

            # Create portfolio from DataFrame
            portfolio = Portfolio.from_dataframe(
                df,
                base_currency=base_currency,
                cash=cash,
                fx_manager=self.fx_manager,
                stock_provider=self.stock_provider
            )

            # Enrich with market data
            with st.spinner("Fetching market data..."):
                portfolio.enrich_positions()

            # Update session state
            st.session_state.portfolio = portfolio
            st.success("Portfolio processed successfully!")

        except Exception as e:
            st.error(f"Error processing portfolio: {str(e)}")

    def get_portfolio(self) -> Optional[Portfolio]:
        """Get current portfolio from session state."""
        return st.session_state.get('portfolio')

    def get_available_currencies(self) -> List[str]:
        """Get list of available currencies for base currency selection."""
        return [
            "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "HKD", "GBp"
        ]

    def clear_portfolio(self) -> None:
        """Clear current portfolio from session state."""
        st.session_state.portfolio = None

    def update_base_currency(self, new_currency: str) -> None:
        """Update portfolio base currency."""
        portfolio = self.get_portfolio()
        if portfolio:
            portfolio.base_currency = new_currency
            # Refresh FX data with new base
            self.fx_manager.ensure_fresh_data(base=new_currency)

    def update_cash(self, cash: float) -> None:
        """Update portfolio cash position."""
        portfolio = self.get_portfolio()
        if portfolio:
            portfolio.cash = cash

    def distribute_fund_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Distribute fund sector weights across other sectors.

        Args:
            df (pd.DataFrame): DataFrame of portfolio positions

        Returns:
            pd.DataFrame: DataFrame with fund sectors distributed
        """
        # Create a copy to avoid modifying the original
        distributed_df = df.copy()

        # Identify fund rows
        fund_rows = distributed_df[distributed_df['sector'] == 'FUND']

        # Remove fund rows from the main dataframe
        distributed_df = distributed_df[distributed_df['sector'] != 'FUND']

        # Process each fund
        for _, fund_row in fund_rows.iterrows():
            try:
                # Fetch fund sector weightings
                ticker = yf.Ticker(fund_row['ticker'])
                fund_sector_weights = ticker.funds_data.sector_weightings

                # Distribute fund value across sectors
                for yf_sector, weight in fund_sector_weights.items():
                    # Map YFinance sector to our sector classification
                    mapped_sector = self._map_sector(yf_sector)

                    # Calculate distributed value
                    distributed_value = fund_row['value_in_base'] * weight

                    # Check if sector exists, if not add a new row
                    matching_rows = distributed_df[distributed_df['sector'] == mapped_sector]

                    if not matching_rows.empty:
                        # Add to existing sector's value
                        distributed_df.loc[
                            distributed_df['sector'] == mapped_sector, 'value_in_base'] += distributed_value
                    else:
                        # Create a new row for this sector
                        new_row = fund_row.copy()
                        new_row['sector'] = mapped_sector
                        new_row['value_in_base'] = distributed_value
                        distributed_df = pd.concat([distributed_df, pd.DataFrame([new_row])], ignore_index=True)

            except Exception as e:
                self.logger.error(f"Error processing fund {fund_row['ticker']}: {str(e)}")

        return distributed_df

    def _map_sector(self, yf_sector: str) -> str:
        """Map YFinance sector names to our sector classification."""
        sector_map = {
            'technology': 'Technology',
            'consumer_cyclical': 'Consumer Cyclical',
            'communication_services': 'Communication Services',
            'consumer_defensive': 'Consumer Defensive',
            'financial_services': 'Financial Services',
            'industrials': 'Industrials',
            'healthcare': 'Healthcare',
            'basic_materials': 'Basic Materials',
            'real_estate': 'Real Estate',
            'energy': 'Energy',
            'utilities': 'Utilities'
        }
        return sector_map.get(yf_sector, 'Unknown')

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/portfolio_divesification_controller.py
import os
import pandas as pd
import yfinance as yf
import logging
from typing import Dict, Optional

from app.models.portfolio import Portfolio

class PortfolioDiversificationController:
    """
    Controller for handling visualization data processing,
    specifically for sunburst and treemap charts.
    """

    SECTOR_COLORS = {
        'Technology': 'blue',
        'Communication Services': 'orange',
        'Healthcare': 'green',
        'Consumer Cyclical': 'gold',
        'Financial Services': 'lightblue',
        'Industrials': 'olive',
        'Consumer Defensive': 'crimson',
        'Energy': 'black',
        'Utilities': 'maroon',
        'Real Estate': 'pink',
        'Basic Materials': 'purple',
        'FUND': 'white',
        'Unknown': 'gray'
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.russell_path = "utils/data/index/data/russell_1000_tickers_components.xlsx"
        self.sp500_path = "utils/data/index/data/sp_500_tickers_components.xlsx"

    def load_index_data(self, file_path: str) -> pd.DataFrame:
        """
        Load index components data from Excel file.
        """
        try:
            df = pd.read_excel(file_path)
            df['value'] = df['marketCap']
            return df[['stock', 'sector', 'industry', 'value']]
        except Exception as e:
            self.logger.error(f"Error loading index data from {file_path}: {e}")
            return pd.DataFrame()

    def get_sp500_data(self) -> pd.DataFrame:
        """Get S&P 500 index data."""
        return self.load_index_data(self.sp500_path)

    def get_russell_data(self) -> pd.DataFrame:
        """Get Russell 1000 index data."""
        return self.load_index_data(self.russell_path)

    def get_portfolio_data(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Process portfolio data for visualization.
        Handles fund re-allocation if possible.
        """
        df = portfolio.to_dataframe()

        # Process each row, handling funds separately
        processed_rows = []
        for _, row in df.iterrows():
            if row['sector'] == 'FUND':
                try:
                    ticker = yf.Ticker(row['ticker'])
                    sector_weightings = ticker.funds_data.sector_weightings

                    if sector_weightings:
                        # Distribute fund value across sectors
                        for sector, weight in sector_weightings.items():
                            processed_rows.append({
                                'stock': row['name'],
                                'sector': sector,
                                'industry': 'FUND',
                                'value': row['value_in_base'] * weight
                            })
                    else:
                        # If no sector weightings, keep as FUND
                        processed_rows.append({
                            'stock': row['name'],
                            'sector': 'FUND',
                            'industry': 'FUND',
                            'value': row['value_in_base']
                        })
                except Exception as e:
                    self.logger.error(f"Error processing fund {row['ticker']}: {e}")
                    processed_rows.append({
                        'stock': row['name'],
                        'sector': 'FUND',
                        'industry': 'FUND',
                        'value': row['value_in_base']
                    })
            else:
                processed_rows.append({
                    'stock': row['name'],
                    'sector': row['sector'],
                    'industry': row['sub_industry'],
                    'value': row['value_in_base']
                })

        return pd.DataFrame(processed_rows)

    def get_sector_colors(self) -> Dict[str, str]:
        """Return sector color mapping."""
        return self.SECTOR_COLORS

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/benchmark_controller.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from utils.stock_provider import YFinanceProvider
from app.models.portfolio import Portfolio
import streamlit as st

class BenchmarkController:
    def __init__(self):
        self.stock_provider = YFinanceProvider()
        self.equity_tickers = {
            "S&P 500": "^GSPC",
            "NASDAQ 100": "^NDX",
            "Russell 2000": "^RUT",
            "Russell 1000": "^RUI",
            "Dow Jones IA": "^DJI",
            "S&P/TSX": "^GSPTSE",
            "FTSE 100": "^FTSE",
            "CAC 40": "^FCHI",
            "DAX (GER 40)": "^GDAXI",
            "Euro Stoxx 50": "^STOXX50E",
            "Swiss Market": "^SSMI"
        }
        self.sector_etf_tickers = {
            "Communication Services": "XLC",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Financials": "XLF",
            "Health Care": "XLV",
            "Industrials": "XLI",
            "Materials": "XLB",
            "Real Estate": "XLRE",
            "Technology": "XLK",
            "Utilities": "XLU"
        }
        self.crypto_tickers = {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD",
            "Binance Coin": "BNB-USD",
            "XRP": "XRP-USD",
            "Cardano": "ADA-USD",
            "Dogecoin": "DOGE-USD",
            "Litecoin": "LTC-USD"
        }
        self.other_asset_tickers = {
            # Energy
            "WTI Crude": "CL=F",
            "Brent Crude": "BZ=F",
            "Natural Gas": "NG=F",
            # Metals
            "Gold": "GC=F",
            "Silver": "SI=F",
            "Copper": "HG=F",
            # Agriculture
            "Corn": "ZC=F",
            "Wheat": "ZW=F",
            "Coffee": "KC=F"
        }

    def get_benchmark_data(self, tickers: Dict[str, str], months: int) -> pd.DataFrame:
        print(f"Fetching data for tickers: {tickers}")  # Debug print
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        provider = YFinanceProvider()
        df = provider.get_historical_data_multiple(list(tickers.values()), start=start_date, end=end_date)
        df = df.rename(columns={v: k for k, v in tickers.items()})
        return (df / df.iloc[0] - 1) * 100

    def get_equity_data(self, months: int) -> pd.DataFrame:
        return self.get_benchmark_data(self.equity_tickers, months)

    def get_sector_data(self, months: int) -> pd.DataFrame:
        return self.get_benchmark_data(self.sector_etf_tickers, months)

    def get_crypto_data(self, months: int) -> pd.DataFrame:
        return self.get_benchmark_data(self.crypto_tickers, months)

    def get_other_asset_data(self, months: int) -> pd.DataFrame:
        return self.get_benchmark_data(self.other_asset_tickers, months)

    def get_portfolio_performance(self, portfolio: 'Portfolio', months: int) -> pd.Series:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        portfolio_values = portfolio.get_historical_values(start_date, end_date)
        return (portfolio_values / portfolio_values.iloc[0] - 1) * 100

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/controllers/treemap_controller.py
import os
import pandas as pd
import yfinance as yf
import logging
from typing import Dict, Optional

from app.models.portfolio import Portfolio
from app.models.stock import sector_to_type


class TreemapController:
    """
    Controller for managing data processing and visualization
    for Treemap charts across different market indices and portfolios.
    """

    SECTOR_COLORS = {
        'Technology': 'blue',
        'Communication Services': 'orange',
        'Healthcare': 'green',
        'Consumer Cyclical': 'gold',
        'Financial Services': 'lightblue',
        'Industrials': 'olive',
        'Consumer Defensive': 'crimson',
        'Energy': 'black',
        'Utilities': 'maroon',
        'Real Estate': 'pink',
        'Basic Materials': 'purple',
        'FUND': 'white',
        'Unknown': 'gray'
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.russell_path = "utils/data/index/data/russell_1000_tickers_components.xlsx"
        self.sp500_path = "utils/data/index/data/sp_500_tickers_components.xlsx"

    def load_index_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process index components data from Excel file.

        Args:
            file_path (str): Path to the index components Excel file

        Returns:
            pd.DataFrame: Processed index data with standardized columns
        """
        try:
            df = pd.read_excel(file_path)
            df['value'] = df['marketCap']
            df['type'] = df.apply(lambda row: self._determine_type(row['sector']), axis=1)
            df['industry'] = df['industry'].fillna('Unknown')
            return df[['type', 'sector', 'industry', 'stock', 'value']]
        except Exception as e:
            self.logger.error(f"Error loading index data from {file_path}: {e}")
            return pd.DataFrame(columns=['type', 'sector', 'industry', 'stock', 'value'])

    def get_sp500_data(self) -> pd.DataFrame:
        """Get S&P 500 index data."""
        return self.load_index_data(self.sp500_path)

    def get_russell_data(self) -> pd.DataFrame:
        """Get Russell 1000 index data."""
        return self.load_index_data(self.russell_path)

    def get_portfolio_data(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Process portfolio data for Treemap visualization.
        Handles fund sector re-allocation and data preparation.

        Args:
            portfolio (Portfolio): User's investment portfolio

        Returns:
            pd.DataFrame: Processed portfolio data for visualization
        """
        if portfolio is None:
            return pd.DataFrame(columns=['type', 'sector', 'industry', 'stock', 'value'])

        df = portfolio.to_dataframe()
        processed_rows = []

        for _, row in df.iterrows():
            if row['sector'] == 'FUND':
                try:
                    ticker = yf.Ticker(row['ticker'])
                    sector_weightings = ticker.funds_data.sector_weightings

                    if sector_weightings and isinstance(sector_weightings, dict):
                        # Distribute fund value across sectors
                        for sector, weight in sector_weightings.items():
                            mapped_sector = self._map_yfinance_sector(sector)
                            mapped_type = self._determine_type(mapped_sector)
                            processed_rows.append({
                                'type': mapped_type,
                                'stock': row['name'],
                                'sector': mapped_sector,
                                'industry': 'FUND',
                                'value': row['value_in_base'] * weight
                            })
                    else:
                        # If no sector weightings, keep as FUND
                        processed_rows.append({
                            'type': 'Diversified Fund',
                            'stock': row['name'],
                            'sector': 'FUND',
                            'industry': 'FUND',
                            'value': row['value_in_base']
                        })
                except Exception as e:
                    self.logger.error(f"Error processing fund {row['ticker']}: {e}")
                    processed_rows.append({
                        'type': 'Diversified Fund',
                        'stock': row['name'],
                        'sector': 'FUND',
                        'industry': 'FUND',
                        'value': row['value_in_base']
                    })
            else:
                mapped_type = self._determine_type(row['sector'])
                processed_rows.append({
                    'type': mapped_type,
                    'stock': row['name'],
                    'sector': row['sector'],
                    'industry': row['sub_industry'] or 'Unknown',
                    'value': row['value_in_base']
                })

        return pd.DataFrame(processed_rows)

    def _map_yfinance_sector(self, yfinance_sector: str) -> str:
        """
        Map YFinance sector names to our standard sector classification.

        Args:
            yfinance_sector (str): Sector name from YFinance

        Returns:
            str: Mapped sector name
        """
        sector_map = {
            'technology': 'Technology',
            'consumer_cyclical': 'Consumer Cyclical',
            'communication_services': 'Communication Services',
            'consumer_defensive': 'Consumer Defensive',
            'financial_services': 'Financial Services',
            'industrials': 'Industrials',
            'healthcare': 'Healthcare',
            'basic_materials': 'Basic Materials',
            'real_estate': 'Real Estate',
            'energy': 'Energy',
            'utilities': 'Utilities'
        }
        return sector_map.get(yfinance_sector.lower(), 'Unknown')

    def _determine_type(self, sector: str) -> str:
        """
        Determine the investment type (Defensive, Cyclical, Growth) based on the sector.

        Args:
            sector (str): Sector name

        Returns:
            str: Investment type
        """
        return sector_to_type.get(sector, 'Unknown')

    def get_sector_colors(self) -> Dict[str, str]:
        """
        Retrieve sector color mapping.

        Returns:
            Dict[str, str]: Mapping of sectors to colors
        """
        return self.SECTOR_COLORS

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/benchmark_view.py
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



# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/portfolio_view.py
import streamlit as st
import pandas as pd
from app.controllers.portfolio_controller import PortfolioController


class PortfolioView:
    """Handles the rendering of portfolio data and controls in the Streamlit UI."""

    def render_sidebar(self, controller: PortfolioController) -> None:
        """Render the sidebar with upload controls and portfolio settings."""
        st.sidebar.image('app/assets/lucidate.png', width=200)
        st.sidebar.title("Portfolio Controls")

        # File upload section with format explanation
        st.sidebar.markdown("### Upload Portfolio")
        st.sidebar.markdown("""
        Upload an Excel file with your portfolio data. in the following format
        Required columns: 
        - `ticker`, **must** be from Yahoo Finance 
        - `quantity`

        """)

        uploaded_file = st.sidebar.file_uploader(
            "Choose Excel file",
            type=['xlsx']
        )

        base_currency = st.sidebar.selectbox(
            "Base Currency",
            options=controller.get_available_currencies(),
            index=2  # GBP default
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
            st.dataframe(data=pd.DataFrame(alloc_data),
                column_config=column_config,
                hide_index=True)

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/performance_view.py
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

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/treemap_view.py
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

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/portfolio_diversification_view.py
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

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/peer_analysis_view.py
# File: app/views/peer_analysis_view.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from app.controllers.peer_analysis_controller import PeerAnalysisController
from app.models.portfolio import Portfolio
import pandas as pd

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

                def highlight_chosen(row):
                    return [
                        "background-color: yellow; color: black"
                        if row.get("symbol") == chosen_symbol else ""
                        for _ in row
                    ]

                st.dataframe(
                    df_peers.style.apply(highlight_chosen, axis=1),
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

            # Bar chart for total returns
            fig_bar = px.bar(
                returns_df,
                x="stock",
                y="return",
                color="is_selected",
                color_discrete_map={True: "yellow", False: "blue"},
                title=f"Total Return over {period_choice}"
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


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/app/views/sunburst_view.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from app.controllers.sunburst_controller import SunburstController
from app.models.portfolio import Portfolio


class SunburstView:
    def __init__(self):
        self.controller = SunburstController()

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

        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['sector', 'industry', 'stock', 'value'])

        # Replace any remaining NaNs with 'Unknown'
        df['sector'] = df['sector'].fillna('Unknown')
        df['industry'] = df['industry'].fillna('Unknown')
        df['stock'] = df['stock'].fillna('Unknown')

        # Ensure value is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)

        # Remove rows with zero or negative values
        df = df[df['value'] > 0]

        if df.empty:
            fig = px.sunburst(
                pd.DataFrame({"sector": [], "industry": [], "stock": [], "value": []}),
                path=["sector", "industry", "stock"],
                values="value",
            )
            fig.update_layout(title=f"{title} (No Valid Data)")
            return fig

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

    def render_sunburst_tab(self, portfolio: Portfolio):
        """Render Sunburst charts for different market indices and portfolio."""
        st.header("Portfolio Composition")

        if portfolio is None:
            st.info("Please upload a portfolio file to view composition")
            return

        # Get visualization data
        sp500_data = self.controller.get_sp500_data()
        russell_data = self.controller.get_russell_data()
        portfolio_data = self.controller.get_portfolio_data(portfolio)

        # Create columns for charts
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

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/utils/fx_manager.py
# File: TBIC_Jubilee/utils/fx_manager.py


import os
import pickle
import datetime
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pandas as pd
import sqlite3  # or use SQLAlchemy if you prefer
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------
# 1. Abstract Strategy: FxProvider
# -------------------------------------
class FxProvider(ABC):
    """
    Abstract base class defining the interface for fetching FX rates.
    """

    @abstractmethod
    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Return a dict of {(base, quote): rate, (quote, base): 1/rate} pairs.
        Must handle special logic for GBp if needed.
        """
        pass


# -------------------------------------------------
# 2. Concrete Strategy: ExchangeRateHostProvider
# -------------------------------------------------
class ExchangeRateHostProvider(FxProvider):
    """
    Fetch rates from ExchangeRate.host (or Apilayer).
    Example API doc: https://apilayer.com/marketplace/exchangerates_data-api
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ERHOST_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Please set ERHOST_API_KEY in environment variables.")

    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Calls the external API to fetch latest rates from base_ccy to each quote.
        Returns a dictionary of cross pairs {(base, quote): rate, (quote, base): inverse_rate}.
        """
        symbols_str = ",".join(quote_ccys)
        url = f"https://api.apilayer.com/exchangerates_data/latest?base={base_ccy}&symbols={symbols_str}"
        headers = {"apikey": self.api_key}

        resp = requests.get(url, headers=headers)
        data = resp.json()

        if not data.get("success", False):
            error_info = data.get("error", {}).get("info", "Unknown error")
            raise RuntimeError(f"Exchange rates request failed: {error_info}")

        base_rates = data.get("rates", {})
        result: Dict[Tuple[str, str], float] = {}

        for quote_ccy, rate_val in base_rates.items():
            if rate_val == 0.0:
                continue
            result[(base_ccy, quote_ccy)] = float(rate_val)
            result[(quote_ccy, base_ccy)] = 1.0 / float(rate_val)

        return result


# -------------------------------------
# 3. FxCache (pickle-based)
# -------------------------------------
class FxCache:
    """
    Responsible for loading/saving FX data from/to a pickle file.
    We store: { "rates": {...}, "timestamp": datetime }
    """

    def __init__(self, pickle_path: str):
        self.pickle_path = Path(pickle_path)

    def load(self) -> Optional[Dict]:
        """Load the pickle if it exists. Returns dict with 'rates' and 'timestamp' or None."""
        if not self.pickle_path.exists():
            return None
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    def save(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """Store rates and timestamp into the pickle."""
        data = {"rates": rates, "timestamp": timestamp}
        self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(data, f)


# -------------------------------------
# 4. FxRepository (Excel & DB I/O)
# -------------------------------------
class FxRepository:
    """
    Handles writing cross rates to/from Excel and a DB table.
    """

    def __init__(self, excel_path: str, db_path: str):
        self.excel_path = excel_path
        self.db_path = db_path

    def save_rates_to_excel(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """
        Saves cross rates to Excel in a tall format:
            base_currency | quote_currency | fx_rate | timestamp
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:  # ignore identity pairs
                records.append({
                    "base_currency": base,
                    "quote_currency": quote,
                    "fx_rate": rate,
                    "timestamp": timestamp
                })
        df = pd.DataFrame(records)
        df.to_excel(self.excel_path, index=False)

    def save_rates_to_database(self,
                               rates: Dict[Tuple[str, str], float],
                               timestamp: datetime.datetime,
                               table_name: str = "FX_Rates"):
        """
        Save cross rates to a DB table using sqlite3 (or any other DB solution).
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:
                records.append((base, quote, rate, timestamp.isoformat()))

        df = pd.DataFrame(records, columns=["base_currency", "quote_currency", "fx_rate", "timestamp"])
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.close()


# -------------------------------------
# 5. FxManager (Facade / Context)
# -------------------------------------
class FxManager:
    """
    Orchestrates fetching rates (via a selected FxProvider), caching them in a pickle,
    saving them to an Excel file, and storing them in a DB.
    Also handles stale data logic, fallback, and the "GBp" <-> "GBP" relationship.
    """

    G8_CURRENCIES = [
        "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "HKD",
        "GBp"  # Pence
    ]

    def __init__(self,
                 provider: FxProvider,
                 cache: FxCache,
                 repository: FxRepository,
                 stale_threshold_hours: int = 48):
        """
        :param provider:    Strategy for fetching FX from an API
        :param cache:       Manages loading/saving a pickle of rates
        :param repository:  For writing to Excel & DB
        :param stale_threshold_hours: 48 hours default
        """
        self.provider = provider
        self.cache = cache
        self.repository = repository
        self.stale_threshold = datetime.timedelta(hours=stale_threshold_hours)

        self.rates: Dict[Tuple[str, str], float] = {}
        self.last_timestamp: datetime.datetime = datetime.datetime.min

    def initialize_rates(self, base: str = "USD"):
        """
        Called once at app startup to ensure we have fresh (or cached) data loaded.
        If there's a valid pickle that isn't stale, load from it.
        Otherwise fetch from the provider, save to pickle/Excel/DB.
        """
        cached_data = self.cache.load()
        now = datetime.datetime.now()

        if cached_data:
            self.rates = cached_data["rates"]
            self.last_timestamp = cached_data["timestamp"]
        else:
            self.rates = {}
            self.last_timestamp = datetime.datetime.min

        age = now - self.last_timestamp
        if age < self.stale_threshold and self.rates:
            # Not stale, do nothing
            print(f"[FxManager] Using cached rates from {self.last_timestamp}")
            return

        # Otherwise fetch new
        print("[FxManager] Fetching fresh rates...")
        self._fetch_and_persist(base)

    def _fetch_and_persist(self, base: str):
        """Fetch from provider, build cross matrix, save to pickle/Excel/DB."""
        new_rates = self._fetch_full_matrix(base=base)
        self.rates = new_rates
        self.last_timestamp = datetime.datetime.now()

        self.cache.save(self.rates, self.last_timestamp)
        self.repository.save_rates_to_excel(self.rates, self.last_timestamp)
        self.repository.save_rates_to_database(self.rates, self.last_timestamp)

        print(f"[FxManager] Updated rates at {self.last_timestamp}")

    def _fetch_full_matrix(self, base: str) -> Dict[Tuple[str, str], float]:
        # Exclude base and GBp for direct fetch
        quote_ccys = [c for c in self.G8_CURRENCIES if c not in [base, "GBp"]]
        direct_rates = self.provider.fetch_rates(base, quote_ccys)

        # Start building full matrix
        full_matrix = dict(direct_rates)
        for c in self.G8_CURRENCIES:
            full_matrix[(c, c)] = 1.0

        all_ccys = set(self.G8_CURRENCIES)
        for x in all_ccys:
            for y in all_ccys:
                if x == y:
                    continue
                if (x, base) in full_matrix and (base, y) in full_matrix:
                    full_matrix[(x, y)] = full_matrix[(x, base)] * full_matrix[(base, y)]

        # Insert GBP -> GBp = 100, vice versa
        full_matrix[("GBP", "GBp")] = 100.0
        full_matrix[("GBp", "GBP")] = 0.01

        for c in all_ccys:
            if c == "GBp":
                continue
            if ("GBP", c) in full_matrix:
                full_matrix[("GBp", c)] = full_matrix[("GBP", c)] * 0.01
            if (c, "GBP") in full_matrix:
                full_matrix[(c, "GBp")] = full_matrix[(c, "GBP")] * 100.0

        return full_matrix

    def get_rate(self, from_ccy: str, to_ccy: str) -> float:
        """
        Return the rate from_ccy -> to_ccy.
        If not found, returns 0.0. Data must be pre-fetched or loaded by `initialize_rates()`.
        """
        return self.rates.get((from_ccy, to_ccy), 0.0)

    def fetch_and_store_all(self, base: str = "USD"):
        """
        If you want a manual CLI approach to forcibly refresh rates.
        """
        self._fetch_and_persist(base)
        print(f"[FxManager] fetch_and_store_all complete.")


# -------------- Example main usage --------------
if __name__ == "__main__":
    provider = ExchangeRateHostProvider()
    cache = FxCache("data/fx/fx_rates.pkl")
    repo = FxRepository("data/fx/fx_rates.xlsx", "data/fx/fx_rates.db")

    fx_manager = FxManager(provider, cache, repo)
    # Will load from existing pickle if fresh, or fetch new if stale/missing
    fx_manager.initialize_rates(base="GBP")

    rate_gbp_to_usd = fx_manager.get_rate("GBP", "USD")
    print(f"1 GBP = {rate_gbp_to_usd:.4f} USD")

import pandas as pd
import sqlite3  # or use SQLAlchemy if you prefer
from dotenv import load_dotenv

load_dotenv()


# -------------------------------------
# 1. Abstract Strategy: FxProvider
# -------------------------------------
class FxProvider(ABC):
    """
    Abstract base class defining the interface for fetching FX rates.
    """

    @abstractmethod
    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Return a dict of {(base, quote): rate, (quote, base): 1/rate} pairs.
        Must handle special logic for GBp if needed.
        """
        pass


# -------------------------------------------------
# 2. Concrete Strategy: ExchangeRateHostProvider
# -------------------------------------------------
class ExchangeRateHostProvider(FxProvider):
    """
    Fetch rates from ExchangeRate.host (or Apilayer).
    Example API doc: https://apilayer.com/marketplace/exchangerates_data-api
    """

    def __init__(self, api_key: Optional[str] = None):
        # Attempt to load from environment if none passed
        self.api_key = api_key or os.getenv("ERHOST_API_KEY")

        if not self.api_key:
            raise ValueError("No API key found. Please set ERHOST_API_KEY in environment variables.")

    def fetch_rates(self, base_ccy: str, quote_ccys: List[str]) -> Dict[Tuple[str, str], float]:
        """
        Calls the external API to fetch the latest rates from base_ccy to each quote.
        Returns a dictionary of cross pairs {(base, quote): rate, (quote, base): inverse_rate}.
        """
        symbols_str = ",".join(quote_ccys)
        url = f"https://api.apilayer.com/exchangerates_data/latest?base={base_ccy}&symbols={symbols_str}"
        headers = {"apikey": self.api_key}

        # Make the request (wrap in backoff if needed)
        resp = requests.get(url, headers=headers)
        data = resp.json()

        # Check for success
        if not data.get("success", False):
            error_info = data.get("error", {}).get("info", "Unknown error")
            raise RuntimeError(f"Exchange rates request failed: {error_info}")

        base_rates = data.get("rates", {})
        result: Dict[Tuple[str, str], float] = {}

        # Populate direct pairs from the API
        for quote_ccy, rate_val in base_rates.items():
            if rate_val == 0.0:
                continue
            result[(base_ccy, quote_ccy)] = float(rate_val)
            # Also store the inverse
            result[(quote_ccy, base_ccy)] = 1.0 / float(rate_val)

        return result


# -------------------------------------
# 3. FxCache (pickle-based)
# -------------------------------------
class FxCache:
    """
    Responsible for loading/saving FX data from/to a pickle file.
    We store: { "rates": {...}, "timestamp": datetime }
    """

    def __init__(self, pickle_path: str):
        self.pickle_path = Path(pickle_path)

    def load(self) -> Optional[Dict]:
        """
        Load the pickle if it exists. Returns dict with 'rates' and 'timestamp' or None if no file.
        """
        if not self.pickle_path.exists():
            return None
        with open(self.pickle_path, "rb") as f:
            return pickle.load(f)

    def save(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """
        Store rates and timestamp into the pickle.
        """
        data = {"rates": rates, "timestamp": timestamp}
        self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_path, "wb") as f:
            pickle.dump(data, f)


# -------------------------------------
# 4. FxRepository (Excel & DB I/O)
# -------------------------------------
class FxRepository:
    """
    Handles writing (and possibly reading) cross rates to/from Excel and a DB table.
    """

    def __init__(self, excel_path: str, db_path: str):
        self.excel_path = excel_path
        self.db_path = db_path

    def save_rates_to_excel(self, rates: Dict[Tuple[str, str], float], timestamp: datetime.datetime):
        """
        Saves cross rates to Excel in a tall format:
            base_currency | quote_currency | fx_rate | timestamp
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:  # ignore identity pairs (ccy->same ccy)
                records.append({
                    "base_currency": base,
                    "quote_currency": quote,
                    "fx_rate": rate,
                    "timestamp": timestamp
                })
        df = pd.DataFrame(records)
        df.to_excel(self.excel_path, index=False)

    def save_rates_to_database(self,
                               rates: Dict[Tuple[str, str], float],
                               timestamp: datetime.datetime,
                               table_name: str = "FX_Rates"):
        """
        Save cross rates to a DB table using sqlite3 (or any other DB solution).
        Table schema example:
            CREATE TABLE FX_Rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                base_currency TEXT,
                quote_currency TEXT,
                fx_rate REAL,
                timestamp TEXT
            );
        """
        records = []
        for (base, quote), rate in rates.items():
            if base != quote:
                records.append((base, quote, rate, timestamp.isoformat()))

        df = pd.DataFrame(records, columns=["base_currency", "quote_currency", "fx_rate", "timestamp"])
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                df.to_sql(table_name, conn, if_exists="append", index=False)
        finally:
            conn.close()


# -------------------------------------
# 5. FxManager (Facade / Context)
# -------------------------------------
class FxManager:
    """
    Orchestrates fetching rates (via a selected FxProvider), caching them in a pickle,
    saving them to an Excel file, and storing them in a DB.
    Also handles stale data logic, fallback, and the "GBp" <-> "GBP" relationship.
    """

    G8_CURRENCIES = [
        "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "HKD",
        "GBp"  # Pence
    ]

    def __init__(self,
                 provider: FxProvider,
                 cache: FxCache,
                 repository: FxRepository,
                 stale_threshold_hours: int = 48):
        """
        :param provider:    The chosen Strategy for fetching FX from an API
        :param cache:       Manages loading/saving a pickle of rates
        :param repository:  For writing to Excel & DB
        :param stale_threshold_hours:  Default = 48 hours
        """
        self.provider = provider
        self.cache = cache
        self.repository = repository
        self.stale_threshold = datetime.timedelta(hours=stale_threshold_hours)

        self.rates: Dict[Tuple[str, str], float] = {}
        self.last_timestamp: datetime.datetime = datetime.datetime.min

    def ensure_fresh_data(self, base: str):
        """
        Loads existing rates from pickle. If older than stale_threshold, tries to fetch new rates.
        If new fetch fails, we keep the old rates and note it.
        """
        cached_data = self.cache.load()
        now = datetime.datetime.now()

        if cached_data:
            self.rates = cached_data["rates"]
            self.last_timestamp = cached_data["timestamp"]
        else:
            # No pickle at all
            self.rates = {}
            self.last_timestamp = datetime.datetime.min

        # Check staleness
        age = now - self.last_timestamp
        if age < self.stale_threshold and self.rates:
            # Data is still fresh enough, do nothing
            return

        # Attempt to fetch new data
        try:
            new_rates = self._fetch_full_matrix(base=base)
            self.rates = new_rates
            self.last_timestamp = now

            # Save to pickle
            self.cache.save(self.rates, self.last_timestamp)

            # Also store to Excel and DB
            self.repository.save_rates_to_excel(self.rates, self.last_timestamp)
            self.repository.save_rates_to_database(self.rates, self.last_timestamp)

        except Exception as e:
            # Fallback to stale data
            if not self.rates:
                # We have no fallback data -> raise
                raise RuntimeError(
                    f"FX update failed and no cached data is available. Error: {str(e)}"
                )
            else:
                # We have some stale data
                stale_date_str = self.last_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[WARN] FX fetch failed; using stale data from {stale_date_str}. Error: {str(e)}")

    def _fetch_full_matrix(self, base: str) -> Dict[Tuple[str, str], float]:
        """
        Calls the provider for the base->others rates, then constructs cross rates
        for all G8_CURRENCIES. Also includes 'GBp' logic.
        """
        # Step 1: Fetch base->others with provider
        # Exclude 'base' itself and 'GBp' from direct quotes
        quote_ccys = [c for c in self.G8_CURRENCIES if c not in [base, "GBp"]]
        direct_rates = self.provider.fetch_rates(base, quote_ccys)

        # Step 2: Build out cross pairs for all G8 combos
        full_matrix = dict(direct_rates)

        # Identity pairs
        for c in self.G8_CURRENCIES:
            full_matrix[(c, c)] = 1.0

        # Cross-multiply pairs via base
        all_ccys = set(self.G8_CURRENCIES)
        for x in all_ccys:
            for y in all_ccys:
                if x == y:
                    continue
                if (x, base) in full_matrix and (base, y) in full_matrix:
                    full_matrix[(x, y)] = full_matrix[(x, base)] * full_matrix[(base, y)]

        # Step 3: Insert GBp logic
        # GBP -> GBp = 100.0 and vice versa
        full_matrix[("GBP", "GBp")] = 100.0
        full_matrix[("GBp", "GBP")] = 0.01

        # For every other currency c != "GBp"
        for c in all_ccys:
            if c == "GBp":
                continue
            if ("GBP", c) in full_matrix:
                full_matrix[("GBp", c)] = full_matrix[("GBP", c)] * 0.01
            if (c, "GBP") in full_matrix:
                full_matrix[(c, "GBp")] = full_matrix[(c, "GBP")] * 100.0

        return full_matrix

    def get_rate(self, from_ccy: str, to_ccy: str, base_ccy_for_fetch: str = "USD") -> float:
        """
        Public method for obtaining an FX rate. Ensures data is fresh, then returns the rate.
        :param from_ccy: e.g. "USD", "EUR", "GBp"
        :param to_ccy:   e.g. "GBP", "HKD", "GBp"
        :param base_ccy_for_fetch: which currency do we use as the reference for fetching from the API?
        :return: float representing the cross rate from_ccy -> to_ccy
        """
        # Ensure we have fresh rates
        if not self.rates:
            self.ensure_fresh_data(base=base_ccy_for_fetch)

        # If the pair doesn't exist or data might be stale, re-check
        if (from_ccy, to_ccy) not in self.rates:
            self.ensure_fresh_data(base=base_ccy_for_fetch)

        # Return the rate or 0.0 if not found
        return self.rates.get((from_ccy, to_ccy), 0.0)

    def fetch_and_store_all(self, base: str = "USD"):
        """
        CLI-like method to do a one-off fetch & store of all cross rates.
        """
        self.ensure_fresh_data(base)
        print(f"FX data updated and stored. Timestamp: {self.last_timestamp}")


def setup_fx_data_directory(base_dir: str = "data/fx") -> None:
    """Create the data directory if it doesn't exist"""
    Path(base_dir).mkdir(parents=True, exist_ok=True)


def main():
    # Create data directory
    data_dir = "data/fx"
    setup_fx_data_directory(data_dir)

    # Define paths for outputs
    pickle_path = os.path.join(data_dir, "fx_rates.pkl")
    excel_path = os.path.join(data_dir, "fx_rates.xlsx")
    db_path = os.path.join(data_dir, "fx_rates.db")

    # Initialize components
    provider = ExchangeRateHostProvider()  # Will use ERHOST_API_KEY from environment
    cache = FxCache(pickle_path)
    repository = FxRepository(excel_path, db_path)

    # Create FX manager with 48-hour stale threshold
    fx_manager = FxManager(
        provider=provider,
        cache=cache,
        repository=repository,
        stale_threshold_hours=48
    )

    # Fetch and store all rates using USD as base currency
    print(f"Fetching and storing FX rates...")
    print(f"Pickle will be saved to: {pickle_path}")
    print(f"Excel file will be saved to: {excel_path}")
    print(f"Database will be saved to: {db_path}")

    fx_manager.fetch_and_store_all(base="USD")

    # Print some sample rates to verify
    test_pairs = [
        ("USD", "EUR"),
        ("GBP", "USD"),
        ("EUR", "JPY"),
        ("GBP", "GBp")
    ]

    print("\nSample rates:")
    print("-" * 30)
    for base, quote in test_pairs:
        rate = fx_manager.get_rate(base, quote)
        print(f"1 {base} = {rate:.4f} {quote}")


if __name__ == "__main__":
    main()


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/utils/__init__.py


# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/utils/backoff.py
from abc import ABC, abstractmethod
import time
from typing import Optional, Callable, TypeVar, Any
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generic type for function return
T = TypeVar('T')


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies.

    This class defines the interface for implementing different backoff strategies
    for handling API rate limits and transient failures.
    """

    @abstractmethod
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute the function with retry logic.

        Args:
            func: The function to execute with retries
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function execution

        Raises:
            Exception: The last exception encountered after all retries
        """
        pass


class ExponentialBackoff(BackoffStrategy):
    """Implements exponential backoff with maximum retries.

    This implementation increases the delay between retries exponentially,
    with a maximum delay cap.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential calculation
    """

    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        """Initialize the exponential backoff strategy.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential calculation
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def calculate_delay(self, attempt: int) -> float:
        """Calculate the delay for the current attempt.

        Args:
            attempt: The current attempt number (0-based)

        Returns:
            The calculated delay in seconds
        """
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        return delay

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute the function with exponential backoff retry logic.

        Args:
            func: The function to execute with retries
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function execution

        Raises:
            Exception: The last exception encountered after all retries
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed. "
                        f"Last error: {str(e)}"
                    )

        if last_exception:
            raise last_exception


def with_backoff(strategy: Optional[BackoffStrategy] = None) -> Callable:
    """Decorator to apply backoff strategy to a function.

    Args:
        strategy: The backoff strategy to use. If None, uses default ExponentialBackoff

    Returns:
        Decorated function with backoff strategy applied
    """
    if strategy is None:
        strategy = ExponentialBackoff()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return strategy.execute(func, *args, **kwargs)

        return wrapper

    return decorator

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/utils/stock_provider.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import yfinance as yf
from datetime import datetime
import pandas as pd
import logging
import time
from .backoff import BackoffStrategy, ExponentialBackoff, with_backoff

logger = logging.getLogger(__name__)


class StockDataProvider(ABC):
    """Abstract base class for stock data providers."""

    @abstractmethod
    def get_info(self, symbol: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_market_cap(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_stock_price(self, symbol: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_business_summary(self, symbol: str) -> str:
        pass


class YFinanceProvider(StockDataProvider):
    """Yahoo Finance implementation with rate limiting & batch functions."""

    def __init__(
            self,
            backoff_strategy: Optional[BackoffStrategy] = None,
            batch_size: int = 10,
            rate_limit_pause: float = 2.0
    ):
        self.backoff = backoff_strategy or ExponentialBackoff()
        self.batch_size = batch_size
        self.rate_limit_pause = rate_limit_pause
        self._cache: Dict[str, Dict[str, Any]] = {}

    @with_backoff()
    def _fetch_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch the ticker info from yfinance."""
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Log raw sector/industry data for debugging
        # logger.debug(f"YFinance raw data for {symbol}:")
        # logger.debug(f"  sector: {info.get('sector')}")
        # logger.debug(f"  industry: {info.get('industry')}")

        return info


    def _get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Get ticker info with caching and rate limiting."""
        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_info(symbol)
            time.sleep(self.rate_limit_pause)  # Rate-limiting pause
        return self._cache[symbol]

    def get_sector_data(self, symbol: str) -> Dict[str, str]:
        """Get sector and industry data for a symbol."""
        info = self.get_info(symbol)
        sector = info.get('sector', '')
        industry = info.get('industry', '')

        # logger.info(f"[StockProvider] Raw YFinance data for {symbol}:")
        # logger.info(f"  Full info keys: {list(info.keys())}")
        # logger.info(f"  Sector: '{sector}'")
        # logger.info(f"  Industry: '{industry}'")

        return {
            'sector': sector,
            'industry': industry
        }

    def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get all available information for a stock."""
        try:
            return self._get_ticker_info(symbol)
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """Get market capitalization for a stock."""
        try:
            info = self._get_ticker_info(symbol)
            return info.get('marketCap')
        except Exception as e:
            logger.error(f"Error fetching market cap for {symbol}: {str(e)}")
            return None

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Return a single 'price' for the symbol.
        Uses 'bid' or 'currentPrice' or 'previousClose' if necessary.
        """
        try:
            info = self._get_ticker_info(symbol)
            price = info.get('bid')
            if price is None:
                price = info.get('currentPrice')
                if price is None:
                    # logger.warning(
                    #     f"No 'currentPrice' found for {symbol}, "
                    #     f"trying fallback keys..."
                    # )
                    price = info.get('regularMarketPreviousClose') or info.get('previousClose')
            if price is None:
                logger.warning(
                    f"No price data found for {symbol}. Info keys: {list(info.keys())}"
                )
            return price
        except Exception as e:
            logger.error(f"Error fetching stock price for {symbol}: {str(e)}")
            return None

    def get_business_summary(self, symbol: str) -> str:
        """Get the business summary for a stock."""
        try:
            info = self._get_ticker_info(symbol)
            return info.get('longBusinessSummary', '')
        except Exception as e:
            logger.error(f"Error fetching summary for {symbol}: {str(e)}")
            return ""

    def get_historical_data(
            self,
            symbol: str,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.
        Returns DataFrame with OHLCV data or empty DataFrame if error.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start,
                end=end,
                interval=interval
            )
            if data.empty:
                logger.warning(f"No historical data for {symbol} "
                               f"({start} to {end}, interval={interval})")
            return data
        except Exception as e:
            logger.error(
                f"Error fetching historical data for {symbol}: {str(e)}"
            )
            return pd.DataFrame()

    # Add or update this method in the YFinanceProvider class
    def get_historical_data_multiple(self, symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        data = yf.download(symbols, start=start, end=end)

        # If we have a MultiIndex DataFrame, prioritize 'Adj Close', fall back to 'Close'
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.levels[0]:
                price_data = data['Adj Close']
            elif 'Close' in data.columns.levels[0]:
                price_data = data['Close']
                logging.warning(
                    "Using 'Close' prices instead of 'Adj Close'. Results may be affected by stock splits and dividends.")
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' columns found in the downloaded data")
        else:
            # If it's not a MultiIndex, assume it's a single ticker and try to get 'Adj Close' or 'Close'
            if 'Adj Close' in data.columns:
                price_data = data['Adj Close']
            elif 'Close' in data.columns:
                price_data = data['Close']
                logging.warning(
                    "Using 'Close' prices instead of 'Adj Close'. Results may be affected by stock splits and dividends.")
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' columns found in the downloaded data")

        # Forward fill and drop any remaining NaN values
        return price_data.ffill().dropna()

    def get_historical_data_multiple_old(
            self,
            symbols: List[str],
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            auto_adjust: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for multiple tickers.
        Returns DataFrame with columns = ticker symbols.
        """
        # logger.info(
        #     f"Fetching multiple tickers: {symbols}, "
        #     f"from {start} to {end}, auto_adjust={auto_adjust}"
        # )

        if not symbols:
            return pd.DataFrame()

        df_all = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            progress=False,
            group_by="column",
            auto_adjust=auto_adjust
        )

        if df_all.empty:
            logger.warning("No data returned for requested symbols.")
            return pd.DataFrame()

        # Handle multi-index case
        if isinstance(df_all.columns, pd.MultiIndex):
            close_col = "Adj Close" if auto_adjust else "Close"
            if close_col in df_all.columns.levels[0]:
                df_close = df_all.xs(close_col, level=0, axis=1)
            else:
                logger.warning(f"'{close_col}' not found. Using 'Close'")
                df_close = df_all.xs("Close", level=0, axis=1)
        else:
            # Single-level columns
            df_close = df_all["Adj Close" if auto_adjust else "Close"]

        if isinstance(df_close, pd.Series):
            df_close = df_close.to_frame()

        df_close.ffill(inplace=True)
        return df_close


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    provider = YFinanceProvider()

    # Test sector data extraction
    test_symbols = ["AAPL", "MSFT", "JPM", "XOM", "GOOGL"]

    print("\nTesting sector data extraction:")
    for symbol in test_symbols:
        sector_data = provider.get_sector_data(symbol)
        print(f"\n{symbol}:")
        print(f"  Sector: {sector_data['sector']}")
        print(f"  Industry: {sector_data['industry']}")

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/utils/index_processor.py
import os
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict, Any
from .stock_provider import YFinanceProvider

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexComponentProcessor:
    """Processes index component files and fetches stock data using YFinance."""

    def __init__(self):
        self.provider = YFinanceProvider()

        # Define directory paths
        self.input_dir = Path("utils/data/index_components")
        self.output_dir = Path("utils/data/index/data")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_input_files(self) -> List[Path]:
        """Get all Excel files in the input directory."""
        return list(self.input_dir.glob("*.xlsx"))

    def read_tickers(self, file_path: Path) -> List[str]:
        """Read tickers from Column A of an Excel file."""
        try:
            df = pd.read_excel(file_path, usecols=[0])  # Read only first column
            tickers = df.iloc[:, 0].astype(str).tolist()
            # Clean tickers (remove any whitespace and convert to uppercase)
            tickers = [ticker.strip().upper() for ticker in tickers if isinstance(ticker, str) and ticker.strip()]
            # logger.info(f"Read {len(tickers)} tickers from {file_path.name}")
            return tickers
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return []

    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock data for a single ticker."""
        try:
            info = self.provider.get_info(ticker)
            sector_data = self.provider.get_sector_data(ticker)
            market_cap = self.provider.get_market_cap(ticker)

            return {
                'stock': ticker,
                'currency': info.get('currency', 'N/A'),
                'sector': sector_data.get('sector', 'N/A'),
                'industry': sector_data.get('industry', 'N/A'),
                'marketCap': market_cap if market_cap is not None else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return {
                'stock': ticker,
                'currency': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'marketCap': 'N/A'
            }

    def process_file(self, file_path: Path) -> None:
        """Process a single index component file."""
        try:
            # logger.info(f"Processing {file_path.name}")

            # Read tickers
            tickers = self.read_tickers(file_path)
            if not tickers:
                logger.warning(f"No tickers found in {file_path.name}")
                return

            # Fetch data for each ticker
            stock_data = []
            for ticker in tickers:
                # logger.info(f"Fetching data for {ticker}")
                data = self.get_stock_data(ticker)
                stock_data.append(data)

            # Create DataFrame and save to Excel
            df = pd.DataFrame(stock_data)

            # Generate output filename
            output_filename = file_path.stem + "_components.xlsx"
            output_path = self.output_dir / output_filename

            # Save to Excel
            df.to_excel(output_path, index=False)
            # logger.info(f"Saved data to {output_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")

    def process_all_files(self) -> None:
        """Process all index component files in the input directory."""
        input_files = self.get_input_files()

        if not input_files:
            logger.warning("No Excel files found in input directory")
            return

        # logger.info(f"Found {len(input_files)} files to process")

        for file_path in input_files:
            self.process_file(file_path)


def main():
    processor = IndexComponentProcessor()
    processor.process_all_files()


if __name__ == "__main__":
    main()

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/scratch_and_test/basic_yf_test.py
import yfinance as yf
bg = yf.Ticker('0P00000VC9.L').funds_data

print(bg.description)
print(bg.top_holdings)
print(bg.sector_weightings)

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/scratch_and_test/fund_test.py
from app.models.portfolio import Portfolio
from app.models.stock import Stock
from utils.fx_manager import FxManager, FxCache, FxRepository, ExchangeRateHostProvider

def main():
    # Initialize FX manager
    provider = ExchangeRateHostProvider()
    cache = FxCache("data/fx/fx_rates.pkl")  # Adjust path if needed
    repo = FxRepository("data/fx/fx_rates.xlsx", "data/fx/fx_rates.db")  # Adjust paths if needed
    fx_manager = FxManager(provider, cache, repo, stale_threshold_hours=48)
    fx_manager.ensure_fresh_data(base="GBP")  # Initialize with GBP as base

    # Create portfolio
    portfolio = Portfolio(base_currency="GBP", fx_manager=fx_manager)
    portfolio.add_position(Stock(ticker="NVDA", quantity=10))
    portfolio.enrich_positions()

    # Get valuations in different currencies
    gbp_value = portfolio.get_total_value()
    portfolio.base_currency = "EUR"
    fx_manager.ensure_fresh_data(base="EUR")
    eur_value = portfolio.get_total_value()
    portfolio.base_currency = "USD"
    fx_manager.ensure_fresh_data(base="USD")
    usd_value = portfolio.get_total_value()

    # Print results
    print(f"Portfolio value in GBP: {gbp_value:.2f}")
    print(f"Portfolio value in EUR: {eur_value:.2f}")
    print(f"Portfolio value in USD: {usd_value:.2f}")

if __name__ == "__main__":
    main()

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/scratch_and_test/multi_test.py
from app.models.portfolio import Portfolio
from app.models.stock import Stock
from utils.fx_manager import FxManager, FxCache, FxRepository, ExchangeRateHostProvider

def main():
    # Initialize FX manager
    provider = ExchangeRateHostProvider()
    cache = FxCache("data/fx/fx_rates.pkl")  # Adjust path if needed
    repo = FxRepository("data/fx/fx_rates.xlsx", "data/fx/fx_rates.db")  # Adjust paths if needed
    fx_manager = FxManager(provider, cache, repo, stale_threshold_hours=48)
    fx_manager.ensure_fresh_data(base="GBP")  # Initialize with GBP as base

    # Create portfolio
    portfolio = Portfolio(base_currency="GBP", fx_manager=fx_manager)
    portfolio.add_position(Stock(ticker="0P00000VC9.L", quantity=474.217))

    # Enrich portfolio
    portfolio.enrich_positions()

    # Output position details
    for ticker, stock in portfolio.positions.items():
        print(f"----- {ticker} -----")
        print(f"Name: {stock.name}")
        print(f"Quantity: {stock.quantity}")
        print(f"Currency: {stock.currency}")
        print(f"Type: {stock.type}")
        print(f"Sector: {stock.sector}")
        print(f"Sub-Industry: {stock.sub_industry}")
        print(f"Price: {stock.price}")
        print("-" * 20)

    # Get valuations in different currencies
    gbp_value = portfolio.get_total_value()
    portfolio.base_currency = "EUR"
    fx_manager.ensure_fresh_data(base="EUR")
    eur_value = portfolio.get_total_value()
    portfolio.base_currency = "USD"
    fx_manager.ensure_fresh_data(base="USD")
    usd_value = portfolio.get_total_value()

    # Print results
    print(f"Portfolio value in GBP: {gbp_value:.2f}")
    print(f"Portfolio value in EUR: {eur_value:.2f}")
    print(f"Portfolio value in USD: {usd_value:.2f}")

if __name__ == "__main__":
    main()

# File: /Users/richardwalker/PycharmProjects/TBIC_Jubilee/scratch_and_test/get_ind_codes.py
#!/usr/bin/env python3
"""
Fetch sector and industry codes via yfinance for a list of tickers.
"""
import pandas as pd
import yfinance as yf
import time

def main():

    df = pd.read_excel("data/sp_501.xlsx")
    tickers = df["stock"].tolist()
    print(tickers)
    for symbol in tickers:
        ticker_obj = yf.Ticker(symbol)
        time.sleep(0.2)
        try:
            info = ticker_obj.info  # This returns a dict of metadata from Yahoo
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            print(f"{symbol} \t {sector} \t {industry}")
        except Exception as e:
            print(e)



if __name__ == "__main__":
    main()


