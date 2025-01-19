from typing import List, Dict, Optional
import pandas as pd
from utils.fx_manager import FxManager
from utils.stock_provider import YFinanceProvider
from .stock import Stock


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