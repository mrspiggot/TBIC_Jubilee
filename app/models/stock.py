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
