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
            logger.info(f"Read {len(tickers)} tickers from {file_path.name}")
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
            logger.info(f"Processing {file_path.name}")

            # Read tickers
            tickers = self.read_tickers(file_path)
            if not tickers:
                logger.warning(f"No tickers found in {file_path.name}")
                return

            # Fetch data for each ticker
            stock_data = []
            for ticker in tickers:
                logger.info(f"Fetching data for {ticker}")
                data = self.get_stock_data(ticker)
                stock_data.append(data)

            # Create DataFrame and save to Excel
            df = pd.DataFrame(stock_data)

            # Generate output filename
            output_filename = file_path.stem + "_components.xlsx"
            output_path = self.output_dir / output_filename

            # Save to Excel
            df.to_excel(output_path, index=False)
            logger.info(f"Saved data to {output_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")

    def process_all_files(self) -> None:
        """Process all index component files in the input directory."""
        input_files = self.get_input_files()

        if not input_files:
            logger.warning("No Excel files found in input directory")
            return

        logger.info(f"Found {len(input_files)} files to process")

        for file_path in input_files:
            self.process_file(file_path)


def main():
    processor = IndexComponentProcessor()
    processor.process_all_files()


if __name__ == "__main__":
    main()