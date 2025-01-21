import logging
import pandas as pd
from typing import List, Dict
from app.models.portfolio import Portfolio
from utils.stock_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class PeerAnalyzer:
    def __init__(self):
        self.stock_provider = YFinanceProvider()

    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """Clean a symbol by removing spaces and special characters."""
        return symbol.strip().replace(" ", "")

    def get_peer_metrics(
            self,
            chosen_symbol: str,
            peer_symbols: List[str],
            metrics: List[str],
            portfolio: Portfolio
    ) -> pd.DataFrame:
        if not peer_symbols or not metrics:
            logger.warning("No peer symbols or no metrics to retrieve.")
            return pd.DataFrame()

        # Clean input symbols
        chosen_symbol = self.clean_symbol(chosen_symbol)
        cleaned_peer_symbols = [self.clean_symbol(sym) for sym in peer_symbols]

        logger.info(f"Getting metrics for peers: {cleaned_peer_symbols}")

        records = []
        for sym in cleaned_peer_symbols:
            try:
                info = self.stock_provider.get_info(sym)
                if not info:
                    logger.warning(f"No info returned for symbol {sym}")
                    continue

                row_data = {
                    "symbol": sym,
                    "name": info.get("shortName") or info.get("longName") or sym
                }

                # Add each requested metric if present in info
                for m in metrics:
                    value = info.get(m)
                    logger.debug(f"Metric {m} for {sym}: {value}")
                    row_data[m] = value

                records.append(row_data)

            except Exception as e:
                logger.error(f"Error processing {sym}: {str(e)}")
                continue

        df_peers = pd.DataFrame(records)

        if df_peers.empty:
            logger.warning("No peer data collected")
            return df_peers

        logger.info(f"Collected data for {len(df_peers)} peers")
        return df_peers