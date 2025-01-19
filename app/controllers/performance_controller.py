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