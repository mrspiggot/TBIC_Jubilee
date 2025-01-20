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