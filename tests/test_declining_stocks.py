from app.models.analysis.declining_stocks import DecliningStocksAnalyzer
from app.models.portfolio import Portfolio
from app.models.stock import Stock
from utils.fx_manager import FxManager, FxCache, FxRepository, ExchangeRateHostProvider



def test_declining_stocks():
    # Initialize FX manager
    provider = ExchangeRateHostProvider()
    cache = FxCache("utils/data/fx/fx_rates.pkl")
    repo = FxRepository("utils/data/fx/fx_rates.xlsx", "utils/data/fx/fx_rates.db")
    fx_manager = FxManager(provider=provider, cache=cache, repository=repo)
    fx_manager.ensure_fresh_data(base="GBP")

    # Create test portfolio
    portfolio = Portfolio(base_currency="GBP", fx_manager=fx_manager)
    portfolio.add_position(Stock(ticker="MSFT", quantity=10))
    portfolio.add_position(Stock(ticker="AAPL", quantity=15))
    portfolio.enrich_positions()

    # Create analyzer and run analysis
    analyzer = DecliningStocksAnalyzer()
    results = analyzer.analyze_portfolio(portfolio, months=3)

    # Print results
    print("\nAnalysis Results:")
    print(f"Number of declining stocks: {len(results['declining_stocks'])}")
    if results['price_table'] is not None:
        print("\nPrice Table:")
        print(results['price_table'])


if __name__ == "__main__":
    test_declining_stocks()