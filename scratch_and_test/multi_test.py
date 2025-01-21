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