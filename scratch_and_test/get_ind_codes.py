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
