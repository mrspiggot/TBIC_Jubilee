import yfinance as yf
bg = yf.Ticker('0P00000VC9.L').funds_data

print(bg.description)
print(bg.top_holdings)
print(bg.sector_weightings)