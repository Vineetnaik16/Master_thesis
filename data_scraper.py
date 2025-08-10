# data_scraper.py

import pandas as pd
import yfinance as yf

# Set trading instrument
trading_instrument = 'SPY'
today = pd.Timestamp.today().strftime('%Y-%m-%d')

# Download data
data = yf.download(trading_instrument, start="1990-01-01", end=today)
data = data.reset_index()[['Date', 'Close']]
data.columns = ['Date', 'Close']

# Save to CSV
data.to_csv('spy_raw_data.csv', index=False)
print("✅ Data saved to spy_raw_data.csv")
