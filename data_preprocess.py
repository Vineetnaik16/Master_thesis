# data_preprocess.py

import pandas as pd
import numpy as np

# Load CSV file saved by data_scraper.py
df = pd.read_csv('spy_raw_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 1: Calculate log returns
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

# Step 2: Calculate rolling volatility (20-day)
df['volatility'] = df['log_return'].rolling(window=20).std()

# Step 3: Calculate drawdown
df['cumulative_max'] = df['Close'].cummax()
df['drawdown'] = (df['Close'] - df['cumulative_max']) / df['cumulative_max']

# Step 4: Moving average distance
df['ma_50'] = df['Close'].rolling(window=50).mean()
df['ma_distance'] = (df['Close'] - df['ma_50']) / df['ma_50']

# Step 5: Trend slope and strength
df['trend_slope'] = df['Close'].rolling(20).apply(
    lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
)
df['trend_strength'] = df['trend_slope'].abs()

# Step 6: Bollinger Band width
df['boll_width'] = 4 * df['Close'].rolling(20).std() / df['Close']

# Final cleanup
df.drop(columns=['cumulative_max'], inplace=True)
df.dropna(inplace=True)

# Export processed features if needed
df.to_csv('spy_processed_data.csv')

# Preview
print("✅ Feature engineering completed. Here's a preview:")
print(df.head())

# Feature matrix for modeling
X = df[['log_return', 'volatility', 'drawdown', 'ma_distance']].values

