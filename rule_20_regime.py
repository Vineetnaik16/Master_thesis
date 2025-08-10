# rule_20_regime.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Load processed data ---
df = pd.read_csv('spy_processed_data.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# --- Calculate peak and drawdown ---
df['peak'] = df['Close'].cummax()
df['drawdown_from_peak'] = (df['Close'] - df['peak']) / df['peak']

# --- Apply 20% rule ---
df['rule_20_regime'] = df['drawdown_from_peak'].apply(lambda x: 'Bear' if x <= -0.20 else 'Bull')

# --- Plot SPY Close with 20% Rule Regimes ---
colors = df['rule_20_regime'].map({'Bull': 'green', 'Bear': 'red'})

fig, ax = plt.subplots(figsize=(14, 6))
for i in range(1, len(df)):
    ax.plot(df.index[i-1:i+1], df['Close'].iloc[i-1:i+1], color=colors.iloc[i], linewidth=1.2)

ax.set_title('SPY Close Price with 20% Rule Regime Classification', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.grid(True)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()

# --- Save rule_20_regime to main processed file ---
df_existing = pd.read_csv('spy_processed_data.csv', parse_dates=['Date'])
df_existing.set_index('Date', inplace=True)
df_existing['rule_20_regime'] = df['rule_20_regime']

df_existing.to_csv('spy_processed_data.csv')
print("✅ 20% rule regime label saved to spy_processed_data.csv")
