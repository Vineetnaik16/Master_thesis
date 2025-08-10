# clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.cluster import AgglomerativeClustering

# --- Load preprocessed data ---
df = pd.read_csv('spy_processed_data.csv', parse_dates=['Date'], index_col='Date')

# --- Feature Matrix ---
X = df[['log_return', 'volatility', 'drawdown', 'ma_distance']].values

# --- Clustering: Agglomerative ---
model = AgglomerativeClustering(n_clusters=2)
df['state'] = model.fit_predict(X)

# --- Assign Regime Names ---
regime_stats = df.groupby('state')['log_return'].agg(['mean', 'std'])
print("📊 Regime stats:\n", regime_stats)

sorted_states = regime_stats['mean'].sort_values()
regime_names = {
    sorted_states.index[0]: 'Bear',
    sorted_states.index[1]: 'Bull'
}
df['Clustering_regime_label'] = df['state'].map(regime_names)

# --- Plot: Market Regime Timeline ---
colors = {'Bull': 'green', 'Bear': 'red'}
plt.figure(figsize=(14, 1))
plt.scatter(df.index, [1]*len(df),
            c=df['Clustering_regime_label'].map(colors),
            marker='|', s=100)
plt.title('Market Regime Timeline (Clustering Output)')
plt.yticks([])
plt.tight_layout()
plt.show()

# --- Plot: SPY Close Price with Color-coded Regimes ---
colors = df['Clustering_regime_label'].map({'Bull': 'green', 'Bear': 'red'})
fig, ax = plt.subplots(figsize=(14, 6))

for i in range(1, len(df)):
    ax.plot(df.index[i-1:i+1], df['Close'].iloc[i-1:i+1], color=colors.iloc[i], linewidth=1.2)

ax.set_title('SPY Close Price with Regime Clustering (Bull/Bear)', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.grid(True)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()

# Save updated file with regime label
df.to_csv('spy_processed_data.csv')
print("✅ Clustering regime label added and saved to spy_processed_data.csv")
