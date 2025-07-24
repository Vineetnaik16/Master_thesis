import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import plotly.io as pio

# Optional visualization and modeling imports (can be used in future analysis)
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')
pio.renderers.default = 'notebook'

# --- Parameters ---
TRADING_INSTRUMENT = 'SPY'  # Options: SPY, QQQ, XLK
START_DATE = "1990-01-01"
TODAY = pd.Timestamp.today().strftime('%Y-%m-%d')

# --- Data Download ---
print(f"Downloading {TRADING_INSTRUMENT} data from {START_DATE} to {TODAY}...")
data = yf.download(TRADING_INSTRUMENT, start=START_DATE, end=TODAY)
