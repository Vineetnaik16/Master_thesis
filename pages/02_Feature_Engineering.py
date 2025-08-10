# pages/02_Feature_Engineering.py
import os
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Feature Engineering", page_icon="🧪", layout="wide")
st.title("Feature Engineering")

# -------- get input --------
meta = st.session_state.get("base_meta", {})
df = st.session_state.get("base_df", None)

def _latest_csv(path="data"):
    if not os.path.isdir(path): return None
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
    return max(files, key=os.path.getmtime) if files else None

if df is None:
    path = meta.get("file_path") or _latest_csv("data")
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        meta.setdefault("ticker", os.path.basename(path).split("_")[0])
        # infer currency from filename if present; else keep provided
        parts = os.path.basename(path).split("_")
        if len(parts) >= 4:
            meta.setdefault("currency", parts[3])
        meta.setdefault("currency", meta.get("display_currency", "USD"))

if df is None:
    st.warning("No input data found. Go back and click **Proceed** on the main page.")
    if st.button("↩ Back to main", key="back_missing"):
        for k in ("base_df","base_meta","nav_to_features"):
            st.session_state.pop(k, None)
        try:
            st.switch_page("streamlit_app.py")
        except Exception:
            st.rerun()
    st.stop()

# Use Adj Close only; drop Close if both exist
fe = df.copy().sort_values("Date").reset_index(drop=True)
if "Adj Close" in fe.columns:
    price_col = "Adj Close"
    if "Close" in fe.columns:
        fe = fe.drop(columns=["Close"])
else:
    price_col = "Close"  # fallback if Adj Close absent

st.caption(
    f"Ticker: **{meta.get('ticker','N/A')}** • Display currency: **{meta.get('currency','USD')}** • "
    f"Price column used: **{price_col}**"
)

# -------- features (based on price_col) --------
fe["Return"] = fe[price_col].pct_change()
fe["LogReturn"] = np.log(fe[price_col]).diff()

for w in (5, 10, 20, 50, 100, 200):
    fe[f"SMA_{w}"] = fe[price_col].rolling(w).mean()
for w in (12, 26):
    fe[f"EMA_{w}"] = fe[price_col].ewm(span=w, adjust=False).mean()

fe["MACD"] = fe["EMA_12"] - fe["EMA_26"]
fe["MACD_Signal"] = fe["MACD"].ewm(span=9, adjust=False).mean()
fe["MACD_Hist"] = fe["MACD"] - fe["MACD_Signal"]

fe["Volatility_20"] = fe["Return"].rolling(20).std()
fe["Volatility_60"] = fe["Return"].rolling(60).std()

mid = fe[price_col].rolling(20).mean()
std = fe[price_col].rolling(20).std()
fe["BB_Mid_20"] = mid
fe["BB_Upper_20"] = mid + 2 * std
fe["BB_Lower_20"] = mid - 2 * std

delta = fe[price_col].diff()
up = pd.Series(np.where(delta > 0, delta, 0.0), index=fe.index)
down = pd.Series(np.where(delta < 0, -delta, 0.0), index=fe.index)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / roll_down
fe["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

fe["Drawdown"] = fe[price_col] / fe[price_col].cummax() - 1.0

# -------- show results --------
st.dataframe(fe, use_container_width=True, height=520)

# Price chart with title
chart = (
    alt.Chart(fe)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(f"{price_col}:Q", title=f"{price_col} ({meta.get('currency','USD')})"),
        tooltip=["Date:T", alt.Tooltip(f"{price_col}:Q", title=price_col)],
    )
    .properties(title=f"{meta.get('ticker','TICKER')} — {price_col} ({meta.get('currency','USD')})")
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

# Download engineered dataset
buf = BytesIO(); fe.to_csv(buf, index=False)
st.download_button(
    "📥 Download engineered dataset",
    data=buf.getvalue(),
    file_name=f"{meta.get('ticker','TICKER')}_features_{meta.get('currency','USD')}.csv",
    mime="text/csv",
)

# -------- Navigation buttons --------
c1, c2 = st.columns(2)

with c1:
    if st.button("↩ Back to main", key="back_main"):
        # Clear nav flags so main loads fresh
        for k in ("nav_to_features",):
            st.session_state.pop(k, None)
        try:
            st.switch_page("streamlit_app.py")
        except Exception:
            st.rerun()

with c2:
    if st.button("🔮 Proceed to forecast", type="primary", key="go_forecast"):
        # Keep base_df/base_meta in session so Forecast can use them
        st.session_state["nav_to_forecast"] = True
        try:
            st.switch_page("pages/03_Forecast.py")
        except Exception:
            st.rerun()

st.success("✅ Features created.")
