# pages/02_Feature_Engineering.py
import os
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Regime History and Key Signals", page_icon="🧪", layout="wide")
st.title("Key Signals")

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
        parts = os.path.basename(path).split("_")
        if len(parts) >= 4:
            meta.setdefault("currency", parts[3])
        meta.setdefault("currency", meta.get("display_currency", "USD"))

if df is None or df.empty:
    st.warning("No input data found. Go back and click **Proceed** on the main page.")
    if st.button("↩ Back to main", key="back_missing"):
        for k in ("base_df","base_meta","nav_to_features","nav_to_forecast"):
            st.session_state.pop(k, None)
        try:
            st.switch_page("Home.py")
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

# -------- Regime History (20% rule) --------
st.markdown("### 🧭 Regime history (20% drawdown rule)")

# Compact controls for this section
c1, c2, c3 = st.columns([1,1,3])
with c1:
    threshold_pct = st.number_input("Bear threshold %", min_value=5, max_value=90, value=20, step=1,
                                    help="Bear when drawdown from peak is at or below this amount.")
with c2:
    show_last_n = st.number_input("Show last N days (0=all)", min_value=0, max_value=len(fe), value=0, step=10)

# Compute regime columns (added to fe)
fe["Peak"] = fe[price_col].cummax()
fe["Drawdown_from_peak"] = fe[price_col] / fe["Peak"] - 1.0
thr = -abs(threshold_pct) / 100.0
fe["Regime"] = np.where(fe["Drawdown_from_peak"] <= thr, "Bear", "Bull")

# Build contiguous intervals for shading
chg = (fe["Regime"] != fe["Regime"].shift()).cumsum()
intervals = (
    fe[["Date","Regime"]]
    .assign(grp=chg)
    .groupby("grp")
    .agg(start=("Date","min"), end=("Date","max"), Regime=("Regime","first"))
    .reset_index(drop=True)
)

# Trim if needed
plot_fe = fe if show_last_n == 0 else fe.iloc[-int(show_last_n):]
plot_intervals = intervals.copy()
if show_last_n > 0:
    start_cut = plot_fe["Date"].min()
    end_cut = plot_fe["Date"].max()
    plot_intervals["start"] = plot_intervals["start"].clip(lower=start_cut)
    plot_intervals["end"] = plot_intervals["end"].clip(upper=end_cut)
    plot_intervals = plot_intervals[plot_intervals["start"] <= plot_intervals["end"]]

bg = (
    alt.Chart(plot_intervals)
    .mark_rect(opacity=0.18)
    .encode(
        x=alt.X("start:T", title=""),
        x2="end:T",
        color=alt.Color("Regime:N", scale=alt.Scale(domain=["Bull","Bear"], range=["#2ca02c", "#d62728"]),
                        legend=alt.Legend(title="Regime")),
    )
)

line = (
    alt.Chart(plot_fe)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(f"{price_col}:Q", title=f"{price_col} ({meta.get('currency','USD')})"),
        tooltip=[
            "Date:T",
            alt.Tooltip(f"{price_col}:Q", title="Price"),
            alt.Tooltip("Drawdown_from_peak:Q", format=".1%", title="Drawdown"),
            "Regime:N",
        ],
        color=alt.value("#1f77b4"),
    )
)

title_txt = f"{meta.get('ticker','TICKER')} — Regime history by 20% rule ({threshold_pct}%)"
st.altair_chart((bg + line).properties(title=title_txt).interactive(), use_container_width=True)

# -------- show results --------
st.markdown("### Price & Key Signals")
st.dataframe(
    fe,
    use_container_width=True,
    height=520,
)

# Price chart (plain)
with st.expander("Show price chart"):
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

# Download engineered dataset (includes Regime column)
buf = BytesIO(); fe.to_csv(buf, index=False)
st.download_button(
    "📥 Download engineered dataset (with Regime)",
    data=buf.getvalue(),
    file_name=f"{meta.get('ticker','TICKER')}_features_{meta.get('currency','USD')}.csv",
    mime="text/csv",
)

# -------- Navigation buttons --------
c_left, c_right = st.columns(2)

with c_left:
    if st.button("↩ Back to main", key="back_main"):
        for k in ("nav_to_features","nav_to_forecast"):
            st.session_state.pop(k, None)
        try:
            st.switch_page("Home.py")
        except Exception:
            st.rerun()

with c_right:
    if st.button("🔮 Proceed to forecast", type="primary", key="go_forecast"):
        st.session_state["nav_to_forecast"] = True
        try:
            st.switch_page("pages/04_Forecast.py")
        except Exception:
            st.rerun()

st.success("✅ Features created (Regime column included).")

# --- 📘 Column guide for non-technical users ---
st.markdown("### 📘 What these columns mean")

COL_EXPLAIN = {
    # Price & date
    "Date": "The trading day.",
    "Adj Close": "Closing price adjusted for dividends/splits (best for long-term comparisons).",
    "Close": "Raw closing price (used only if Adj Close isn’t available).",

    # Returns & momentum
    "Return": "Day-to-day % change in price.",
    "LogReturn": "Like Return but on a log scale (handy for math; very similar story).",
    "RSI_14": "Momentum from 0–100; >70 often ‘overbought’, <30 ‘oversold’.",

    # Trend (moving averages)
    "SMA_5": "5-day simple moving average (short-term trend).",
    "SMA_10": "10-day simple moving average.",
    "SMA_20": "20-day simple moving average (about a month).",
    "SMA_50": "50-day simple moving average (medium trend).",
    "SMA_100": "100-day simple moving average.",
    "SMA_200": "200-day simple moving average (long-term trend).",
    "EMA_12": "12-day exponential moving average (recent days matter more).",
    "EMA_26": "26-day exponential moving average.",

    # MACD
    "MACD": "EMA_12 minus EMA_26 (positive = upward momentum).",
    "MACD_Signal": "9-day EMA of MACD (smoother reference).",
    "MACD_Hist": "MACD minus MACD_Signal (above zero = momentum picking up).",

    # Volatility
    "Volatility_20": "How choppy returns were over ~1 month (20 days).",
    "Volatility_60": "How choppy returns were over ~3 months (60 days).",

    # Bollinger Bands
    "BB_Mid_20": "20-day moving average (middle band).",
    "BB_Upper_20": "Middle band + 2× recent volatility (upper band).",
    "BB_Lower_20": "Middle band − 2× recent volatility (lower band).",

    # Drawdowns & regimes
    "Peak": "Highest price reached so far (running record high).",
    "Drawdown_from_peak": "How far today’s price is below that peak (e.g., −0.22 = 22% below high).",
    "Regime": "Bull or Bear using the 20% rule (Bear when drawdown ≤ −20%).",
}

SECTIONS = {
    "Price & date": ["Date","Adj Close","Close"],
    "Returns & momentum": ["Return","LogReturn","RSI_14"],
    "Trend (moving averages)": ["SMA_5","SMA_10","SMA_20","SMA_50","SMA_100","SMA_200","EMA_12","EMA_26"],
    "MACD": ["MACD","MACD_Signal","MACD_Hist"],
    "Volatility": ["Volatility_20","Volatility_60"],
    "Bollinger Bands": ["BB_Mid_20","BB_Upper_20","BB_Lower_20"],
    "Drawdowns & regimes": ["Peak","Drawdown_from_peak","Regime"],
}

present = set(fe.columns)

for title, cols in SECTIONS.items():
    shown = [c for c in cols if c in present]
    if not shown:
        continue
    st.markdown(f"#### {title}")
    bullets = "\n".join([f"- **{c}** — {COL_EXPLAIN[c]}" for c in shown])
    st.markdown(bullets)
