# pages/04_Regime_20_Rule.py
import os
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Regime History", page_icon="🧭", layout="wide")
st.title("🧭 Regime History — as per 20% Rule")

# -------- get input --------
meta = st.session_state.get("base_meta", {})
df = st.session_state.get("base_df", None)

def _latest_csv(path="data"):
    if not os.path.isdir(path): return None
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
    return max(files, key=os.path.getmtime) if files else None

if df is None:
    # Fallback: try latest saved CSV from /data
    path = meta.get("file_path") or _latest_csv("data")
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        meta.setdefault("ticker", os.path.basename(path).split("_")[0])
        parts = os.path.basename(path).split("_")
        if len(parts) >= 4:
            meta.setdefault("currency", parts[3])
        meta.setdefault("currency", meta.get("display_currency", "USD"))

if df is None or df.empty:
    st.warning("No input data found. Go back to **Main**, fetch data, then click **Proceed**.")
    c1, _ = st.columns([1,3])
    if c1.button("↩ Back to main"):
        for k in ("base_df","base_meta","nav_to_features","nav_to_forecast"):
            st.session_state.pop(k, None)
        try:
            st.switch_page("Home.py")
        except Exception:
            st.rerun()
    st.stop()

# -------- choose price column --------
price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
if price_col is None:
    st.error("Neither **Adj Close** nor **Close** exists in the dataset.")
    st.stop()

data = df.copy()[["Date", price_col]].sort_values("Date").reset_index(drop=True)
data["Date"] = pd.to_datetime(data["Date"])
data = data.dropna(subset=[price_col])

ticker = meta.get("ticker", "TICKER")
currency = meta.get("currency", "USD")

# -------- sidebar controls --------
st.sidebar.header("Regime Settings")
threshold_pct = st.sidebar.number_input("Bear threshold (drawdown % from peak)", min_value=5, max_value=90, value=20, step=1)
show_last_n = st.sidebar.number_input("Show last N days (0 = all)", min_value=0, max_value=len(data), value=0, step=10)
save_back = st.sidebar.checkbox("Save regime column back to current CSV (if available)", value=False)

# -------- compute regimes (20% rule) --------
d = data.copy()
d["peak"] = d[price_col].cummax()
d["drawdown"] = d[price_col] / d["peak"] - 1.0
thr = -abs(threshold_pct) / 100.0
d["Regime"] = np.where(d["drawdown"] <= thr, "Bear", "Bull")

# Build contiguous regime intervals for background shading
chg = (d["Regime"] != d["Regime"].shift()).cumsum()
intervals = (
    d.groupby(chg)
     .agg(start=("Date","min"), end=("Date","max"), Regime=("Regime","first"))
     .reset_index(drop=True)
)

# Optionally trim to last N days for plotting
plot_d = d if show_last_n == 0 else d.iloc[-int(show_last_n):]
plot_intervals = intervals.copy()
if show_last_n > 0:
    start_cut = plot_d["Date"].min()
    end_cut = plot_d["Date"].max()
    # Clip intervals to visible window
    plot_intervals["start"] = plot_intervals["start"].clip(lower=start_cut)
    plot_intervals["end"] = plot_intervals["end"].clip(upper=end_cut)
    plot_intervals = plot_intervals[plot_intervals["start"] <= plot_intervals["end"]]

# -------- plot --------
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
    alt.Chart(plot_d)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(f"{price_col}:Q", title=f"Price ({currency})"),
        tooltip=["Date:T", alt.Tooltip(f"{price_col}:Q", title="Price"), "Regime:N", alt.Tooltip("drawdown:Q", format=".1%")],
        color=alt.value("#1f77b4"),
    )
)

title_txt = f"{ticker} — {price_col} with 20% Rule Regimes ({threshold_pct}%)"
st.altair_chart((bg + line).properties(title=title_txt).interactive(), use_container_width=True)

# -------- table + download --------
with st.expander("Show data (with peak, drawdown, regime)"):
    st.dataframe(d[["Date", price_col, "peak", "drawdown", "Regime"]], use_container_width=True, height=420)

enriched = df.merge(d[["Date","Regime"]], on="Date", how="left")
buf = BytesIO()
enriched.to_csv(buf, index=False)
st.download_button("📥 Download dataset with Regime column", data=buf.getvalue(),
                   file_name=f"{ticker}_with_regime_{threshold_pct}pct.csv", mime="text/csv")

# -------- optional: save back to current CSV on disk --------
if save_back:
    curr_path = meta.get("file_path")
    if curr_path and os.path.exists(curr_path):
        try:
            on_disk = pd.read_csv(curr_path, parse_dates=["Date"])
            on_disk = on_disk.merge(d[["Date","Regime"]], on="Date", how="left")
            on_disk.to_csv(curr_path, index=False)
            st.success(f"✅ Regime column saved back to `{curr_path}`")
        except Exception as e:
            st.error(f"Could not save back to file: {e}")
    else:
        st.info("No current CSV path available; use the download button instead.")

# -------- nav buttons --------
c1, c2 = st.columns(2)
with c1:
    if st.button("↩ Back to main"):
        for k in ("nav_to_features","nav_to_forecast"):
            st.session_state.pop(k, None)
        try:
            st.switch_page("Home.py")
        except Exception:
            st.rerun()

with c2:
    if st.button("🔮 Proceed to forecast", type="primary"):
        st.session_state["nav_to_forecast"] = True
        try:
            st.switch_page("pages/04_Forecast.py")
        except Exception:
            st.rerun()
