# pages/05_Portfolio_Builder.py
import os
from typing import Dict, Any
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import yfinance as yf

st.set_page_config(page_title="Build Portfolio", page_icon="🧺", layout="wide")
st.title("🧺 Build Your Portfolio")

# --- session store ---
cart: Dict[str, Any] = st.session_state.get("portfolio_cart", {})
store: Dict[str, Any] = st.session_state.get("portfolio_store", {})  # cache per-ticker details

# --- helpers ---
@st.cache_data(show_spinner=False)
def dl_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    end_plus = pd.to_datetime(end) + timedelta(days=1)
    df = yf.download(ticker, start=start, end=end_plus, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns.values]
    return df.reset_index()

def pick_price_col_robust(df: pd.DataFrame):
    """
    Return (raw_col_name, friendly_label).
    Prefers 'Adj Close'; then 'Close' (including 'Close*' / 'Close BTC-USD'); then falls back to 'Open'.
    """
    cols = [str(c) for c in df.columns]
    norm = {c: "".join(str(c).lower().split()) for c in cols}

    adj = next((c for c, n in norm.items() if ("adj" in n and "close" in n) or n == "adjclose"), None)
    if adj: return adj, "Adj Close"

    close_exact = next((c for c in cols if str(c).strip().lower() == "close"), None)
    if close_exact: return close_exact, "Close"

    close_like = next((c for c in cols if "close" in str(c).lower()), None)
    if close_like: return close_like, "Close"

    open_like = next((c for c in cols if str(c).lower().startswith("open")), None)
    if open_like: return open_like, "Close"

    return None, None

def get_currency(tkr: str) -> str:
    try:
        finfo = yf.Ticker(tkr).fast_info
        curr = getattr(finfo, "currency", None) or (finfo.get("currency", None) if isinstance(finfo, dict) else None)
        return str(curr) if curr else "USD"
    except Exception:
        return "USD"

def compute_snapshot(df: pd.DataFrame, price_col: str) -> Dict[str, Any]:
    d = df.copy().dropna(subset=[price_col]).sort_values("Date")
    if d.empty:
        return {"price": np.nan, "trend30": np.nan, "regime": "—"}
    last = float(d[price_col].iloc[-1])
    if len(d) > 30:
        trend30 = float((d[price_col].iloc[-1] / d[price_col].iloc[-31] - 1) * 100.0)
    else:
        trend30 = float(d[price_col].pct_change().fillna(0).tail(30).add(1).prod() - 1) * 100.0
    peak = d[price_col].cummax()
    draw = d[price_col] / peak - 1.0
    regime = "Bear" if float(draw.iloc[-1]) <= -0.20 else "Bull"
    return {"price": last, "trend30": trend30, "regime": regime}

def sparkline_chart(d: pd.DataFrame, price_col: str, title: str = ""):
    # 1Y window, indexed to 100 at start for better visual contrast in small chart
    dd = d.copy()
    dd["Date"] = pd.to_datetime(dd["Date"])
    cutoff = dd["Date"].max() - pd.Timedelta(days=365)
    hist = dd[dd["Date"] >= cutoff]
    if hist.empty:
        hist = dd.tail(365)
    base = float(hist[price_col].iloc[0])
    hist = hist.assign(Index100=(hist[price_col] / base) * 100.0)
    color = "#16a34a" if hist["Index100"].iloc[-1] >= 100.0 else "#dc2626"

    return (
        alt.Chart(hist)
        .mark_line(strokeWidth=2.5, color=color)
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("Index100:Q", title=None, scale=alt.Scale(nice=True),
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            tooltip=[
                "Date:T",
                alt.Tooltip(f"{price_col}:Q", title="Price"),
                alt.Tooltip("Index100:Q", title="Index (start=100)", format=".1f"),
            ],
        )
        .properties(height=90, width="container", title=title or "1Y (start=100)")
    )

# --- sidebar controls ---
with st.sidebar:
    st.header("Add tickers")
    tkr = st.text_input("Symbol (e.g., AAPL, MSFT, BTC-USD, ^GSPC)").strip().upper()
    start = st.date_input("Start date", pd.to_datetime("2018-01-01"))
    end   = st.date_input("End date", pd.Timestamp.today())
    add_disabled = (len(tkr) == 0) or (start >= end)

    if st.button("Add to list ➕", type="primary", use_container_width=True, disabled=add_disabled):
        with st.spinner(f"Fetching {tkr}..."):
            try:
                df = dl_prices(tkr, start, end)
                if df.empty:
                    st.toast(f"No data returned for {tkr}", icon="⚠️")
                else:
                    price_col_raw, price_label = pick_price_col_robust(df)
                    if price_col_raw is None:
                        st.toast(f"Couldn't find a price column for {tkr}. Columns: {list(df.columns)}", icon="⚠️")
                    else:
                        df_use = df[["Date", price_col_raw]].rename(columns={price_col_raw: price_label})
                        snap = compute_snapshot(df_use, price_label)
                        currency = get_currency(tkr)
                        store[tkr] = {
                            "df": df_use,
                            "price_col": price_label,
                            "currency": currency,
                            "snapshot": snap,
                        }
                        st.session_state["portfolio_store"] = store
                        st.toast(f"Added {tkr} to the list", icon="✅")
            except Exception as e:
                st.error(f"Download failed for {tkr}: {e}")

# --- list of fetched tickers (cards with Add to Cart) ---
st.markdown("### Your ticker list")
if not store:
    st.info("Use the sidebar to add symbols. We’ll show a 1-year mini-chart (indexed), price, and trend here.")
else:
    cols = st.columns(2)
    i = 0
    for tkr, rec in list(store.items()):
        col = cols[i % 2]; i += 1
        with col.container(border=True):
            d = rec["df"].copy()
            price_col = rec["price_col"]
            currency = rec["currency"]
            snap = rec["snapshot"]

            st.markdown(f"**{tkr}** · {currency}")
            st.altair_chart(sparkline_chart(d, price_col), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{snap['price']:.2f} {currency}" if np.isfinite(snap["price"]) else "—")
            c2.metric("30d Trend", f"{snap['trend30']:+.2f}%" if np.isfinite(snap["trend30"]) else "—")
            c3.metric("Regime", snap["regime"])

            c4, c5 = st.columns(2)
            if c4.button(f"Add to cart 🛒", key=f"add_{tkr}", use_container_width=True):
                cart[tkr] = {"currency": currency}
                st.session_state["portfolio_cart"] = cart
                st.toast(f"{tkr} added to cart", icon="🛒")
            if c5.button("Remove from list", key=f"rm_{tkr}", use_container_width=True):
                store.pop(tkr, None)
                st.session_state["portfolio_store"] = store
                st.rerun()

st.divider()
left, right = st.columns([1,1])
left.write(f"Cart: **{len(cart)}** tickers")
if right.button("Go to Portfolio →", type="primary", use_container_width=True, disabled=(len(cart)==0)):
    try:
        st.switch_page("pages/06_Portfolio.py")
    except Exception:
        st.rerun()

st.caption("Tip: Add a few symbols (e.g., BTC-USD), then head to Portfolio to see suggestions and weights.")
