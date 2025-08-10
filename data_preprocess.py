# streamlit_app.py
import os
from io import BytesIO
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Market Data Scraper", page_icon="📈", layout="wide")
st.title(" Market Data Scraper")
st.caption(f"Working dir: `{os.getcwd()}`")

# Ensure data dir
os.makedirs("data", exist_ok=True)

# ---------- Ticker groups ----------
CATS = {
    "US Index ETFs": {
        "SPDR S&P 500 (SPY)": "SPY",
        "Invesco QQQ (QQQ)": "QQQ",
        "SPDR Dow Jones (DIA)": "DIA",
        "iShares Russell 2000 (IWM)": "IWM",
    },
    "US Sector ETFs": {
        "Tech (XLK)": "XLK",
        "Financials (XLF)": "XLF",
        "Energy (XLE)": "XLE",
        "Healthcare (XLV)": "XLV",
        "Utilities (XLU)": "XLU",
        "Consumer Discretionary (XLY)": "XLY",
        "Consumer Staples (XLP)": "XLP",
        "Industrials (XLI)": "XLI",
        "Materials (XLB)": "XLB",
        "Real Estate (XLRE)": "XLRE",
    },
    "International ETFs": {
        "Vanguard FTSE Europe (VGK)": "VGK",
        "iShares MSCI Japan (EWJ)": "EWJ",
        "iShares China Large-Cap (FXI)": "FXI",
        "iShares Emerging Markets (EEM)": "EEM",
        "iShares MSCI EAFE (EFA)": "EFA",
        "Vanguard Total World (VT)": "VT",
    },
    "Commodities": {
        "Gold (GLD)": "GLD",
        "Silver (SLV)": "SLV",
        "US Oil (USO)": "USO",
        "Nat Gas (UNG)": "UNG",
        "Copper (CPER)": "CPER",
    },
    "Crypto (Yahoo tickers)": {
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Ethereum (ETH-USD)": "ETH-USD",
        "Solana (SOL-USD)": "SOL-USD",
        "Ripple (XRP-USD)": "XRP-USD",
        "Cardano (ADA-USD)": "ADA-USD",
    },
    "Bonds & Rates": {
        "iShares 20+ Yr Treasury (TLT)": "TLT",
        "iShares 7–10 Yr Treasury (IEF)": "IEF",
        "US Aggregate Bond (AGG)": "AGG",
        "Short-Term Treasury (SHY)": "SHY",
        "High Yield (HYG)": "HYG",
    },
    "Custom": {"(Type your own below)": ""},
}

CURRENCY_SYMBOL = {"USD": "$","EUR":"€","GBP":"£","JPY":"¥","CNY":"¥","INR":"₹","CAD":"$","AUD":"$","CHF":"CHF"}

# ---------- Sidebar ----------
st.sidebar.header("Configuration")
if "cat" not in st.session_state:
    st.session_state.cat = "US Index ETFs"
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = "SPY"

cat = st.sidebar.selectbox("Category", list(CATS.keys()),
                           index=list(CATS.keys()).index(st.session_state.cat))
st.session_state.cat = cat

choice_label = st.sidebar.selectbox("Select ticker", list(CATS[cat].keys()), index=0)
selected_ticker = CATS[cat][choice_label]

if cat == "Custom":
    user_ticker = st.sidebar.text_input("Custom ticker (e.g., MSFT, ^GSPC, BTC-USD)",
                                        value=selected_ticker or st.session_state.last_ticker)
    ticker = (user_ticker or "SPY").strip().upper()
else:
    ticker = selected_ticker

start_date = st.sidebar.date_input("Start date", pd.to_datetime("1990-01-01"))
end_date   = st.sidebar.date_input("End date", pd.Timestamp.today())
fetch = st.sidebar.button("Fetch Data")

# ---------- Helpers ----------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns.values]
    return df

@st.cache_data(show_spinner=False)
def fetch_prices(tkr: str, start, end) -> pd.DataFrame:
    end_plus = pd.to_datetime(end) + timedelta(days=1)  # include end day
    df = yf.download(tkr, start=start, end=end_plus, progress=False, auto_adjust=False)
    return _flatten_columns(df)

def get_currency(tkr: str) -> str:
    try:
        finfo = yf.Ticker(tkr).fast_info
        curr = getattr(finfo, "currency", None) or finfo.get("currency", None)
        return str(curr) if curr else "USD"
    except Exception:
        return "USD"

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return Date, Close[, Adj Close] (Adj Close optional; never fabricated)."""
    if df is None or df.empty:
        return pd.DataFrame()
    dfr = df.reset_index().copy()

    # pick date col
    date_col = "Date" if "Date" in dfr.columns else next(
        (c for c in dfr.columns if pd.api.types.is_datetime64_any_dtype(dfr[c])),
        dfr.columns[0]
    )

    # case/space-agnostic pickers
    def canon(s): return "".join(str(s).lower().split())
    close_col = next((c for c in dfr.columns if canon(c) == "close"), None)
    adj_col   = next((c for c in dfr.columns if canon(c) == "adjclose"), None)
    if close_col is None:
        for c in dfr.columns:
            cc = canon(c)
            if "close" in cc and "adjclose" not in cc:
                close_col = c; break

    if close_col is None and adj_col is None:
        return pd.DataFrame()

    if close_col is None and adj_col is not None:
        out = dfr[[date_col, adj_col]].copy()
        out.columns = ["Date", "Close"]
    else:
        keep = [date_col, close_col] + ([adj_col] if adj_col else [])
        out = dfr[keep].copy()
        out.columns = ["Date", "Close"] + (["Adj Close"] if adj_col else [])

    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)

# ---------- Flow ----------
if not fetch:
    st.info("Pick a **category** and **ticker**, set dates, then click **Fetch Data**.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner(f"Downloading {ticker}…"):
    raw = fetch_prices(ticker, start_date, end_date)

if raw is None or raw.empty:
    st.warning("No data returned. Check ticker or date range.")
    st.stop()

data = prep_df(raw)
if data.empty:
    st.warning("Couldn't find expected Close/Adj Close columns in the download.")
    st.stop()

# Metadata
currency = get_currency(ticker)
curr_symbol = CURRENCY_SYMBOL.get(currency.upper(), currency.upper())
has_adj = "Adj Close" in data.columns

st.session_state.last_ticker = ticker

# ----- Display -----
title = f"{ticker} — Close" + (" & Adjusted Close" if has_adj else "")
st.subheader(title)
st.caption(f"Currency: **{currency}** ({curr_symbol}) • Rows: **{len(data)}**")

st.dataframe(data, use_container_width=True, height=420)
cols_to_plot = ["Close"] + (["Adj Close"] if has_adj else [])
st.line_chart(data.set_index("Date")[cols_to_plot])

buf = BytesIO()
data.to_csv(buf, index=False)
fname = f"{ticker}_{'close_adjclose' if has_adj else 'close'}.csv"
st.download_button("📥 Download CSV", data=buf.getvalue(), file_name=fname, mime="text/csv")

st.success("✅ Finished fetching and rendering data.")

# ---------- NEW: Proceed button ----------
proceed = st.button(f"🚀 Proceed with {ticker}")
if proceed:
    # stash data and meta for the next page
    st.session_state.base_df = data
    st.session_state.base_meta = {
        "ticker": ticker,
        "currency": currency,
        "has_adj": has_adj,
    }
    # navigate to feature engineering page
    try:
        st.switch_page("pages/02_Feature_Engineering.py")
    except Exception:
        st.page_link("pages/02_Feature_Engineering.py", label="Open Feature Engineering ▶", icon="➡️")
