# streamlit_app.py
import os
from io import BytesIO
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Market Data Scraper", page_icon="📈", layout="wide")
st.title(" Market Data Scraper")
st.caption(f"Working dir: `{os.getcwd()}`")

# --------------------- Grouped tickers ---------------------
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

# --------------------- Sidebar ---------------------
st.sidebar.header("Configuration")

if "cat" not in st.session_state:
    st.session_state.cat = "US Index ETFs"

cat = st.sidebar.selectbox("Category", list(CATS.keys()), index=list(CATS.keys()).index(st.session_state.cat))
st.session_state.cat = cat

choice_label = st.sidebar.selectbox("Select ticker", list(CATS[cat].keys()), index=0)
selected_ticker = CATS[cat][choice_label]

if cat == "Custom":
    user_ticker = st.sidebar.text_input("Custom ticker (e.g., MSFT, ^GSPC, BTC-USD)", value=selected_ticker)
    ticker = (user_ticker or "SPY").strip().upper()
else:
    ticker = selected_ticker

start_date = st.sidebar.date_input("Start date", pd.to_datetime("1990-01-01"))
end_date = st.sidebar.date_input("End date", pd.Timestamp.today())
fetch = st.sidebar.button("Fetch Data")

# --------------------- Helpers ---------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tkr: str, start, end) -> pd.DataFrame:
    df = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=False)
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # Dynamically find columns in case ticker symbol is appended
    close_col = [c for c in df.columns if c.lower().startswith("close") and not c.lower().startswith("adj")][0]
    adj_col = [c for c in df.columns if c.lower().startswith("adj close")][0]
    out = df.reset_index()[["Date", close_col, adj_col]]
    out.columns = ["Date", "Close", "Adj Close"]
    return out

# --------------------- Flow ---------------------
if not fetch:
    st.info("Pick a **category** and **ticker** in the sidebar, set dates, then click **Fetch Data**.")
    st.stop()

with st.spinner(f"Downloading {ticker}…"):
    try:
        raw = fetch_prices(ticker, start_date, end_date)
    except Exception as e:
        st.error("Download failed.")
        st.exception(e)
        st.stop()

if raw is None or raw.empty:
    st.warning("No data returned. Check ticker or date range.")
    st.stop()

data = prep_df(raw)

# Currency info
try:
    finfo = yf.Ticker(ticker).fast_info
    currency = getattr(finfo, "currency", None) or finfo.get("currency", "USD")
except Exception:
    currency = "USD"

st.subheader(f"{ticker} — Close Prices (Raw & Adjusted) in US $")
st.caption(f"Currency: **{currency}** • Rows: **{len(data)}**")

# Table
st.dataframe(data, use_container_width=True, height=420)

# Chart with both series
chart_data = data.set_index("Date")[["Close", "Adj Close"]]
st.line_chart(chart_data)

# Download CSV
buf = BytesIO()
data.to_csv(buf, index=False)
st.download_button(
    "📥 Download CSV",
    data=buf.getvalue(),
    file_name=f"{ticker}_close_adjclose.csv",
    mime="text/csv"
)

st.success("✅ Done")
