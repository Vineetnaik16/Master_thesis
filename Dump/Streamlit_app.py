# streamlit_app.py
import os
from io import BytesIO
from datetime import timedelta

import pandas as pd
import yfinance as yf
import streamlit as st
import altair as alt

st.set_page_config(page_title="BullVision", page_icon="📈", layout="wide")

# ---------- NAV GUARD: if user just clicked Proceed, jump right away ----------
if st.session_state.get("nav_to_features") and st.session_state.get("base_df") is not None:
    try:
        st.switch_page("pages/02_Feature_Engineering.py")
    except Exception:
        st.page_link("pages/02_Feature_Engineering.py", label="Open Feature Engineering ▶", icon="➡️")
        st.stop()

st.title(" BullVision")
st.caption(f"Working dir: `{os.getcwd()}`")

# Ensure folders
os.makedirs("data", exist_ok=True)

# --------------------- Ticker groups ---------------------
CATS = {
    "US Index ETFs": {
        "SPDR S&P 500 (SPY)": "SPY",
        "Invesco QQQ (QQQ)": "QQQ",
        "SPDR Dow Jones (DIA)": "DIA",
        "iShares Russell 2000 (IWM)": "IWM",
    },
    "US Sector ETFs": {
        "Tech (XLK)": "XLK", "Financials (XLF)": "XLF", "Energy (XLE)": "XLE",
        "Healthcare (XLV)": "XLV", "Utilities (XLU)": "XLU", "Consumer Discretionary (XLY)": "XLY",
        "Consumer Staples (XLP)": "XLP", "Industrials (XLI)": "XLI", "Materials (XLB)": "XLB",
        "Real Estate (XLRE)": "XLRE",
    },
    "International ETFs": {
        "Vanguard FTSE Europe (VGK)": "VGK", "iShares MSCI Japan (EWJ)": "EWJ",
        "iShares China Large-Cap (FXI)": "FXI", "iShares Emerging Markets (EEM)": "EEM",
        "iShares MSCI EAFE (EFA)": "EFA", "Vanguard Total World (VT)": "VT",
    },
    "Commodities": {"Gold (GLD)": "GLD", "Silver (SLV)": "SLV", "US Oil (USO)": "USO", "Nat Gas (UNG)": "UNG", "Copper (CPER)": "CPER"},
    "Crypto (Yahoo tickers)": {
        "Bitcoin (BTC-USD)": "BTC-USD", "Ethereum (ETH-USD)": "ETH-USD", "Solana (SOL-USD)": "SOL-USD",
        "Ripple (XRP-USD)": "XRP-USD", "Cardano (ADA-USD)": "ADA-USD",
    },
    "Bonds & Rates": {"iShares 20+ Yr Treasury (TLT)": "TLT", "iShares 7–10 Yr Treasury (IEF)": "IEF",
                      "US Aggregate Bond (AGG)": "AGG", "Short-Term Treasury (SHY)": "SHY", "High Yield (HYG)": "HYG"},
    "Custom": {"(Type your own below)": ""},
}
CURRENCY_SYMBOL = {"USD":"$","EUR":"€","GBP":"£","JPY":"¥","CNY":"¥","INR":"₹","CAD":"$","AUD":"$","CHF":"CHF"}
DISPLAY_CURRENCIES = ["As reported", "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "INR"]

# --------------------- Sidebar ---------------------
st.sidebar.header("Configuration")
if "cat" not in st.session_state: st.session_state.cat = "US Index ETFs"
if "last_ticker" not in st.session_state: st.session_state.last_ticker = "SPY"

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

display_currency = st.sidebar.selectbox("Display currency", DISPLAY_CURRENCIES, index=0)

start_date = st.sidebar.date_input("Start date", pd.to_datetime("1990-01-01"))
end_date   = st.sidebar.date_input("End date", pd.Timestamp.today())
fetch = st.sidebar.button("Fetch Data")

# --------------------- Helpers ---------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns.values]
    return df

@st.cache_data(show_spinner=False)
def fetch_prices(tkr: str, start, end) -> pd.DataFrame:
    end_plus = pd.to_datetime(end) + timedelta(days=1)  # include 'end' day
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
    """
    Robust: returns Date, Close[, Adj Close] if available.
    Handles 'Close', 'Adj Close', 'Close SPY', 'Adj Close SPY', etc.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    dfr = df.reset_index().copy()

    # Find date column
    date_col = "Date" if "Date" in dfr.columns else next(
        (c for c in dfr.columns if pd.api.types.is_datetime64_any_dtype(dfr[c])),
        dfr.columns[0]
    )

    def is_close(c: str) -> bool:
        s = str(c).lower()
        return s.startswith("close") and not s.startswith("adj close")

    def is_adj(c: str) -> bool:
        return str(c).lower().startswith("adj close")

    close_col = next((c for c in dfr.columns if is_close(c)), None)
    adj_col   = next((c for c in dfr.columns if is_adj(c)), None)

    if close_col is None and adj_col is None:
        return pd.DataFrame()

    if close_col is not None and adj_col is not None:
        out = dfr[[date_col, close_col, adj_col]].copy()
        out.columns = ["Date", "Close", "Adj Close"]
    elif close_col is not None:
        out = dfr[[date_col, close_col]].copy()
        out.columns = ["Date", "Close"]
    else:
        out = dfr[[date_col, adj_col]].copy()
        out.columns = ["Date", "Adj Close"]

    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def _dl_fx(ticker: str, start, end) -> pd.Series:
    """
    Download FX (Yahoo pair like 'EURUSD=X') and return a Date-indexed series.
    Handles flattened MultiIndex cols like 'Close EURUSD=X' / 'Adj Close EURUSD=X'.
    """
    end_plus = pd.to_datetime(end) + timedelta(days=1)
    fx = yf.download(ticker, start=start, end=end_plus, progress=False, auto_adjust=False)

    if fx is None or fx.empty:
        return pd.Series(dtype=float)

    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = [" ".join([c for c in tup if c]).strip() for tup in fx.columns.values]

    dfr = fx.reset_index().copy()

    # date column
    if "Date" in dfr.columns:
        date_col = "Date"
    else:
        date_col = next((c for c in dfr.columns if pd.api.types.is_datetime64_any_dtype(dfr[c])), dfr.columns[0])

    def is_close(col: str) -> bool:
        s = str(col).lower()
        return s.startswith("close") and not s.startswith("adj close")

    def is_adj(col: str) -> bool:
        return str(col).lower().startswith("adj close")

    close_col = next((c for c in dfr.columns if is_close(c)), None)
    adj_col   = next((c for c in dfr.columns if is_adj(c)), None)

    price_col = close_col or adj_col
    if price_col is None:
        return pd.Series(dtype=float)

    out = dfr[[date_col, price_col]].rename(columns={price_col: "FX"}).copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.set_index(date_col)["FX"].sort_index()

def _fx_series(src: str, dst: str, start, end) -> pd.Series:
    """
    Return f(t) such that: value_in_dst = value_in_src * f(t).
    Try direct (src+dst)=X, inverse (dst+src)=X, else triangulate via USD.
    """
    if src == dst:
        return pd.Series(dtype=float)

    # Direct
    s = _dl_fx(f"{src}{dst}=X", start, end)
    if not s.empty:
        return s  # dst per 1 src

    # Inverse
    inv = _dl_fx(f"{dst}{src}=X", start, end)
    if not inv.empty:
        return 1.0 / inv

    # Triangulate via USD
    if src != "USD" and dst != "USD":
        s_usd = _dl_fx(f"{src}USD=X", start, end)   # USD per src
        d_usd = _dl_fx(f"{dst}USD=X", start, end)   # USD per dst
        if not s_usd.empty and not d_usd.empty:
            return (s_usd / d_usd)  # (USD/src)/(USD/dst) = dst/src

        usd_s = _dl_fx(f"USD{src}=X", start, end)   # src per USD
        usd_d = _dl_fx(f"USD{dst}=X", start, end)   # dst per USD
        if not usd_s.empty and not usd_d.empty:
            return (1.0 / usd_s) * usd_d            # (USD/src)*(dst/USD) = dst/src

    return pd.Series(dtype=float)

def convert_currency(df: pd.DataFrame, src: str, dst: str, start, end) -> tuple[pd.DataFrame, bool]:
    """Convert Close/Adj Close from src -> dst using Yahoo FX."""
    if dst == "As reported" or dst == src:
        return df, False
    fx = _fx_series(src, dst, start, end)
    if fx.empty:
        return df, False

    out = df.copy().set_index("Date")
    fx = fx.sort_index().reindex(out.index).ffill().bfill()  # align, fill gaps
    for col in ["Close", "Adj Close"]:
        if col in out.columns:
            out[col] = out[col] * fx
    return out.reset_index(), True

# --------------------- Flow ---------------------
if not fetch and "base_df" not in st.session_state:
    st.info("Pick a **category**, **ticker**, choose **display currency**, set dates, then click **Fetch Data**.")
    st.stop()

if fetch:
    with st.spinner(f"Downloading {ticker}…"):
        raw = fetch_prices(ticker, start_date, end_date)

    if raw is None or raw.empty:
        st.warning("No data returned. Check ticker or date range."); st.stop()

    data = prep_df(raw)
    if data.empty:
        st.warning("Couldn't find expected Close/Adj Close columns."); st.stop()

    # --- currency handling ---
    original_currency = get_currency(ticker)
    target_currency = original_currency if display_currency == "As reported" else display_currency
    data_conv, did_convert = convert_currency(data, original_currency, target_currency, start_date, end_date)

    # --- display ---
    curr_symbol = CURRENCY_SYMBOL.get(target_currency.upper(), target_currency.upper())
    has_adj = "Adj Close" in data_conv.columns

    st.session_state.last_ticker = ticker

    st.subheader(f"{ticker} — {'Close' + (' & Adjusted Close' if has_adj else '')}")
    st.caption(
        f"Original currency: **{original_currency}** → Display: **{target_currency}** "
        f"({curr_symbol}) • Rows: **{len(data_conv)}**" + (" • FX conversion applied" if did_convert else "")
    )

    # TABLE: shows both columns when present
    st.dataframe(data_conv, use_container_width=True, height=420)

    # CHART: title + legend; plots both when present
    cols_to_plot = [c for c in ["Close", "Adj Close"] if c in data_conv.columns]
    plot_df = data_conv.melt("Date", value_vars=cols_to_plot, var_name="Series", value_name="Price")
    chart_title = f"{ticker} — {', '.join(cols_to_plot)} ({target_currency})"
    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title=f"Price ({target_currency})"),
            color=alt.Color("Series:N", title="Series"),
            tooltip=["Date:T", "Series:N", "Price:Q"],
        )
        .properties(title=chart_title)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # Save base CSV for fallback on feature page (in display currency)
    start_s = pd.to_datetime(data_conv["Date"].min()).strftime("%Y%m%d")
    end_s   = pd.to_datetime(data_conv["Date"].max()).strftime("%Y%m%d")
    fname = f"{ticker}_{start_s}_{end_s}_{target_currency}_{'close_adjclose' if has_adj else 'close'}.csv"
    fpath = os.path.join("data", fname)
    data_conv.to_csv(fpath, index=False)

    # Download
    buf = BytesIO(); data_conv.to_csv(buf, index=False)
    st.download_button("📥 Download CSV", data=buf.getvalue(), file_name=fname, mime="text/csv")

    st.success("✅ Finished fetching and rendering data.")

    # ---------- Proceed button (set flag, rerun → NAV GUARD switches) ----------
    def _go_feature():
        st.session_state.base_df = data_conv
        st.session_state.base_meta = {
            "ticker": ticker,
            "currency": target_currency,
            "display_currency": target_currency,
            "file_path": fpath,
        }
        st.session_state.nav_to_features = True
        st.rerun()

    st.button(f"🚀 Proceed with {ticker}", type="primary", key="proceed_btn", on_click=_go_feature)
