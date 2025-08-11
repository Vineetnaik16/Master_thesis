# pages/06_Portfolio.py
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import yfinance as yf

st.set_page_config(page_title="Portfolio", page_icon="📊", layout="wide")
st.title("📊 Portfolio")

# ----- segmented control fallback -----
def segmented(label, options, default=None, help=None):
    try:
        return st.segmented_control(label, options, default=default, help=help)
    except Exception:
        idx = options.index(default) if default in options else 0
        return st.radio(label, options, index=idx, help=help, horizontal=True)

cart = st.session_state.get("portfolio_cart", {})
store = st.session_state.get("portfolio_store", {})

if not cart:
    st.warning("Your cart is empty. Add tickers in **Build Your Portfolio**.")
    if st.button("← Go to Builder"):
        try: st.switch_page("pages/05_Portfolio_Builder.py")
        except Exception: st.rerun()
    st.stop()

# --- Controls ---
c1, c2, c3 = st.columns(3)
with c1:
    horizon_label = segmented("Time horizon", ["1M", "3M", "6M"], default="3M")
    H = {"1M": 30, "3M": 90, "6M": 180}[horizon_label]
with c2:
    risk = segmented("Risk appetite", ["Conservative","Balanced","Aggressive"], default="Balanced")
with c3:
    cap_single = st.slider("Max weight per asset", 0.05, 1.0, 0.35, 0.05)

def confidence_for_risk(label: str) -> float:
    return {"Conservative": 0.95, "Balanced": 0.80, "Aggressive": 0.60}.get(label, 0.80)
CONF = confidence_for_risk(risk)

# --- helpers ---
@st.cache_data(show_spinner=False)
def dl_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    end_plus = pd.to_datetime(end) + timedelta(days=1)
    df = yf.download(ticker, start=start, end=end_plus, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([c for c in tup if c]).strip() for tup in df.columns.values]
    return df.reset_index()

def pick_price_col_robust(df: pd.DataFrame):
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

def prepare_series(df: pd.DataFrame, price_label: str) -> pd.Series:
    s = df.set_index("Date")[price_label].sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx).ffill()

def regime_20(s: pd.Series) -> str:
    dd = s / s.cummax() - 1.0
    return "Bear" if float(dd.iloc[-1]) <= -0.20 else "Bull"

def expected_return(s: pd.Series, horizon_days: int) -> float:
    """Linear trend on last 60 days -> project H days ahead (quick & dependency-light)."""
    tail = s.dropna().iloc[-60:] if len(s) > 60 else s.dropna()
    if len(tail) < 5:
        return 0.0
    x = np.arange(len(tail))
    a, b = np.polyfit(x, tail.values.astype(float), 1)  # y ~ a*x + b
    last = float(tail.iloc[-1])
    pred = a * (len(tail) - 1 + horizon_days) + b
    return float((pred / last) - 1.0)

def volatility(s: pd.Series, window: int = 30) -> float:
    r = s.pct_change().dropna()
    if len(r) < 2: return 0.0
    return float(r.iloc[-window:].std(ddof=1) if len(r) >= window else r.std(ddof=1))

def currency_of(t: str) -> str:
    try:
        finfo = yf.Ticker(t).fast_info
        return finfo.get("currency", "USD") if isinstance(finfo, dict) else getattr(finfo, "currency", "USD")
    except Exception:
        return "USD"

# --- per-ticker prep ---
rows = []
currencies = set()
for tkr in cart.keys():
    rec = store.get(tkr)
    if rec and "df" in rec:
        df = rec["df"].copy()
        price_label = rec["price_col"]
        curr = rec.get("currency", currency_of(tkr))
    else:
        df_raw = dl_prices(tkr, pd.to_datetime("2018-01-01"), pd.Timestamp.today())
        col_raw, price_label = pick_price_col_robust(df_raw)
        if col_raw is None:
            st.warning(f"{tkr}: no price column found, skipped.")
            continue
        df = df_raw[["Date", col_raw]].rename(columns={col_raw: price_label})
        curr = currency_of(tkr)

    s = prepare_series(df, price_label)
    if s.dropna().empty:
        st.warning(f"{tkr}: empty series, skipped.")
        continue

    mu = expected_return(s, H)     # expected % over horizon
    vol = volatility(s, 30)        # 30d volatility (daily stdev)
    reg = regime_20(s)
    last_price = float(s.iloc[-1])
    currencies.add(curr)

    rows.append({
        "Ticker": tkr, "Currency": curr, "Last Price": last_price,
        f"Exp Return {H}d %": mu * 100.0,
        "Volatility 30d %": vol * 100.0,
        "Regime": reg,
        "Series": s,  # keep for chart
    })

if not rows:
    st.error("No valid data found for the tickers in your cart.")
    st.stop()

if len(currencies) > 1:
    st.warning("⚠️ Your portfolio includes multiple currencies. Returns/weights don’t account for FX.")

# --- suggestions ---
def suggestion(mu: float, reg: str, risk_level: str) -> str:
    up_thr = {"Conservative": 0.02, "Balanced": 0.04, "Aggressive": 0.06}[risk_level]
    down_thr = {"Conservative": -0.02, "Balanced": -0.04, "Aggressive": -0.06}[risk_level]
    if reg == "Bear" and mu <= down_thr: return "Reduce"
    if reg == "Bull" and mu >= up_thr:   return "Buy More"
    return "Hold"

dfp = pd.DataFrame(rows)
dfp["Suggestion"] = [
    suggestion(mu=r/100.0, reg=reg, risk_level=risk)
    for r, reg in zip(dfp[f"Exp Return {H}d %"], dfp["Regime"])
]

# --- suggested weights ---
eps = 1e-9
scores = np.maximum(dfp[f"Exp Return {H}d %"].values / 100.0, 0.0) / (dfp["Volatility 30d %"].values/100.0 + eps)
weights = (np.ones_like(scores) / len(scores)) if np.all(scores <= 0) else (scores / (scores.sum() + eps))
cap = float(cap_single)
weights = np.minimum(weights, cap)
if weights.sum() == 0:
    weights = np.ones_like(weights) / len(weights)
weights /= weights.sum()
dfp["Suggested Weight %"] = (weights * 100.0)

# --- overview tiles ---
avg_mu = float((dfp[f"Exp Return {H}d %"] * dfp["Suggested Weight %"] / 100.0).sum())
port_vol = float(np.average(dfp["Volatility 30d %"], weights=dfp["Suggested Weight %"]))  # heuristic
bull_share = float((dfp["Regime"] == "Bull").mean() * 100.0)

t1, t2, t3 = st.columns(3)
t1.metric("Portfolio expected return", f"{avg_mu:.2f}%")
t2.metric("Avg volatility (30d)", f"{port_vol:.2f}%")
t3.metric("Bull regime share", f"{bull_share:.0f}%")

# --- table ---
st.markdown("### Portfolio suggestions")
show_cols = ["Ticker","Currency","Last Price", f"Exp Return {H}d %","Volatility 30d %","Regime","Suggestion","Suggested Weight %"]
st.dataframe(
    dfp[show_cols].style.format({
        "Last Price": "{:.2f}",
        f"Exp Return {H}d %": "{:+.2f}",
        "Volatility 30d %": "{:.2f}",
        "Suggested Weight %": "{:.1f}",
    }),
    use_container_width=True, height=360
)

# --- mini charts (1Y indexed) ---
st.markdown("### Price mini-charts (last 1 year, start=100)")
cols = st.columns(3)
for i, row in dfp.iterrows():
    col = cols[i % 3]
    s_full = row["Series"].dropna()
    if s_full.empty:
        continue
    cutoff = s_full.index.max() - pd.Timedelta(days=365)
    s = s_full[s_full.index >= cutoff]
    if s.empty:
        s = s_full.iloc[-365:]
    base = float(s.iloc[0])
    idx = (s / base) * 100.0
    color = "#16a34a" if idx.iloc[-1] >= 100.0 else "#dc2626"

    dfc = pd.DataFrame({"Date": idx.index, "Index100": idx.values, "Price": s.values})
    chart = (
        alt.Chart(dfc)
        .mark_line(strokeWidth=2.5, color=color)
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("Index100:Q", title=None, scale=alt.Scale(nice=True),
                    axis=alt.Axis(labels=False, ticks=False, grid=False)),
            tooltip=[
                "Date:T",
                alt.Tooltip("Price:Q", title="Price"),
                alt.Tooltip("Index100:Q", title="Index (start=100)", format=".1f"),
            ],
        )
        .properties(height=90, title=row["Ticker"])
    )
    col.altair_chart(chart, use_container_width=True)

# --- download ---
csv = dfp[show_cols].to_csv(index=False).encode()
st.download_button("📥 Download suggestions", data=csv, file_name=f"portfolio_suggestions_{H}d.csv", mime="text/csv")

# --- nav ---
c_left, c_right = st.columns(2)
with c_left:
    if st.button("← Back to Builder"):
        try: st.switch_page("pages/05_Portfolio_Builder.py")
        except Exception: st.rerun()
with c_right:
    if st.button("🏠 Back to Home"):
        try: st.switch_page("Home.py")
        except Exception: st.rerun()

st.caption("Educational use only. Suggestions are heuristic and do not constitute investment advice.")
