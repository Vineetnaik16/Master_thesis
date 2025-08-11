# pages/07_Global_Investment_Planner.py
from __future__ import annotations
from datetime import timedelta
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import yfinance as yf

st.set_page_config(page_title="Global Investment Planner", page_icon="🌍", layout="wide")
st.title("🌍 Global Investment Planner")

# ----------------------- Helpers -----------------------
def segmented(label, options, default=None, help=None):
    try:
        return st.segmented_control(label, options, default=default, help=help)
    except Exception:
        idx = options.index(default) if default in options else 0
        return st.radio(label, options, index=idx, help=help, horizontal=True)

@st.cache_data(show_spinner=False)
def dl_multi(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Batch download (chunked) and return tidy dataframe with columns: Ticker, Date, Price."""
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Date", "Price"])
    end_plus = pd.to_datetime(end) + timedelta(days=1)
    chunks = [tickers[i:i+40] for i in range(0, len(tickers), 40)]
    frames = []
    for ch in chunks:
        df = yf.download(ch, start=start, end=end_plus, progress=False, auto_adjust=False, group_by="ticker")
        if isinstance(df.columns, pd.MultiIndex):
            for t in ch:
                try:
                    sub = df[t].copy()
                except Exception:
                    continue
                col = None
                for cand in ["Adj Close", "Close", "Open"]:
                    if cand in sub.columns:
                        col = cand; break
                if col is None:
                    continue
                s = sub[[col]].reset_index().rename(columns={"Date":"Date", col:"Price"})
                s.insert(0, "Ticker", t)
                frames.append(s)
        else:
            cols = list(df.columns)
            col = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else ("Open" if "Open" in cols else None))
            if col:
                s = df[[col]].reset_index().rename(columns={col:"Price"})
                s.insert(0, "Ticker", ch[0] if ch else tickers[0])
                frames.append(s)
    if not frames:
        return pd.DataFrame(columns=["Ticker","Date","Price"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Price"])
    return out

def prepare_series(df_tidied: pd.DataFrame, tkr: str) -> pd.Series:
    """Return daily ffilled price series for one ticker, with duplicate-date safety."""
    d = df_tidied[df_tidied["Ticker"] == tkr][["Date", "Price"]].copy()
    if d.empty:
        return pd.Series(dtype=float)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.dropna().sort_values("Date")
    # remove duplicate dates (keep last)
    d = d[~d["Date"].duplicated(keep="last")]
    s = d.set_index("Date")["Price"]
    s = s[~s.index.duplicated(keep="last")]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx).ffill()

def expected_return(s: pd.Series, horizon_days: int) -> float:
    """Linear trend on last 60 obs → project H forward → pct change vs last."""
    s = s.dropna()
    tail = s.iloc[-60:] if len(s) > 60 else s
    if len(tail) < 5:
        return 0.0
    x = np.arange(len(tail))
    a, b = np.polyfit(x, tail.values.astype(float), 1)  # y ~ a*x + b
    last = float(tail.iloc[-1])
    pred = a * (len(tail) - 1 + horizon_days) + b
    return float((pred / last) - 1.0)

def volatility_30d(s: pd.Series) -> float:
    r = s.pct_change().dropna()
    if len(r) < 2:
        return 0.0
    r30 = r.iloc[-30:] if len(r) >= 30 else r
    return float(r30.std(ddof=1))

def regime_20(s: pd.Series) -> str:
    dd = s / s.cummax() - 1.0
    return "Bear" if float(dd.iloc[-1]) <= -0.20 else "Bull"

@st.cache_data(show_spinner=False)
def currency_of(t: str) -> str:
    try:
        finfo = yf.Ticker(t).fast_info
        if isinstance(finfo, dict):
            return finfo.get("currency", "USD") or "USD"
        return getattr(finfo, "currency", "USD") or "USD"
    except Exception:
        return "USD"

@st.cache_data(show_spinner=False)
def get_fx_rate(src: str, dst: str) -> float:
    """Return units of dst per 1 src. Tries direct, inverse, then USD bridge."""
    src = (src or "USD").upper()
    dst = (dst or "USD").upper()
    if src == dst:
        return 1.0

    def _last_close(symbol: str) -> float | None:
        try:
            df = yf.download(symbol, period="10d", interval="1d", progress=False, auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                val = float(df["Close"].dropna().iloc[-1])
                if np.isfinite(val):
                    return val
        except Exception:
            pass
        return None

    # Try direct pair (src->dst)
    direct = f"{src}{dst}=X"
    v = _last_close(direct)
    if v is not None:
        return float(v)

    # Try inverse pair (dst->src)
    inv = f"{dst}{src}=X"
    v = _last_close(inv)
    if v is not None and v != 0:
        return float(1.0 / v)

    # Bridge via USD
    if src != "USD" and dst != "USD":
        a = get_fx_rate(src, "USD")
        b = get_fx_rate("USD", dst)
        return float(a * b)

    # Fallback (rare)
    return 1.0

def confidence_for_risk(label: str) -> float:
    return {"Conservative": 0.95, "Balanced": 0.80, "Aggressive": 0.60}.get(label, 0.80)

# ----------------------- Universes -----------------------
GLOBAL_MEGA_ETF_CRYPTO = [
    "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","TSLA","BRK-B","JPM","V","MA","JNJ","WMT","PG","HD",
    "XOM","CVX","AVGO","ORCL","KO","PEP","UNH","MRK","ABBV","COST","NFLX","ADBE","PFE","NKE","AMD","QCOM",
    "SPY","QQQ","IWM","EEM","EFA","VGK","EWJ","VT","GLD","SLV","TLT","HYG","USO","CPER",
    "TSM","NVO","ASML","SAP","TM","NSRGY","LVMUY","TCEHY","BABA",
    "BTC-USD","ETH-USD"
]
NASDAQ_100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST","ADBE","NFLX","AMD","PEP","QCOM",
    "CSCO","AMAT","TMUS","TXN","INTU","LIN","PYPL","SBUX","BKNG","INTC","MU","MDLZ","VRTX","REGN","ADI","LRCX",
    "ISRG","MRVL","GILD","PANW","ABNB","CRWD","KLAC","SNPS","ADP","FTNT","CSX","ORLY","NXPI","ROST","PAYX",
    "CTAS","MAR","ODFL","IDXX","MRNA","PCAR","CDNS","ADSK","WDAY","EXC","XEL","CTSH","EBAY","TEAM","SIRI"
]
SPY_ETF_ONLY = ["SPY","QQQ","DIA","IWM","VT","EFA","EEM","EWJ","VGK","GLD","SLV","USO","TLT","HYG","XLF","XLK","XLE","XLI","XLY","XLP","XLV","XLU","XLB","XLRE"]

BUDGET_CURRENCIES = ["USD","EUR","GBP","JPY","INR","CAD","AUD","CHF","CNY"]

# ----------------------- Sidebar Controls -----------------------
with st.sidebar:
    st.header("Universe & Budget")
    universe = st.selectbox(
        "Universe",
        ["Global (Megacaps + ETFs + Crypto)", "NASDAQ-100 (fast)", "ETFs only (fast)", "Custom (paste)", "Custom (upload CSV)"],
        index=0,
        help="Scanning the entire world live isn't feasible; pick a broad universe or paste your own."
    )
    max_tickers = st.slider("Max tickers to scan", 20, 300, 120, 10,
                            help="Limits how many we fetch to avoid rate limits.")
    budget_ccy = st.selectbox("Budget currency", BUDGET_CURRENCIES, index=0)
    budget_amt = st.number_input(f"Budget amount ({budget_ccy})", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    risk = segmented("Risk appetite", ["Conservative","Balanced","Aggressive"], default="Balanced")
    cap_single = st.slider("Max weight per asset", 0.05, 1.0, 0.30, 0.05)
    opt_horizon_label = segmented("Optimize for", ["Next day","1 week","2 weeks","3 weeks","1 month"], default="1 month")
    H_MAP = {"Next day":1, "1 week":7, "2 weeks":14, "3 weeks":21, "1 month":30}
    H_OPT = H_MAP[opt_horizon_label]

    pasted = ""
    uploaded = None
    if universe == "Custom (paste)":
        pasted = st.text_area("Paste tickers separated by spaces or commas", value="AAPL MSFT NVDA AMZN META SPY QQQ BTC-USD")
    elif universe == "Custom (upload CSV)":
        uploaded = st.file_uploader("Upload CSV with a 'Ticker' column (or first column = tickers)", type=["csv"])

    run = st.button("Scan & Build Plan", type="primary", use_container_width=True)

# ----------------------- Build ticker list -----------------------
def _parse_pasted(s: str) -> List[str]:
    raw = [t.strip().upper() for t in s.replace("\n"," ").replace(","," ").split(" ") if t.strip()]
    return list(dict.fromkeys(raw))

def _read_uploaded(file) -> List[str]:
    try:
        dfu = pd.read_csv(file)
        col = "Ticker" if "Ticker" in dfu.columns else dfu.columns[0]
        vals = [str(x).strip().upper() for x in dfu[col].tolist() if str(x).strip()]
        return list(dict.fromkeys(vals))
    except Exception:
        return []

if universe == "Global (Megacaps + ETFs + Crypto)":
    tickers = GLOBAL_MEGA_ETF_CRYPTO[:]
elif universe == "NASDAQ-100 (fast)":
    tickers = NASDAQ_100[:]
elif universe == "ETFs only (fast)":
    tickers = SPY_ETF_ONLY[:]
elif universe == "Custom (paste)":
    tickers = _parse_pasted(pasted)
elif universe == "Custom (upload CSV)":
    tickers = _read_uploaded(uploaded) if uploaded is not None else []
else:
    tickers = []

tickers = [t for t in tickers if t]
if len(tickers) > max_tickers:
    tickers = tickers[:max_tickers]

# ----------------------- Main Action -----------------------
if not run:
    st.info("Choose a universe, set your budget + currency, then click **Scan & Build Plan**.")
    st.stop()

if not tickers:
    st.warning("No tickers to scan. Pick a universe, paste tickers, or upload a CSV.")
    st.stop()

st.write(f"Scanning **{len(tickers)}** tickers… this can take ~10–40s depending on limits.")

start = pd.to_datetime("today") - pd.Timedelta(days=900)
end = pd.Timestamp.today()

with st.spinner("Downloading prices…"):
    tidy = dl_multi(tickers, start, end)

if tidy.empty:
    st.error("No price data returned. Try fewer tickers, a different universe, or try again later.")
    st.stop()

# ----------------------- Compute metrics -----------------------
CONF = confidence_for_risk(risk)
rows: List[Dict[str, Any]] = []

for t in tickers:
    s = prepare_series(tidy, t)
    if s.dropna().empty:
        continue
    last_native = float(s.iloc[-1])
    vol = volatility_30d(s)
    reg = regime_20(s)
    mu_1  = expected_return(s, 1)
    mu_7  = expected_return(s, 7)
    mu_14 = expected_return(s, 14)
    mu_21 = expected_return(s, 21)
    mu_30 = expected_return(s, 30)
    # currency + FX to budget currency
    t_ccy = currency_of(t)
    fx_to_budget = get_fx_rate(t_ccy, budget_ccy)  # budget_ccy per 1 t_ccy
    last_budget = last_native * fx_to_budget

    rows.append({
        "Ticker": t,
        "Currency": t_ccy,
        "Last_native": last_native,
        "Last_budget": last_budget,
        "FX_to_budget": fx_to_budget,
        "Vol30": vol,
        "Regime": reg,
        "mu_1": mu_1, "mu_7": mu_7, "mu_14": mu_14, "mu_21": mu_21, "mu_30": mu_30,
        "Series": s,
    })

if not rows:
    st.error("No usable series found after download.")
    st.stop()

df = pd.DataFrame(rows)

# Score = positive expected return (chosen horizon) / volatility, regime-adjusted
H_COL = {1:"mu_1",7:"mu_7",14:"mu_14",21:"mu_21",30:"mu_30"}[H_OPT]
eps = 1e-9
base = np.maximum(df[H_COL].values, 0.0) / (df["Vol30"].values + eps)
reg_mult = np.where(df["Regime"].values == "Bull",
                    1.20 if risk != "Aggressive" else 1.10,
                    0.80 if risk != "Aggressive" else 0.90)
score = base * reg_mult
df["Score"] = score

df = df.sort_values("Score", ascending=False).reset_index(drop=True)

# ----------------------- Build Plan within Budget (in budget currency) -----------------------
if np.all(score <= 0):
    weights = np.ones(len(df)) / len(df)
else:
    weights = score / (score.sum() + eps)

cap = float(cap_single)
weights = np.minimum(weights, cap)
if weights.sum() == 0:
    weights = np.ones_like(weights) / len(weights)
weights /= weights.sum()

df["TargetWeight"] = weights
df["Buy_budget_$"] = df["TargetWeight"] * float(budget_amt)
# Use budget-currency price to compute quantities
df["Qty"] = np.floor(df["Buy_budget_$"] / df["Last_budget"]).astype(int)
df["Cost_budget_$"] = df["Qty"] * df["Last_budget"]

plan = df[df["Qty"] > 0].copy()
current_value_budget = float(plan["Cost_budget_$"].sum())
leftover_cash = max(0.0, float(budget_amt) - current_value_budget)

# If nothing affordable, greedy one-share method by Score until budget used
if plan.empty and budget_amt > 0:
    leftover = float(budget_amt)
    qty = np.zeros(len(df), dtype=int)
    for _ in range(2000):
        affordable = np.where(df["Last_budget"].values <= leftover)[0]
        if len(affordable) == 0:
            break
        k = affordable[np.argmax(df["Score"].values[affordable])]
        qty[k] += 1
        leftover -= df["Last_budget"].values[k]
    df["Qty"] = qty
    df["Cost_budget_$"] = df["Qty"] * df["Last_budget"]
    plan = df[df["Qty"] > 0].copy()
    current_value_budget = float(plan["Cost_budget_$"].sum())
    leftover_cash = max(0.0, float(budget_amt) - current_value_budget)

# ----------------------- Forecast Plan Value (constant FX) -----------------------
def projected_value_budget_at(h: int) -> float:
    if plan.empty:
        return 0.0
    mu = plan[{1:"mu_1",7:"mu_7",14:"mu_14",21:"mu_21",30:"mu_30"}[h]].values
    # Forecast native price, then convert to budget currency with today's FX (constant FX assumption)
    price_native_f = plan["Last_native"].values * (1.0 + mu)
    price_budget_f = price_native_f * plan["FX_to_budget"].values
    return float(np.sum(price_budget_f * plan["Qty"].values))

horizons = [1, 7, 14, 21, 30]
fcast = []
for h in horizons:
    v = projected_value_budget_at(h)
    chg = v - current_value_budget
    pct = (chg / current_value_budget * 100.0) if current_value_budget > 0 else 0.0
    fcast.append({
        "Horizon": h,
        "Label": {1:"Next day",7:"1 week",14:"2 weeks",21:"3 weeks",30:"1 month"}[h],
        f"Projected_{budget_ccy}": v,
        f"Change_{budget_ccy}": chg,
        "Change_%": pct
    })
fdf = pd.DataFrame(fcast)

# ----------------------- Output -----------------------
st.markdown("### 🛒 Tailored buy list (within budget)")
if plan.empty:
    st.info("Budget too small to buy any one share from the universe. Try a larger budget or a different universe.")
else:
    show = plan[["Ticker","Currency","Regime","Last_native","Last_budget","Vol30","TargetWeight","Qty","Cost_budget_$",H_COL]].copy()
    show = show.rename(columns={
        "Last_native": f"Last ({plan['Currency'].iloc[0]} variable)",  # informational
        "Last_budget": f"Last ({budget_ccy})",
        "Vol30": "Vol 30d",
        "TargetWeight": "Target Weight",
        "Cost_budget_$": f"Cost ({budget_ccy})",
        H_COL: f"Exp Return {opt_horizon_label} %",
    })
    st.dataframe(
        show.style.format({
            f"Last ({budget_ccy})": "{:.2f}",
            "Vol 30d": "{:.2%}",
            "Target Weight": "{:.1%}",
            f"Exp Return {opt_horizon_label} %": "{:+.2%}",
            f"Cost ({budget_ccy})": "{:,.2f}",
            "Qty": "{:d}",
        }),
        use_container_width=True, height=460
    )
    st.caption(f"Budget: **{float(budget_amt):,.2f} {budget_ccy}**  •  Deployed: **{current_value_budget:,.2f} {budget_ccy}**  •  Leftover: **{leftover_cash:,.2f} {budget_ccy}**")

st.markdown(f"### 📈 Forecast of plan value (in {budget_ccy}, constant FX)")
if plan.empty:
    st.info("No purchases → no forecast.")
else:
    cols_show = ["Label", f"Projected_{budget_ccy}", f"Change_{budget_ccy}", "Change_%"]
    st.dataframe(
        fdf[cols_show].rename(columns={
            "Label": "Horizon",
            f"Projected_{budget_ccy}": f"Projected Value ({budget_ccy})",
            f"Change_{budget_ccy}": f"Change ({budget_ccy})",
            "Change_%": "Change (%)",
        }).style.format({
            f"Projected Value ({budget_ccy})": "{:,.2f}",
            f"Change ({budget_ccy})": "{:+,.2f}",
            "Change (%)": "{:+.2f}%",
        }),
        use_container_width=True, height=250
    )
    ch = (
        alt.Chart(fdf.rename(columns={f"Projected_{budget_ccy}":"Projected"}))
        .mark_bar()
        .encode(
            x=alt.X("Label:N", title="Horizon"),
            y=alt.Y("Projected:Q", title=f"Projected value ({budget_ccy})", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("Projected:Q", title="Projected value", format=",.2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(ch, use_container_width=True)

# ----------------------- Explainer -----------------------
with st.expander("How this works"):
    st.write(f"""
    - You chose a **budget** in **{budget_ccy}**. We convert each ticker’s price into {budget_ccy} using live FX (Yahoo pairs like `EUR{budget_ccy}=X`).
    - For every symbol we compute a short-term **trend projection**, **30d volatility**, and a **Bull/Bear** regime tag (20% drawdown rule).
    - We rank by **expected-return / volatility**, nudge for regime (Bull favored), cap single weights, and allocate within your budget.
    - Forecasts assume **constant FX** (no currency movement) and show values in **{budget_ccy}** for: next day / 1–4 weeks.
    """)

# ----------------------- Nav -----------------------
c1, c2 = st.columns(2)
with c1:
    if st.button("← Back to Portfolio"):
        try: st.switch_page("pages/06_Portfolio.py")
        except Exception: st.rerun()
with c2:
    if st.button("🏠 Home"):
        try: st.switch_page("Home.py")
        except Exception: st.rerun()
