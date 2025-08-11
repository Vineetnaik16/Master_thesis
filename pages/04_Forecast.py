# pages/03_Forecast.py
import os
from io import BytesIO
from typing import Tuple, List, Callable, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------- Page ----------------
st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Forecast")

# -------- Optional deps (graceful fallbacks) --------
HAS_STATSMODELS = True
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    HAS_STATSMODELS = False

HAS_XGB = True
try:
    import xgboost as xgb
except Exception:
    HAS_XGB = False

# ---------------- Data access ----------------
meta = st.session_state.get("base_meta", {})
df = st.session_state.get("base_df")

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
    st.warning("No input data found. Go back to **Home**, fetch data, then click **Proceed**.")
    if st.button("↩ Back to main"):
        for k in ("base_df","base_meta","nav_to_features","nav_to_forecast"):
            st.session_state.pop(k, None)
        try: st.switch_page("Home.py")
        except Exception: st.rerun()
    st.stop()

# Use Adj Close if present
price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
if price_col is None:
    st.error("Neither **Adj Close** nor **Close** exists in the dataset."); st.stop()

data = df.copy()[["Date", price_col]].sort_values("Date").dropna()
data["Date"] = pd.to_datetime(data["Date"])

ticker = meta.get("ticker", "TICKER")
currency = meta.get("currency", "USD")
st.caption(f"Ticker: **{ticker}** • Currency: **{currency}** • Series: **{price_col}**")

# ---------------- Helpers ----------------
def segmented(label, options, default=None, help=None):
    """Use st.segmented_control if available, else fall back to radio."""
    try:
        return st.segmented_control(label, options, default=default, help=help)
    except Exception:
        idx = options.index(default) if default in options else 0
        return st.radio(label, options, index=idx, help=help, horizontal=True)

def _prepare_daily_series(df: pd.DataFrame) -> pd.Series:
    y = df.set_index("Date")[price_col].sort_index()
    full_idx = pd.date_range(y.index.min(), y.index.max(), freq="D")
    y = y.reindex(full_idx).ffill()
    return y

def _z_from_conf(conf: float) -> float:
    # two-sided z ~ quantile; map common levels
    mapping = {0.60: 0.84162, 0.80: 1.28155, 0.90: 1.64485, 0.95: 1.95996}
    return mapping.get(round(conf, 2), 1.95996)

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1e-8, np.abs(y_true)))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

# ----- Forecast engines -----
def naive_forecast(y: pd.Series, steps: int):
    last = y.iloc[-1]
    idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
    fc = pd.Series(last, index=idx, name="Forecast")
    diffs = y.diff().dropna()
    resid_std = float(diffs.std(ddof=1)) if len(diffs) > 1 else float(y.pct_change().std(ddof=1) * abs(last))
    return fc, resid_std

def ma_forecast(y: pd.Series, steps: int, window: int = 20):
    y_ma = y.rolling(window).mean().dropna()
    last = (y_ma.iloc[-1] if len(y_ma) else y.iloc[-window:].mean())
    idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
    fc = pd.Series(last, index=idx, name="Forecast")
    resid = (y - y.rolling(window).mean()).dropna()
    resid_std = float(resid.std(ddof=1)) if len(resid) > 1 else float((y - y.mean()).std(ddof=1))
    return fc, resid_std

def ets_forecast(y: pd.Series, steps: int):
    if not HAS_STATSMODELS: return ma_forecast(y, steps, 20)
    model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
    res = model.fit(optimized=True)
    fc = res.forecast(steps).rename("Forecast")
    resid = res.resid.dropna()
    resid_std = float(resid.std(ddof=1)) if len(resid) > 1 else float(y.std(ddof=1))
    return fc, resid_std

def sarimax_forecast(y: pd.Series, steps: int):
    if not HAS_STATSMODELS: return ma_forecast(y, steps, 20)
    # simple, robust default
    model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps).predicted_mean.rename("Forecast")
    resid = res.resid.dropna()
    resid_std = float(resid.std(ddof=1)) if len(resid) > 1 else float(y.std(ddof=1))
    return fc, resid_std

def xgb_forecast(y: pd.Series, steps: int, lags: int = 30):
    if not HAS_XGB: return ma_forecast(y, steps, 20)
    if len(y) <= lags + 5:
        return ma_forecast(y, steps, 20)
    arr = y.values.astype(np.float32)
    X, Y = [], []
    for i in range(lags, len(arr)):
        X.append(arr[i-lags:i]); Y.append(arr[i])
    X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
        random_state=42, n_jobs=2,
    )
    model.fit(X, Y)
    preds, hist = [], arr.tolist()
    for _ in range(steps):
        x_in = np.array(hist[-lags:], dtype=np.float32).reshape(1, -1)
        yhat = float(model.predict(x_in)[0])
        preds.append(yhat); hist.append(yhat)
    idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
    fc = pd.Series(preds, index=idx, name="Forecast")
    train_pred = model.predict(X)
    resid_std = float(np.std(Y - train_pred, ddof=1))
    return fc, resid_std

def make_bands(fc: pd.Series, resid_std: float, conf: float) -> pd.DataFrame:
    z = _z_from_conf(conf)
    bands = pd.DataFrame(index=fc.index)
    bands[f"lower_{int(conf*100)}"] = fc - z * resid_std
    bands[f"upper_{int(conf*100)}"] = fc + z * resid_std
    return bands

# ----- Candidate registry -----
def candidate_models() -> List[tuple[str, Callable[[pd.Series, int], Tuple[pd.Series, float]]]]:
    cands: List[tuple[str, Callable]] = [
        ("Naive", lambda s, k: naive_forecast(s, k)),
        ("Moving Average (w=20)", lambda s, k: ma_forecast(s, k, 20)),
    ]
    cands.append(("Holt-Winters (ETS)" if HAS_STATSMODELS else "ETS (fallback to MA)", lambda s, k: ets_forecast(s, k)))
    cands.append(("SARIMAX (1,1,1)" if HAS_STATSMODELS else "SARIMAX (fallback to MA)", lambda s, k: sarimax_forecast(s, k)))
    if HAS_XGB:
        cands.append(("XGBoost (lags=30)", lambda s, k: xgb_forecast(s, k, 30)))
    return cands

# ----- Auto select with evaluation (for Simple mode) -----
def evaluate_candidates(y: pd.Series, steps: int, holdout: int) -> List[Dict[str, Any]]:
    y_train, y_test = y.iloc[:-holdout], y.iloc[-holdout:]
    results: List[Dict[str, Any]] = []
    for name, fn in candidate_models():
        try:
            # backtest
            fc_bt, _ = fn(y_train, len(y_test))
            yhat = fc_bt.reindex(y_test.index)
            m = metrics(y_test.values, yhat.values)
            # real forecast for horizon
            fc, resid_std = fn(y, steps)
            results.append({"name": name, "metrics": m, "fc": fc, "resid_std": resid_std})
        except Exception as e:
            results.append({"name": name, "metrics": {"MAE": np.nan, "RMSE": np.nan, "MAPE%": np.nan}, "fc": None, "resid_std": np.nan, "error": str(e)})
    # sort by RMSE (asc), NaNs at bottom
    results = sorted(results, key=lambda r: (np.inf if np.isnan(r["metrics"]["RMSE"]) else r["metrics"]["RMSE"]))
    return results

def auto_select_from(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    for r in results:
        if not np.isnan(r["metrics"]["RMSE"]) and r["fc"] is not None:
            return r
    # fallback
    fc, resid_std = naive_forecast(y, 30)
    return {"name": "Naive", "metrics": {"MAE": np.nan, "RMSE": np.nan, "MAPE%": np.nan}, "fc": fc, "resid_std": resid_std}

# ---------------- UI: Simple vs Advanced ----------------
mode = segmented("Mode", ["Simple", "Advanced"], default="Simple",
                 help="Simple hides model details; Advanced exposes full controls.")

y = _prepare_daily_series(data)
if len(y) < 30:
    st.warning("Not enough data to forecast. Try a longer date range."); st.stop()

# Risk/confidence mapping (used in both modes)
def confidence_for_risk(label: str) -> float:
    return {"Conservative": 0.95, "Balanced": 0.80, "Aggressive": 0.60}.get(label, 0.80)

if mode == "Simple":
    # Controls for novices
    horizon_label = segmented("Time horizon", ["1M", "3M", "6M"], default="3M")
    H = {"1M": 30, "3M": 90, "6M": 180}[horizon_label]

    risk = segmented(
        "Risk appetite", ["Conservative", "Balanced", "Aggressive"], default="Balanced",
        help="Wider bands = more conservative; narrower bands = more aggressive."
    )
    conf = confidence_for_risk(risk)

    st.caption("Forecast engine: **Auto** — we test a few models on recent data and pick the best.")

    holdout = min(250, max(60, len(y)//5))
    evals = evaluate_candidates(y, H, holdout)
    best = auto_select_from(evals)

    fc, resid_std = best["fc"], best["resid_std"]
    best_name = best["name"]
    bands = make_bands(fc, resid_std, conf)

    # We'll show the "other models" section using the evals list below.

else:
    # ---------- Advanced: expose models & knobs ----------
    with st.sidebar:
        st.header("Advanced settings")
        H = st.number_input("Forecast horizon (days)", min_value=1, max_value=365*2, value=60, step=1)
        model_choice = st.selectbox(
            "Model",
            ["Naive", "Moving Average (w=20)", "Exponential Smoothing (Holt-Winters)", "SARIMAX (1,1,1)"] + (["XGBoost (lags=30)"] if HAS_XGB else []),
        )
        risk = segmented("Risk appetite", ["Conservative", "Balanced", "Aggressive"], default="Balanced",
                         help="Affects how wide the forecast band looks.")
    conf = confidence_for_risk(risk)

    if model_choice == "Naive":
        fc, resid_std = naive_forecast(y, H); best_name = "Naive"
    elif model_choice == "Moving Average (w=20)":
        fc, resid_std = ma_forecast(y, H, 20); best_name = "Moving Average (w=20)"
    elif model_choice == "Exponential Smoothing (Holt-Winters)":
        fc, resid_std = ets_forecast(y, H); best_name = "Holt-Winters (ETS)" if HAS_STATSMODELS else "ETS (fallback to MA)"
    elif model_choice == "SARIMAX (1,1,1)":
        fc, resid_std = sarimax_forecast(y, H); best_name = "SARIMAX (1,1,1)" if HAS_STATSMODELS else "SARIMAX (fallback to MA)"
    elif model_choice == "XGBoost (lags=30)" and HAS_XGB:
        fc, resid_std = xgb_forecast(y, H, 30); best_name = "XGBoost (lags=30)"
    else:
        fc, resid_std = naive_forecast(y, H); best_name = "Naive"

    bands = make_bands(fc, resid_std, conf)

    # For the comparison section, also evaluate everything (once)
    holdout = min(250, max(60, len(y)//5))
    evals = evaluate_candidates(y, H, holdout)

# ---------------- Insights ----------------
# Trend (change over last 30 days), Volatility (std of returns 30d), Regime (20% rule)
tail = y.iloc[-30:] if len(y) > 30 else y
trend_pct = float((tail.iloc[-1] / tail.iloc[0] - 1) * 100.0) if len(tail) > 1 else 0.0
vol = float(y.pct_change().iloc[-30:].std(ddof=1)) if len(y) > 30 else float(y.pct_change().std(ddof=1))
vol_label = "Low" if vol < 0.01 else ("Medium" if vol < 0.02 else "High")

peak = y.cummax()
drawdown = (y / peak) - 1.0
regime = "Bear" if drawdown.iloc[-1] <= -0.20 else "Bull"

c1, c2, c3 = st.columns(3)
c1.metric("Trend (30d)", f"{trend_pct:+.2f}%")
c2.metric("Volatility", vol_label, f"{vol*100:.2f}% stdev")
c3.metric("Current regime", regime)

# ---------------- Plot (last 50 days + forecast) ----------------
hist_tail = y.iloc[-50:] if len(y) > 50 else y
hist_df = pd.DataFrame({"Date": hist_tail.index, "Price": hist_tail.values})
fc_df = pd.DataFrame({"Date": fc.index, "Forecast": fc.values})

# grab band cols
lower_col = [c for c in bands.columns if c.startswith("lower_")][0]
upper_col = [c for c in bands.columns if c.startswith("upper_")][0]
band_df = bands.copy(); band_df["Date"] = band_df.index

title_txt = f"{ticker} — {price_col} forecast ({best_name}, {int(conf*100)}% band)"

base = alt.Chart(hist_df).mark_line().encode(
    x=alt.X("Date:T", title="Date"),
    y=alt.Y("Price:Q", title=f"Price ({currency})"),
    tooltip=["Date:T", "Price:Q"],
).properties(title=title_txt)

fc_line = alt.Chart(fc_df).mark_line(color="#d97706").encode(
    x="Date:T", y="Forecast:Q", tooltip=["Date:T","Forecast:Q"]
)

band = alt.Chart(band_df).mark_area(opacity=0.18, color="#d97706").encode(
    x="Date:T", y=f"{lower_col}:Q", y2=f"{upper_col}:Q"
)

st.altair_chart(alt.layer(base, band, fc_line).interactive(), use_container_width=True)

# ---------------- Backtest (expander) ----------------
with st.expander("How good is this model? (quick backtest)"):
    bt_n = min(60, max(10, len(y)//10))
    y_train, y_test = y.iloc[:-bt_n], y.iloc[-bt_n:]
    # re-fit same chosen method
    def mimic_choice(train: pd.Series, k: int) -> pd.Series:
        if best_name.startswith("Naive"):
            return naive_forecast(train, k)[0]
        if best_name.startswith("Moving Average"):
            return ma_forecast(train, k, 20)[0]
        if best_name.startswith("Holt-Winters") or best_name.startswith("ETS"):
            return ets_forecast(train, k)[0]
        if best_name.startswith("SARIMAX"):
            return sarimax_forecast(train, k)[0]
        if best_name.startswith("XGBoost"):
            return xgb_forecast(train, k, 30)[0]
        return naive_forecast(train, k)[0]

    try:
        yhat = mimic_choice(y_train, len(y_test)).reindex(y_test.index)
        m = metrics(y_test.values, yhat.values)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{m['MAE']:.4f}")
        c2.metric("RMSE", f"{m['RMSE']:.4f}")
        c3.metric("MAPE", f"{m['MAPE%']:.2f}%")

        bt_df = pd.DataFrame({"Date": y_test.index, "Actual": y_test.values, "Predicted": yhat.values})
        lc = (
            alt.Chart(bt_df.melt("Date", value_vars=["Actual","Predicted"], var_name="Series", value_name="Value"))
            .mark_line()
            .encode(x="Date:T", y="Value:Q", color="Series:N", tooltip=["Date:T","Series:N","Value:Q"])
            .properties(title="Backtest: last N days")
            .interactive()
        )
        st.altair_chart(lc, use_container_width=True)
    except Exception as e:
        st.info(f"Backtest skipped: {e}")

# ---------------- Comparison: other models ----------------
st.markdown("### 📊 How do other methods compare?")
# Table of metrics (sorted by RMSE)
table_rows = []
for r in evals:
    row = {
        "Model": r["name"],
        "RMSE": r["metrics"]["RMSE"],
        "MAE": r["metrics"]["MAE"],
        "MAPE%": r["metrics"]["MAPE%"],
    }
    table_rows.append(row)
comp_df = pd.DataFrame(table_rows)
st.dataframe(comp_df, use_container_width=True)

# Mini charts for each candidate (tabs)
tabs = st.tabs([r["name"] for r in evals])
for tab, r in zip(tabs, evals):
    with tab:
        if r.get("fc") is None or r["fc"] is np.nan:
            st.warning(f"{r['name']} could not run on this dataset.")
            continue
        # Build bands with same confidence for visual parity
        try:
            b = make_bands(r["fc"], float(r["resid_std"]), conf)
        except Exception:
            b = pd.DataFrame(index=r["fc"].index)
            b[f"lower_{int(conf*100)}"] = r["fc"].values
            b[f"upper_{int(conf*100)}"] = r["fc"].values

        # chart: last 50 days hist + this model's forecast
        hist_tail = y.iloc[-50:] if len(y) > 50 else y
        hist_df = pd.DataFrame({"Date": hist_tail.index, "Price": hist_tail.values})
        f_df = pd.DataFrame({"Date": r["fc"].index, "Forecast": r["fc"].values})
        b_df = b.copy(); b_df["Date"] = b_df.index
        lower_col = [c for c in b_df.columns if c.startswith("lower_")][0]
        upper_col = [c for c in b_df.columns if c.startswith("upper_")][0]

        base = alt.Chart(hist_df).mark_line().encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title=f"Price ({currency})"),
            tooltip=["Date:T", "Price:Q"],
        ).properties(title=f"{r['name']} — {int(conf*100)}% band")

        fc_line = alt.Chart(f_df).mark_line(color="#7c3aed").encode(
            x="Date:T", y="Forecast:Q", tooltip=["Date:T","Forecast:Q"]
        )

        band = alt.Chart(b_df).mark_area(opacity=0.15, color="#7c3aed").encode(
            x="Date:T", y=f"{lower_col}:Q", y2=f"{upper_col}:Q"
        )

        st.altair_chart(alt.layer(base, band, fc_line).interactive(), use_container_width=True)

        # small metrics row
        m = r["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{m['MAE']:.4f}" if not np.isnan(m["MAE"]) else "—")
        c2.metric("RMSE", f"{m['RMSE']:.4f}" if not np.isnan(m["RMSE"]) else "—")
        c3.metric("MAPE", f"{m['MAPE%']:.2f}%" if not np.isnan(m["MAPE%"]) else "—")

# ---------------- Download ----------------
out = pd.DataFrame({"Date": fc.index, "Forecast": fc.values, lower_col: band_df[lower_col].values, upper_col: band_df[upper_col].values})
buf = BytesIO(); out.to_csv(buf, index=False)
st.download_button("📥 Download forecast CSV", data=buf.getvalue(),
                   file_name=f"{ticker}_forecast_{best_name.replace(' ','_')}.csv", mime="text/csv")

# ---------------- Navigation ----------------
cols = st.columns(2)
if cols[0].button("↩ Back to main"):
    for k in ("base_df","base_meta","nav_to_features","nav_to_forecast"): st.session_state.pop(k, None)
    try: st.switch_page("Home.py")
    except Exception: st.rerun()

if cols[1].button("🧪 Go to Feature Engineering"):
    try: st.switch_page("pages/03_Feature_Engineering.py")
    except Exception: st.rerun()

# ---------------- Disclaimer ----------------
st.caption("This tool is for educational purposes only and not investment advice. Forecasts are uncertain by nature.")
