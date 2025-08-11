# pages/03_Forecast.py
import os
from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")
st.title("🔮 Forecast")

# ---- Optional deps (flags + safe imports) ----
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

HAS_SKLEARN = True
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances_argmin_min
except Exception:
    HAS_SKLEARN = False

HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
except Exception:
    HAS_TORCH = False

# ---------- Get input data ----------
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
    st.warning("No input data found. Go back to the main page, fetch data, then click **Proceed**.")
    st.page_link("streamlit_app.py", label="↩ Back to main", icon="↩️")
    st.stop()

price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
if price_col is None:
    st.error("Neither **Adj Close** nor **Close** was found in the data.")
    st.stop()

data = df.copy().sort_values("Date").reset_index(drop=True)
data = data[["Date", price_col]].rename(columns={price_col: "Price"})
data["Date"] = pd.to_datetime(data["Date"])

ticker = meta.get("ticker", "TICKER")
currency = meta.get("currency", "USD")
st.caption(f"Ticker: **{ticker}** • Currency: **{currency}** • Series: **{price_col}**")

# ---------- Sidebar ----------
st.sidebar.header("Forecast Settings")
H = st.sidebar.number_input("Forecast horizon (days)", 1, 365*2, 60, 1)

model_options = ["Naive (last value)", "Moving Average"]
if HAS_STATSMODELS:
    model_options += ["Exponential Smoothing (Holt-Winters)", "SARIMAX (ARIMA)"]
if HAS_XGB:
    model_options += ["XGBoost (windowed)"]
if HAS_SKLEARN:
    model_options += ["Agglomerative clustering (pattern)"]
if HAS_TORCH:
    model_options += ["Transformer (PyTorch)", "LSTM (PyTorch)"]

MODEL = st.sidebar.selectbox("Model", model_options)

# Model params
if MODEL == "Moving Average":
    ma_window = st.sidebar.number_input("MA window", 2, 365, 20, 1)
elif MODEL == "Exponential Smoothing (Holt-Winters)" and HAS_STATSMODELS:
    seasonal = st.sidebar.selectbox("Seasonality", ["None", "Additive", "Multiplicative"], 0)
    seasonal_periods = st.sidebar.number_input("Seasonal period (e.g., 5 for ~weekly trading)", 0, 365, 0, 1)
    trend = st.sidebar.selectbox("Trend", ["None", "Additive", "Multiplicative"], 1)
elif MODEL == "SARIMAX (ARIMA)" and HAS_STATSMODELS:
    p = st.sidebar.number_input("p", 0, 5, 1); d = st.sidebar.number_input("d", 0, 2, 1); q = st.sidebar.number_input("q", 0, 5, 1)
    use_seasonal = st.sidebar.checkbox("Seasonal", False)
    if use_seasonal:
        P = st.sidebar.number_input("P", 0, 3, 0); D = st.sidebar.number_input("D", 0, 2, 0)
        Q = st.sidebar.number_input("Q", 0, 3, 0); s = st.sidebar.number_input("Seasonal period (s)", 2, 365, 5)
elif MODEL == "XGBoost (windowed)" and HAS_XGB:
    xgb_lags = st.sidebar.number_input("Lag window (steps)", 5, 365, 30, 1)
    xgb_estimators = st.sidebar.number_input("n_estimators", 50, 2000, 400, 50)
    xgb_depth = st.sidebar.number_input("max_depth", 1, 12, 4, 1)
    xgb_lr = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.05, 0.01)
elif MODEL == "Agglomerative clustering (pattern)" and HAS_SKLEARN:
    agg_window = st.sidebar.number_input("Pattern window (steps)", 5, 200, 30, 1)
    agg_clusters = st.sidebar.number_input("Clusters (K)", 2, 40, 8, 1)
elif MODEL in ["Transformer (PyTorch)", "LSTM (PyTorch)"] and HAS_TORCH:
    nn_seq = st.sidebar.number_input("Input sequence length", 5, 120, 30, 1)
    nn_epochs = st.sidebar.number_input("Epochs", 1, 300, 30, 1)
    nn_hidden = st.sidebar.number_input("Hidden size", 8, 256, 32, 1)

# ---------- Helpers ----------
def _prepare_index(df: pd.DataFrame) -> pd.Series:
    out = df.dropna(subset=["Price"]).sort_values("Date").set_index("Date")["Price"]
    full_idx = pd.date_range(start=out.index.min(), end=out.index.max(), freq="D")
    return out.reindex(full_idx).ffill()

def _z(alpha: float) -> float:
    return {0.20: 1.28155, 0.10: 1.64485, 0.05: 1.95996}.get(alpha, 1.95996)

def naive_forecast(y: pd.Series, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
    last = y.iloc[-1]
    idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
    fc = pd.Series(last, index=idx, name="Forecast")
    diffs = y.diff().dropna()
    sigma = diffs.std(ddof=1) if len(diffs) > 1 else (y.pct_change().std(ddof=1) * abs(last))
    bands = {f"lower_{p}": fc - _z(1 - int(p)/100) * sigma for p in (80, 95)}
    bands.update({f"upper_{p}": fc + _z(1 - int(p)/100) * sigma for p in (80, 95)})
    return fc, pd.DataFrame(bands, index=idx)

def ma_forecast(y: pd.Series, steps: int, window: int) -> Tuple[pd.Series, pd.DataFrame]:
    y_ma = y.rolling(window).mean().dropna()
    last = y_ma.iloc[-1] if not y_ma.empty else y.iloc[-window:].mean()
    idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
    fc = pd.Series(last, index=idx, name="Forecast")
    resid = (y - y.rolling(window).mean()).dropna()
    sigma = resid.std(ddof=1) if len(resid) > 1 else (y - y.mean()).std(ddof=1)
    bands = {f"lower_{p}": fc - _z(1 - int(p)/100) * sigma for p in (80, 95)}
    bands.update({f"upper_{p}": fc + _z(1 - int(p)/100) * sigma for p in (80, 95)})
    return fc, pd.DataFrame(bands, index=idx)

def ets_forecast(y: pd.Series, steps: int, trend_opt: str, seasonal_opt: str, sp: int):
    model = ExponentialSmoothing(y, trend=None if trend_opt=="None" else trend_opt.lower(),
                                 seasonal=None if seasonal_opt=="None" else seasonal_opt.lower(),
                                 seasonal_periods=sp if seasonal_opt!="None" else None,
                                 initialization_method="estimated")
    res = model.fit(optimized=True)
    fc = res.forecast(steps).rename("Forecast")
    resid = res.resid.dropna()
    sigma = resid.std(ddof=1) if len(resid) > 1 else y.std(ddof=1)
    bands = {f"lower_{p}": fc - _z(1 - int(p)/100) * sigma for p in (80, 95)}
    bands.update({f"upper_{p}": fc + _z(1 - int(p)/100) * sigma for p in (80, 95)})
    return fc, pd.DataFrame(bands, index=fc.index)

def sarimax_forecast(y: pd.Series, steps: int, order, seasonal_order=None):
    res = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc_obj = res.get_forecast(steps=steps)
    fc = fc_obj.predicted_mean.rename("Forecast")
    ci80 = fc_obj.conf_int(alpha=0.20); ci95 = fc_obj.conf_int(alpha=0.05)
    bands = pd.DataFrame({
        "lower_80": ci80.iloc[:, 0], "upper_80": ci80.iloc[:, 1],
        "lower_95": ci95.iloc[:, 0], "upper_95": ci95.iloc[:, 1]
    }, index=fc.index)
    return fc, bands

def make_supervised(series: np.ndarray, lags: int):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i]); y.append(series[i])
    return np.array(X), np.array(y)

def xgb_forecast(y: pd.Series, steps: int, lags: int, n_estimators: int, max_depth: int, lr: float):
    if len(y) <= lags + 5: raise ValueError("Not enough data for the chosen lag window.")
    arr = y.values.astype(np.float32)
    X, Y = make_supervised(arr, lags)
    model = xgb.XGBRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth),
                             learning_rate=float(lr), subsample=0.9, colsample_bytree=0.9,
                             objective="reg:squarederror", random_state=42, n_jobs=2)
    model.fit(X, Y)
    preds, history = [], arr.tolist()
    for _ in range(int(steps)):
        x_in = np.array(history[-lags:], dtype=np.float32).reshape(1, -1)
        yhat = float(model.predict(x_in)[0]); preds.append(yhat); history.append(yhat)
    idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
    fc = pd.Series(preds, index=idx, name="Forecast")
    train_pred = model.predict(X); sigma = float(np.std(Y - train_pred, ddof=1))
    bands = {f"lower_{p}": fc - _z(1 - int(p)/100) * sigma for p in (80, 95)}
    bands.update({f"upper_{p}": fc + _z(1 - int(p)/100) * sigma for p in (80, 95)})
    return fc, pd.DataFrame(bands, index=idx)

if HAS_SKLEARN:
    def agglomerative_forecast(y: pd.Series, steps: int, window: int, k: int):
        if len(y) <= window + 5: raise ValueError("Not enough data for the chosen pattern window.")
        vals = y.values.astype(float)
        X = np.lib.stride_tricks.sliding_window_view(vals, window+1)
        X = X[:-1]
        patterns = X[:, :window]
        next_delta = X[:, -1] - X[:, -2]
        ac = AgglomerativeClustering(n_clusters=int(k))
        labels = ac.fit_predict(patterns)
        centers = np.vstack([patterns[labels==c].mean(axis=0) for c in range(int(k))])
        deltas_mean = np.array([next_delta[labels==c].mean() for c in range(int(k))])
        deltas_std  = np.array([next_delta[labels==c].std(ddof=1) if np.sum(labels==c)>1 else np.std(next_delta) for c in range(int(k))])
        preds, last_seq = [], vals[-window:].copy()
        idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
        for _ in range(int(steps)):
            lbl, _ = pairwise_distances_argmin_min(last_seq.reshape(1,-1), centers)
            c = int(lbl[0]); yhat = last_seq[-1] + deltas_mean[c]
            preds.append(yhat); last_seq = np.roll(last_seq, -1); last_seq[-1] = yhat
        fc = pd.Series(preds, index=idx, name="Forecast")
        sigma = float(np.nanmean(deltas_std)) if np.isfinite(deltas_std).any() else float(np.std(next_delta))
        bands = {f"lower_{p}": fc - _z(1 - int(p)/100) * sigma for p in (80, 95)}
        bands.update({f"upper_{p}": fc + _z(1 - int(p)/100) * sigma for p in (80, 95)})
        return fc, pd.DataFrame(bands, index=idx)
else:
    def agglomerative_forecast(*args, **kwargs):
        raise RuntimeError("scikit-learn not installed")

# ---- Torch models only if PyTorch is available ----
if HAS_TORCH:
    def torch_sequences(arr: np.ndarray, seq_len: int):
        X, Y = [], []
        for i in range(seq_len, len(arr)):
            X.append(arr[i-seq_len:i]); Y.append(arr[i])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)

    class TinyLSTM(nn.Module):
        def __init__(self, input_size=1, hidden=32, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x); return self.fc(out[:, -1, :])

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=1000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.pe = pe.unsqueeze(0)
        def forward(self, x):
            return x + self.pe[:, :x.size(1), :].to(x.device)

    class TinyTransformer(nn.Module):
        def __init__(self, d_model=32, nhead=4, num_layers=2):
            super().__init__()
            self.input = nn.Linear(1, d_model)
            self.pe = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)
        def forward(self, x):
            x = self.input(x); x = self.pe(x); x = self.enc(x); return self.fc(x[:, -1, :])

    def torch_forecast(y: pd.Series, steps: int, model_type: str, seq_len: int, hidden: int, epochs: int):
        device = torch.device("cpu")
        arr = y.values.astype(np.float32)
        mean, std = float(arr.mean()), float(arr.std() if arr.std() > 1e-8 else 1.0)
        norm = (arr - mean) / std
        Xn, Yn = torch_sequences(norm, seq_len)
        if len(Xn) < 10: raise ValueError("Not enough data for the chosen sequence length.")
        X = torch.tensor(Xn[:, :, None]).to(device); Y = torch.tensor(Yn).to(device)
        net = TinyLSTM(hidden=int(hidden), layers=2).to(device) if model_type=="LSTM (PyTorch)" else TinyTransformer(d_model=int(hidden)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3); loss_fn = nn.MSELoss()
        net.train()
        for _ in range(int(epochs)):
            opt.zero_grad(); pred = net(X); loss = loss_fn(pred, Y); loss.backward(); opt.step()
        net.eval(); seq = norm[-seq_len:].tolist(); preds = []
        with torch.no_grad():
            for _ in range(int(steps)):
                x = torch.tensor(seq, dtype=torch.float32).view(1, seq_len, 1).to(device)
                yhat = net(x).cpu().numpy().ravel()[0]; preds.append(yhat); seq = seq[1:] + [yhat]
        preds = np.array(preds) * std + mean
        idx = pd.date_range(y.index[-1], periods=steps+1, freq="D")[1:]
        fc = pd.Series(preds, index=idx, name="Forecast")
        with torch.no_grad():
            tr = net(X).cpu().numpy().ravel()
        sigma = float(np.std((tr * std + mean) - (Y.cpu().numpy().ravel() * std + mean), ddof=1))
        bands = {f"lower_{p}": fc - _z(1 - int(p)/100) * sigma for p in (80, 95)}
        bands.update({f"upper_{p}": fc + _z(1 - int(p)/100) * sigma for p in (80, 95)})
        return fc, pd.DataFrame(bands, index=idx)
else:
    def torch_forecast(*args, **kwargs):
        raise RuntimeError("PyTorch not installed")

# -------- Dispatcher --------
def forecast_series(y: pd.Series, steps: int):
    if MODEL == "Naive (last value)":
        return naive_forecast(y, steps)
    if MODEL == "Moving Average":
        return ma_forecast(y, steps, int(ma_window))
    if MODEL == "Exponential Smoothing (Holt-Winters)" and HAS_STATSMODELS:
        sp = int(seasonal_periods) if 'seasonal_periods' in locals() else 0
        return ets_forecast(y, steps, trend, seasonal, sp)
    if MODEL == "SARIMAX (ARIMA)" and HAS_STATSMODELS:
        if 'use_seasonal' in locals() and use_seasonal:
            return sarimax_forecast(y, steps, (int(p), int(d), int(q)), (int(P), int(D), int(Q), int(s)))
        else:
            return sarimax_forecast(y, steps, (int(p), int(d), int(q)), (0, 0, 0, 0))
    if MODEL == "XGBoost (windowed)" and HAS_XGB:
        return xgb_forecast(y, steps, int(xgb_lags), int(xgb_estimators), int(xgb_depth), float(xgb_lr))
    if MODEL == "Agglomerative clustering (pattern)" and HAS_SKLEARN:
        return agglomerative_forecast(y, steps, int(agg_window), int(agg_clusters))
    if MODEL in ["Transformer (PyTorch)", "LSTM (PyTorch)"] and HAS_TORCH:
        return torch_forecast(y, steps, MODEL, int(nn_seq), int(nn_hidden), int(nn_epochs))
    raise ValueError("Selected model requires an optional package not installed.")

# ---------- Build forecast ----------
try:
    y = _prepare_index(data)
    if len(y) < 20:
        st.warning("Not enough data points to forecast. Try a longer date range."); st.stop()

    fc, bands = forecast_series(y, int(H))

    # ---------- Plot (last 50 days only + forecast) ----------
    hist_tail = y.iloc[-500:] if len(y) > 500 else y
    hist_df = pd.DataFrame({"Date": hist_tail.index, "Price": hist_tail.values})
    fc_df = pd.DataFrame({"Date": fc.index, "Forecast": fc.values})
    band_df = bands.copy(); band_df["Date"] = band_df.index

    base = alt.Chart(hist_df).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Price:Q", title=f"Price ({currency})"),
        tooltip=["Date:T", "Price:Q"],
    ).properties(title=f"{ticker} — Forecast ({MODEL})")

    fc_line = alt.Chart(fc_df).mark_line(color="#d62728").encode(
        x="Date:T", y="Forecast:Q", tooltip=["Date:T", "Forecast:Q"]
    )

    layers = [base, fc_line]
    if {"lower_80","upper_80"}.issubset(band_df.columns):
        layers.append(alt.Chart(band_df).mark_area(opacity=0.25, color="#1f77b4").encode(
            x="Date:T", y="lower_80:Q", y2="upper_80:Q"))
    if {"lower_95","upper_95"}.issubset(band_df.columns):
        layers.append(alt.Chart(band_df).mark_area(opacity=0.15, color="#1f77b4").encode(
            x="Date:T", y="lower_95:Q", y2="upper_95:Q"))

    st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)

    # ---------- Download ----------
    out = pd.DataFrame({"Date": fc.index, "Forecast": fc.values})
    for col in ["lower_80","upper_80","lower_95","upper_95"]:
        if col in bands.columns: out[col] = bands[col].values
    buf = BytesIO(); out.to_csv(buf, index=False)
    st.download_button("📥 Download forecast CSV", data=buf.getvalue(),
                       file_name=f"{ticker}_forecast_{MODEL.replace(' ', '')}.csv", mime="text/csv")

    # ---------- Nav ----------
    cols = st.columns(2)
    if cols[0].button("↩ Back to main"):
        for k in ("base_df","base_meta","nav_to_features"): st.session_state.pop(k, None)
        try: st.switch_page("Home.py")
        except Exception: st.rerun()
        
    st.success("✅ Forecast ready.")

except Exception as e:
    st.error(f"Forecast failed: {e}")
    if not HAS_STATSMODELS: st.info("Install `statsmodels` for Holt-Winters/SARIMAX:  `pip install statsmodels`")
    if not HAS_XGB:        st.info("Install `xgboost` for XGBoost:  `pip install xgboost`")
    if not HAS_SKLEARN:    st.info("Install `scikit-learn` for Agglomerative:  `pip install scikit-learn`")
    if not HAS_TORCH:      st.info("Install PyTorch for Transformer/LSTM:  https://pytorch.org/get-started/locally/")
