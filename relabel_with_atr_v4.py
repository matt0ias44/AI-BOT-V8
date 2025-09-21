import pandas as pd
import numpy as np

# =====================
# Config (V4)
# =====================
IN_CSV  = "training_dataset_v2.csv"        # news + returns
BTC_5M  = "btcusdt_5m_sample.csv"          # bougies 5m (3 ans)
OUT_CSV = "training_dataset_v4.csv"

ATR_WINDOW     = 6      # 6 * 5m = 30 min
THRESH_COEFF   = 0.60   # <-- V4 : plus permissif que 0.8
MIN_MOVE_COEFF = 0.10   # drop si |ret| < 0.1 * atr_norm

print("[i] Loading...")
df = pd.read_csv(IN_CSV)
px = pd.read_csv(BTC_5M)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
px["timestamp"] = pd.to_datetime(px["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
px = px.dropna(subset=["timestamp","high","low","close"])

# ATR normalisé (True Range)
px = px.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
high, low, close = px["high"].astype(float), px["low"].astype(float), px["close"].astype(float)
prev_close = close.shift(1)
tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean()
px["atr_norm"] = (atr/close).clip(lower=1e-9)

# map atr_norm aux news (asof)
news = df.sort_values("timestamp").copy()
atr_series = px[["timestamp","atr_norm"]].dropna().sort_values("timestamp")
news = pd.merge_asof(news, atr_series, on="timestamp", direction="backward", allow_exact_matches=True)

def label_adaptive(ret, atrn, c=THRESH_COEFF):
    if pd.isna(ret) or pd.isna(atrn): return "unknown"
    if ret >  c*atrn:  return "bullish"
    if ret < -c*atrn:  return "bearish"
    return "neutral"

print("[i] Labeling V4 (ATR-adaptive)...")
news["label_30m_atr"] = [label_adaptive(r,a) for r,a in zip(news.get("return_30m"), news.get("atr_norm"))]

# filtre ultra-plat
mask = news["return_30m"].abs() >= (MIN_MOVE_COEFF * news["atr_norm"])
print(f"[i] Keep {int(mask.sum())}/{len(news)} rows after flat filter")
news = news[mask].copy()

cols = [c for c in [
    "timestamp","title","score","symbol","price_now",
    "return_30m","return_60m","return_120m",
    "atr_norm","label_30m_atr"
] if c in news.columns]

news[cols].to_csv(OUT_CSV, index=False)
print(f"✅ Exported {OUT_CSV} ({len(news)} rows)")
