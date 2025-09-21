#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ajoute des features de marché à t0 pour chaque événement (sans fuite).
Réutilisable comme fonction dans live_pipeline.
"""

import pandas as pd
import numpy as np


def as_utc(s):
    s = pd.to_datetime(s, utc=True, errors="coerce")
    if s.isna().any():
        raise ValueError("Erreur de parsing des timestamps.")
    return s


def rsi(series, period=14):
    delta = series.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def realized_vol(ret1m, window):
    return ret1m.rolling(window).std()


def true_range(df):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr


def attach_features_to_df(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features marché (RSI, ATR, vol, SMA…) à un dataframe d'événements."""
    ev = ev.copy()
    ev["event_time"] = as_utc(ev["event_time"])
    px["timestamp"] = as_utc(px["timestamp"])
    px = px.sort_values("timestamp").reset_index(drop=True).set_index("timestamp")

    # timeline complète en 1m
    full_idx = pd.date_range(px.index.min(), px.index.max(), freq="1min", tz="UTC")
    px = px.reindex(full_idx).ffill()

    # Retours
    px["ret1m"] = np.log(px["close"] / px["close"].shift(1))

    # Volatilité réalisée
    for w in [30, 60, 120]:
        px[f"vol_{w}m"] = realized_vol(px["ret1m"], w)
        px[f"ret_{w}m_back"] = np.log(px["close"] / px["close"].shift(w))

    # RSI
    px["rsi_14"] = rsi(px["close"], period=14)

    # ATR
    tr = true_range(px)
    for w in [30, 60, 120]:
        px[f"atr_{w}m"] = tr.rolling(w).mean()

    # Z-score volume
    px["vol_z_60m"] = (px["volume"] - px["volume"].rolling(60).mean()) / (px["volume"].rolling(60).std() + 1e-12)

    # Moyennes mobiles
    px["sma_10"] = px["close"].rolling(10).mean()
    px["sma_50"] = px["close"].rolling(50).mean()
    px["sma10_sma50_diff"] = (px["sma_10"] - px["sma_50"]) / (px["sma_50"] + 1e-12)

    feat_cols = [c for c in px.columns if c not in ["open", "high", "low", "close", "volume", "ret1m"]]

    def get_feats_at(t):
        try:
            return px.loc[:t].iloc[-1][feat_cols]
        except Exception:
            return pd.Series({c: np.nan for c in feat_cols})

    feats = pd.DataFrame([get_feats_at(t) for t in ev["event_time"].tolist()]).reset_index(drop=True)
    feats.columns = [f"feat_{c}" for c in feats.columns]

    out = pd.concat([ev.reset_index(drop=True), feats], axis=1)
    return out


if __name__ == "__main__":
    EVENTS_FILE = "events_with_magnitude.csv"
    PRICE_FILE = "btcusdt_1m_full.csv"
    OUT_FILE = "events_with_features.csv"

    print("[i] Chargement...")
    ev = pd.read_csv(EVENTS_FILE)
    px = pd.read_csv(PRICE_FILE)

    out = attach_features_to_df(ev, px)
    out.to_csv(OUT_FILE, index=False)
    print(f"[ok] écrit {OUT_FILE}")
