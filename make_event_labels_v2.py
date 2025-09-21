#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_event_labels.py
Ajoute les labels directionnels et magnitude aux news (GDELT ou format interne).
"""

import argparse, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

# =====================
# Helpers
# =====================
def as_utc_series(s: pd.Series):
    return pd.to_datetime(s, utc=True, errors="coerce")

def compute_labels(ev_time, prices, horizons=[30,60,120]):
    """
    Calcule labels directionnels et magnitude sur plusieurs horizons.
    """
    out = {}
    for h in horizons:
        dt_future = ev_time + timedelta(minutes=h)
        # trouver la bougie la plus proche
        sub = prices.loc[prices["ts"] >= dt_future]
        if sub.empty:
            out[f"label_{h}m"] = np.nan
            out[f"mag_{h}m"] = np.nan
            continue
        p0 = float(prices.loc[prices["ts"] >= ev_time].iloc[0]["close"])
        p1 = float(sub.iloc[0]["close"])
        ret = (p1 - p0) / p0
        if ret > 0.001:
            lab = "bullish"
        elif ret < -0.001:
            lab = "bearish"
        else:
            lab = "neutral"
        out[f"label_{h}m"] = lab
        out[f"mag_{h}m"] = abs(ret)
    return out

# =====================
# Main
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="CSV des events (news GDELT ou interne)")
    ap.add_argument("--ohlcv", required=True, help="CSV OHLCV BTC 1m")
    ap.add_argument("--out", required=True, help="CSV de sortie avec labels")
    args = ap.parse_args()

    print("[i] Chargement des events...")
    ev = pd.read_csv(args.events)

    # Normalisation des colonnes
    if "titles_joined" not in ev.columns and "title" in ev.columns:
        ev["titles_joined"] = ev["title"]
    if "body_concat" not in ev.columns and "body" in ev.columns:
        ev["body_concat"] = ev["body"]

    if "event_time" not in ev.columns:
        raise ValueError("Le CSV d'events doit contenir 'event_time'")
    ev["event_time"] = as_utc_series(ev["event_time"])
    ev = ev.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    print("[i] Chargement OHLCV...")
    prices = pd.read_csv(args.ohlcv)
    if "open_time" in prices.columns:
        prices["ts"] = pd.to_datetime(prices["open_time"], unit="ms", utc=True)
        prices["close"] = prices["close"].astype(float)
    elif "ts" in prices.columns:
        prices["ts"] = pd.to_datetime(prices["ts"], utc=True)
    else:
        raise ValueError("OHLCV doit contenir 'open_time' ou 'ts'")

    # Calcul des labels
    print("[i] Calcul des labels...")
    labels = []
    for _, row in ev.iterrows():
        labs = compute_labels(row["event_time"], prices)
        labels.append(labs)
    lab_df = pd.DataFrame(labels)

    out = pd.concat([ev, lab_df], axis=1)
    out.to_csv(args.out, index=False)
    print(f"[ok] Sauvegardé -> {args.out}, {len(out)} lignes")

if __name__ == "__main__":
    main()
