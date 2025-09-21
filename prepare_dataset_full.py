#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline complet de préparation dataset pour le modèle BTC news -> signal
Inclut :
1. Récupération / enrichissement articles (fetch_article_bodies)
2. Nettoyage + prétokenisation (pretokenize_datasetv2)
3. Génération des labels multi-horizons (30/60/120 min) avec magnitude
4. Ajout de features marché (volatilité, RSI, ATR…)
5. Sauvegarde dataset final prêt pour l’entraînement
"""

import os, re, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# 1) Fetch article bodies
# =========================
def fetch_article_bodies(news_path: str, out_path: str):
    try:
        import trafilatura
    except ImportError:
        raise RuntimeError("⚠️ Installe `trafilatura`: pip install trafilatura")

    df = pd.read_csv(news_path)
    if "url" not in df.columns:
        print("⚠️ Pas de colonne url, skip enrichissement")
        df.to_csv(out_path, index=False)
        return df

    texts = []
    for i, url in enumerate(df["url"].fillna("")):
        body = ""
        if isinstance(url, str) and url.startswith("http"):
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    body = trafilatura.extract(downloaded) or ""
            except Exception as e:
                print(f"[fetch] {i}/{len(df)} fail {url}: {e}")
        texts.append(body)
    df["body"] = texts
    df.to_csv(out_path, index=False)
    print(f"[fetch] Sauvegardé {out_path} ({len(df)} lignes)")
    return df

# =========================
# 2) Pretokenize
# =========================
def pretokenize_dataset(df: pd.DataFrame, out_path: str):
    def clean_text(txt: str) -> str:
        if not isinstance(txt, str): return ""
        txt = txt.lower()
        txt = re.sub(r"http\S+", "", txt)
        txt = re.sub(r"[^a-z0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    df["text_clean"] = (df["title"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)
    df.to_csv(out_path, index=False)
    print(f"[pretok] Sauvegardé {out_path}")
    return df

# =========================
# 3) Labels multi-horizons
# =========================
def make_event_labels_multi(news_df: pd.DataFrame, price_path: str, out_path: str,
                            horizons=[30,60,120], ret_thr=0.002):
    price = pd.read_csv(price_path, parse_dates=["dt"])
    price.set_index("dt", inplace=True)

    all_labels = {f"label_{h}": [] for h in horizons}
    all_mags = {f"mag_{h}": [] for h in horizons}

    for _, row in news_df.iterrows():
        try:
            ts = pd.to_datetime(row.get("published_at") or row.get("date"), utc=True)
        except:
            ts = None
        if ts is None or ts not in price.index:
            for h in horizons:
                all_labels[f"label_{h}"].append("neutral")
                all_mags[f"mag_{h}"].append(0.0)
            continue

        p0 = price.loc[ts, "close"]

        for h in horizons:
            t1 = ts + timedelta(minutes=h)
            if t1 not in price.index:
                all_labels[f"label_{h}"].append("neutral")
                all_mags[f"mag_{h}"].append(0.0)
                continue
            p1 = price.loc[t1, "close"]
            ret = (p1 / p0) - 1.0
            if ret > ret_thr:
                lab = "bullish"
            elif ret < -ret_thr:
                lab = "bearish"
            else:
                lab = "neutral"
            all_labels[f"label_{h}"].append(lab)
            all_mags[f"mag_{h}"].append(ret)

    for k,v in all_labels.items(): news_df[k] = v
    for k,v in all_mags.items(): news_df[k] = v

    news_df.to_csv(out_path, index=False)
    print(f"[labels] Multi-horizons sauvé {out_path}")
    return news_df

# =========================
# 4) Features marché
# =========================
def attach_market_features(news_df: pd.DataFrame, price_path: str, out_path: str):
    price = pd.read_csv(price_path, parse_dates=["dt"])
    price.set_index("dt", inplace=True)

    feats = []
    for _, row in news_df.iterrows():
        try:
            ts = pd.to_datetime(row.get("published_at") or row.get("date"), utc=True)
        except:
            ts = None
        if ts is None or ts not in price.index:
            feats.append({})
            continue

        window = price.loc[:ts].tail(120)  # 120 * 5m = 10h
        if len(window) < 20:
            feats.append({})
            continue

        logrets = np.diff(np.log(window["close"]))

        def rsi(prices, period=14):
            deltas = np.diff(prices)
            if len(deltas) < period: return np.nan
            seed = deltas[:period]
            up = seed[seed >= 0].sum()/period
            down = -seed[seed < 0].sum()/period
            rs = up/down if down > 0 else 0
            rsi = 100. - 100./(1.+rs)
            return rsi

        feats.append({
            "feat_vol_30m": np.std(logrets[-6:]) if len(logrets) >= 6 else np.nan,
            "feat_ret_30m_back": (window["close"].iloc[-1]/window["close"].iloc[-6]-1) if len(window) >= 6 else np.nan,
            "feat_vol_60m": np.std(logrets[-12:]) if len(logrets) >= 12 else np.nan,
            "feat_ret_60m_back": (window["close"].iloc[-1]/window["close"].iloc[-12]-1) if len(window) >= 12 else np.nan,
            "feat_vol_120m": np.std(logrets[-24:]) if len(logrets) >= 24 else np.nan,
            "feat_ret_120m_back": (window["close"].iloc[-1]/window["close"].iloc[-24]-1) if len(window) >= 24 else np.nan,
            "feat_rsi_14": rsi(window["close"].values, 14),
            "feat_vol_z_12": (window["volume"].iloc[-1]-window["volume"].tail(12).mean())/ (window["volume"].tail(12).std()+1e-9),
        })

    feat_df = pd.DataFrame(feats)
    out_df = pd.concat([news_df.reset_index(drop=True), feat_df], axis=1)
    out_df.to_csv(out_path, index=False)
    print(f"[features] Sauvegardé {out_path}")
    return out_df

# =========================
# MAIN PIPELINE
# =========================
def prepare_dataset(news_csv="btc_news_dataset.csv",
                    price_csv="btcusdt_5m.csv",
                    out_csv="events_final.csv"):

    tmp1, tmp2, tmp3 = "news_with_bodies.csv", "news_pretok.csv", "news_labeled.csv"

    # 1. fetch bodies
    df1 = fetch_article_bodies(news_csv, tmp1)

    # 2. pretokenize
    df2 = pretokenize_dataset(df1, tmp2)

    # 3. labels multi-horizons
    df3 = make_event_labels_multi(df2, price_csv, tmp3)

    # 4. market features
    df4 = attach_market_features(df3, price_csv, out_csv)

    print(f"[DONE] Dataset final prêt: {out_csv} ({len(df4)} lignes)")
    return df4


if __name__ == "__main__":
    prepare_dataset()
