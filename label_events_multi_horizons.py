import pandas as pd
import numpy as np

# paramètres
BTC_FILE = "btcusdt_5m_full.csv"
EVENTS_FILE = "btc_events.csv"
OUT_FILE = "btc_events_labeled_multi.csv"

RETURN_HORIZONS = [30, 60, 120]   # minutes
THRESH = 0.1  # seuil = 0.1 * ATR

def main():
    print("[i] Chargement des événements...")
    events = pd.read_csv(EVENTS_FILE)
    print("[i] Colonnes événements:", list(events.columns))

    print("[i] Chargement des prix BTC...")
    btc = pd.read_csv(BTC_FILE)
    print("[i] Colonnes BTC:", list(btc.columns))

    # parsing des timestamps
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
    btc = btc.set_index("timestamp").sort_index()

    # ⚡ fix: supprimer les doublons de timestamp
    btc = btc[~btc.index.duplicated(keep="first")]

    # calcul ATR (simplifié)
    btc["hl"] = btc["high"] - btc["low"]
    btc["hc"] = (btc["high"] - btc["close"].shift()).abs()
    btc["lc"] = (btc["low"] - btc["close"].shift()).abs()
    tr = btc[["hl", "hc", "lc"]].max(axis=1)
    btc["atr"] = tr.rolling(30).mean()

    # normalisation par ATR
    btc["return"] = btc["close"].pct_change()
    btc["ret_norm"] = btc["return"] / btc["atr"].pct_change().replace(0, np.nan)

    # initialisation des labels multi-horizons
    for h in RETURN_HORIZONS:
        events[f"label_{h}m"] = "neutral"

    # boucle sur les événements
    for i, row in events.iterrows():
        t0 = pd.to_datetime(row["first_timestamp"], utc=True)

        if t0 not in btc.index:
            continue

        idx = btc.index.get_indexer([t0])[0]

        for h in RETURN_HORIZONS:
            if idx + h >= len(btc):
                continue

            p0 = btc.iloc[idx]["close"]
            p1 = btc.iloc[idx + h]["close"]
            atr_now = btc.iloc[idx]["atr"]

            if pd.isna(atr_now) or atr_now == 0:
                continue

            ret_norm = (p1 - p0) / p0 / (atr_now / p0)

            if ret_norm > THRESH:
                events.at[i, f"label_{h}m"] = "bullish"
            elif ret_norm < -THRESH:
                events.at[i, f"label_{h}m"] = "bearish"
            else:
                events.at[i, f"label_{h}m"] = "neutral"

    print("[i] Distribution labels :")
    for h in RETURN_HORIZONS:
        print(f"{h}m ->", events[f"label_{h}m"].value_counts().to_dict())

    events.to_csv(OUT_FILE, index=False)
    print(f"✅ Export terminé : {OUT_FILE}")

if __name__ == "__main__":
    main()
