#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live trader (paper trading).
- Watches live_predictions.csv (produced by bridge_inference.py)
- Reads latest BTC price from price_pipe.csv or Binance fallback
- Converts predictions into trades with dynamic sizing and risk settings
- Persists state in bot_state.json (consumed by Streamlit dashboard)
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

PRED_FILE = Path("live_predictions.csv")
PRICE_FILE = Path("price_pipe.csv")
STATE_FILE = Path("bot_state.json")
MODEL_DIR = Path("./models/bert_v7_1_multi")
REFRESH_SEC = 2

CONF_OPEN = 0.62
CONF_CLOSE = 0.55
MIN_COOLDOWN_SEC = 60
BASE_POSITION_USD = 1000
MAX_LEVERAGE = 8
SL_FLOOR = 0.003  # 0.3%
TP_MULTIPLIER = 1.4
SL_MULTIPLIER = 0.9

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

LABELS = ["bearish", "neutral", "bullish"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_last_price() -> Optional[float]:
    if PRICE_FILE.exists():
        try:
            with PRICE_FILE.open("r", newline="", encoding="utf-8") as fh:
                last = None
                for row in csv.reader(fh):
                    if len(row) >= 2:
                        try:
                            last = float(row[1])
                        except ValueError:
                            continue
                if last is not None:
                    return last
        except Exception as exc:
            print(f"[WARN] unable to read {PRICE_FILE}: {exc}")

    # fallback Binance
    try:
        resp = requests.get(BINANCE_PRICE_URL, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("price"))
    except Exception as exc:
        print(f"[WARN] Binance price fetch failed: {exc}")
        return None


def init_state() -> Dict:
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[WARN] unable to read {STATE_FILE}: {exc}")
    return {
        "starting_equity": 10000.0,
        "equity": 10000.0,
        "position": None,
        "trades": [],
        "equity_curve": [[now_iso(), 10000.0]],
        "last_pred_id": None,
    }


def save_state(state: Dict) -> None:
    if len(state.get("equity_curve", [])) > 5000:
        state["equity_curve"] = state["equity_curve"][-5000:]
    if len(state.get("trades", [])) > 2000:
        state["trades"] = state["trades"][-2000:]
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def load_thresholds() -> Optional[Dict[str, Dict[str, float]]]:
    path = MODEL_DIR / "thresholds_mag.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else None
    except Exception as exc:
        print(f"[WARN] unable to read {path}: {exc}")
        return None


def compute_leverage(mag_value: float, thresholds: Optional[Dict[str, Dict[str, float]]]) -> int:
    if thresholds and "60" in thresholds:
        q = thresholds["60"]
        q2 = max(q.get("q2", 0.001), 1e-6)
        scaled = mag_value / q2
        leverage = 1 + int(round(min(MAX_LEVERAGE - 1, max(0.0, scaled * 4))))
        return max(1, min(MAX_LEVERAGE, leverage))
    return 1


def compute_tp_sl(mag_value: float, side: str) -> Dict[str, float]:
    target = max(SL_FLOOR, mag_value)
    tp_pct = target * TP_MULTIPLIER
    sl_pct = max(SL_FLOOR, target * SL_MULTIPLIER)
    return {
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
    }


def latest_predictions(n: int = 40) -> pd.DataFrame:
    if not PRED_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PRED_FILE)
    except Exception as exc:
        print(f"[WARN] unable to read {PRED_FILE}: {exc}")
        return pd.DataFrame()
    required = {
        "news_id",
        "datetime_utc",
        "prediction",
        "prob_bear",
        "prob_neut",
        "prob_bull",
        "confidence",
        "mag_pred",
        "mag_bucket",
    }
    for col in required:
        if col not in df.columns:
            df[col] = None
    return df.tail(n)


def close_position(state: Dict, price: float, reason: str = "CLOSE") -> None:
    pos = state.get("position")
    if not pos:
        return
    side = pos["side"]
    entry = float(pos["entry"])
    size = float(pos["size"])
    direction = -1.0 if side == "short" else 1.0
    pnl = direction * (price - entry) * (size / max(entry, 1e-9))
    state["equity"] = float(state["equity"]) + pnl
    state["trades"].append(
        {
            "time": now_iso(),
            "action": reason if reason.startswith("CLOSE") else f"CLOSE {side.upper()}",
            "price": round(price, 2),
            "size_usd": size,
            "pnl": round(pnl, 2),
            "title": pos.get("title", ""),
            "leverage": pos.get("leverage", 1),
        }
    )
    state["position"] = None


def open_position(state: Dict, side: str, price: float, title: str, leverage: int, mag_value: float) -> None:
    tp_sl = compute_tp_sl(mag_value, side)
    tp = price * (1 + tp_sl["tp_pct"]) if side == "long" else price * (1 - tp_sl["tp_pct"])
    sl = price * (1 - tp_sl["sl_pct"]) if side == "long" else price * (1 + tp_sl["sl_pct"])
    size = BASE_POSITION_USD * leverage
    state["position"] = {
        "side": side,
        "entry": float(price),
        "size": float(size),
        "entry_ts": now_iso(),
        "tp": float(tp),
        "sl": float(sl),
        "title": title[:160],
        "leverage": leverage,
        "mag_pred": mag_value,
    }
    state["trades"].append(
        {
            "time": now_iso(),
            "action": f"OPEN {side.upper()}",
            "price": round(price, 2),
            "size_usd": float(size),
            "pnl": 0.0,
            "title": title[:160],
            "leverage": leverage,
        }
    )


def mark_to_market(state: Dict, price: Optional[float]) -> None:
    if price is None:
        state["equity_curve"].append([now_iso(), state["equity"]])
        return
    pos = state.get("position")
    if pos:
        side = pos["side"]
        entry = pos["entry"]
        size = pos["size"]
        direction = -1.0 if side == "short" else 1.0
        unrealized = direction * (price - entry) * (size / max(entry, 1e-9))
        equity = state["starting_equity"] + unrealized
    else:
        equity = state["equity"]
    state["equity_curve"].append([now_iso(), equity])


def main():
    print("[LIVE] trader started")
    state = init_state()
    thresholds = load_thresholds()
    last_action_ts = 0.0
    seen_ids = set()

    while True:
        time.sleep(REFRESH_SEC)
        price = read_last_price()

        if price and state.get("position"):
            pos = state["position"]
            side = pos["side"]
            hit_tp = (side == "long" and price >= pos["tp"]) or (side == "short" and price <= pos["tp"])
            hit_sl = (side == "long" and price <= pos["sl"]) or (side == "short" and price >= pos["sl"])
            if hit_tp or hit_sl:
                reason = "CLOSE TP" if hit_tp else "CLOSE SL"
                close_position(state, price, reason=reason)

        df = latest_predictions()
        if not df.empty:
            last_row = df.iloc[-1]
            news_id = str(last_row.get("news_id", ""))
            if news_id and news_id not in seen_ids:
                seen_ids.add(news_id)
                state["last_pred_id"] = news_id
                pred = str(last_row.get("prediction", "neutral")).lower()
                confidence = float(last_row.get("confidence", 0.0) or 0.0)
                mag_value = float(last_row.get("mag_pred", 0.0) or 0.0)
                title = str(last_row.get("title", ""))

                now_s = time.time()
                if now_s - last_action_ts >= MIN_COOLDOWN_SEC:
                    if state.get("position") is None:
                        if pred == "bullish" and confidence >= CONF_OPEN and price:
                            lev = compute_leverage(mag_value, thresholds)
                            open_position(state, "long", price, title, lev, mag_value)
                            last_action_ts = now_s
                        elif pred == "bearish" and confidence >= CONF_OPEN and price:
                            lev = compute_leverage(mag_value, thresholds)
                            open_position(state, "short", price, title, lev, mag_value)
                            last_action_ts = now_s
                    else:
                        pos = state["position"]
                        side = pos["side"]
                        if pred == "neutral" and confidence >= CONF_CLOSE and price:
                            close_position(state, price, reason="CLOSE neutral")
                            last_action_ts = now_s
                        elif side == "long" and pred == "bearish" and confidence >= CONF_OPEN and price:
                            close_position(state, price, reason="CLOSE flip")
                            lev = compute_leverage(mag_value, thresholds)
                            open_position(state, "short", price, title, lev, mag_value)
                            last_action_ts = now_s
                        elif side == "short" and pred == "bullish" and confidence >= CONF_OPEN and price:
                            close_position(state, price, reason="CLOSE flip")
                            lev = compute_leverage(mag_value, thresholds)
                            open_position(state, "long", price, title, lev, mag_value)
                            last_action_ts = now_s

        mark_to_market(state, price)
        save_state(state)


if __name__ == "__main__":
    main()
