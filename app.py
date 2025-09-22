#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit dashboard for the BTC sentiment trading bot."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import warnings
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pytz import timezone as pytz_timezone
from streamlit_autorefresh import st_autorefresh

from live.model_paths import resolve_model_dir

warnings.filterwarnings("ignore", message=".*Styler.applymap.*", category=FutureWarning)

TZ_PARIS = pytz_timezone("Europe/Paris")
APP_ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = APP_ROOT / "live_predictions.csv"
STATE_FILE = APP_ROOT / "bot_state.json"
MODEL_DIR = resolve_model_dir(
    base_dir=APP_ROOT,
    env_var="MODEL_DIR",
    candidates=("models/bert_v7_1_plus", "models/bert_v7_1_multi", "models/bert_v7_multi"),
)
LIVE_RAW_FILE = APP_ROOT / "live_raw.csv"

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
BINANCE_24H_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
REFRESH_SEC = 5

DEFAULT_STATE = {
    "starting_equity": 10000.0,
    "equity": 10000.0,
    "position": None,
    "trades": [],
    "equity_curve": [],
    "last_pred_id": None,
    "last_signal": None,
}


def humanize_delta(delta: timedelta) -> str:
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return "moins d'une minute"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    minutes %= 60
    if hours < 24:
        return f"{hours} h {minutes:02d}"
    days = hours // 24
    hours %= 24
    return f"{days} j {hours:02d} h"


def file_recent_status(path: Path, threshold_minutes: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "fichier manquant"
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(tz=timezone.utc) - mtime
    ok = age <= timedelta(minutes=threshold_minutes)
    detail = f"maj il y a {humanize_delta(age)}"
    return ok, detail


def check_rss_status() -> tuple[bool, str]:
    ok, detail = file_recent_status(LIVE_RAW_FILE, 30)
    return ok, detail


def check_model_status(preds: pd.DataFrame) -> tuple[bool, str]:
    if not MODEL_DIR.exists():
        return False, "répertoire modèle absent"
    if not PREDICTIONS_CSV.exists():
        return False, "live_predictions.csv manquant"
    if preds.empty:
        ok, detail = file_recent_status(PREDICTIONS_CSV, 30)
        return ok, detail + " (aucune ligne)"
    if "datetime_utc" in preds.columns and preds["datetime_utc"].notna().any():
        last_dt = preds["datetime_utc"].dropna().iloc[-1]
        last_dt = pd.to_datetime(last_dt, utc=True)
        age = datetime.now(tz=timezone.utc) - last_dt.to_pydatetime()
    else:
        mtime = datetime.fromtimestamp(PREDICTIONS_CSV.stat().st_mtime, tz=timezone.utc)
        age = datetime.now(tz=timezone.utc) - mtime
    ok = age <= timedelta(minutes=30)
    detail = f"dernière prédiction il y a {humanize_delta(age)}"
    return ok, detail


def check_trader_status(state: Dict[str, Any]) -> tuple[bool, str]:
    if not STATE_FILE.exists():
        return False, "bot_state.json manquant"
    curve = state.get("equity_curve") or []
    if not curve:
        ok, detail = file_recent_status(STATE_FILE, 10)
        return ok, detail + " (courbe vide)"
    last_ts = curve[-1][0]
    dt = pd.to_datetime(last_ts, utc=True, errors="coerce")
    if pd.isna(dt):
        return False, "horodatage equity invalide"
    age = datetime.now(tz=timezone.utc) - dt.to_pydatetime()
    ok = age <= timedelta(minutes=30)
    detail = f"maj trader il y a {humanize_delta(age)}"
    return ok, detail


def render_status_badge(column, label: str, ok: bool, detail: str) -> None:
    color = "#16c784" if ok else "#ea3943"
    status = "OK" if ok else "OFF"
    column.markdown(
        f"""
        <div style="background:#111;border:1px solid #333;border-radius:10px;padding:10px 12px;margin-bottom:6px;">
          <div style="color:#bbb;font-size:12px;margin-bottom:4px;">{label}</div>
          <div style="color:{color};font-size:22px;font-weight:700;">{status}</div>
          <div style="color:#777;font-size:11px;margin-top:3px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def now_paris_str() -> str:
    return datetime.now(TZ_PARIS).strftime("%d/%m/%Y %H:%M:%S")


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for key, value in DEFAULT_STATE.items():
                state.setdefault(key, value)
            return state
        except Exception as exc:
            st.warning(f"Unable to load bot_state.json: {exc}")
    return DEFAULT_STATE.copy()


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
    except Exception as exc:
        st.error(f"Unable to read {PREDICTIONS_CSV}: {exc}")
        return pd.DataFrame()

    if "datetime_paris" in df.columns:
        df["datetime_paris"] = pd.to_datetime(df["datetime_paris"], utc=True, errors="coerce").dt.tz_convert(TZ_PARIS)
    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    return df


def fetch_btc_ticker() -> Dict[str, Optional[float]]:
    try:
        pr = requests.get(BINANCE_PRICE_URL, timeout=4).json()
        s24 = requests.get(BINANCE_24H_URL, timeout=4).json()
        price = float(pr.get("price"))
        change_pct = float(s24.get("priceChangePercent"))
        high = float(s24.get("highPrice"))
        low = float(s24.get("lowPrice"))
        volume = float(s24.get("volume"))
        return {"price": price, "change_pct": change_pct, "high": high, "low": low, "volume": volume}
    except Exception as exc:
        return {"error": str(exc)}


def fetch_klines(interval: str = "1m", limit: int = 300) -> pd.DataFrame:
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=6)
    resp.raise_for_status()
    data = resp.json()
    rows = [
        {
            "open_time": datetime.fromtimestamp(item[0] / 1000, tz=TZ_PARIS),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        }
        for item in data
    ]
    return pd.DataFrame(rows)


def kpi_block(label: str, value: str, help_text: Optional[str] = None, color: str = "white") -> None:
    st.markdown(
        f"""
        <div style="background:#111;border:1px solid #333;border-radius:10px;padding:10px 12px;margin-bottom:6px;">
          <div style="color:#bbb;font-size:12px;margin-bottom:4px;">{label}</div>
          <div style="color:{color};font-size:22px;font-weight:700;">{value}</div>
          {f'<div style="color:#777;font-size:11px;margin-top:3px;">{help_text}</div>' if help_text else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    view = df.tail(25).iloc[::-1].copy()
    if "datetime_paris" in view.columns:
        view["heure_paris"] = view["datetime_paris"].dt.strftime("%d/%m %H:%M:%S")
    else:
        view["heure_paris"] = ""

    for required in ["confidence", "ret_pred", "mag_pred", "article_found", "article_chars", "article_status", "text_source"]:
        if required not in view.columns:
            if required in {"article_status", "text_source"}:
                view[required] = ""
            else:
                view[required] = np.nan

    view["confidence"] = pd.to_numeric(view["confidence"], errors="coerce")
    view["confidence"] = view["confidence"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    view["ret_pred_pct"] = pd.to_numeric(view["ret_pred"], errors="coerce").mul(100.0)
    view["ret_pred_pct"] = view["ret_pred_pct"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
    view["mag_pred_pct"] = pd.to_numeric(view["mag_pred"], errors="coerce").abs().mul(100.0)
    view["mag_pred_pct"] = view["mag_pred_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    view["article_found"] = view["article_found"].map(lambda x: "✅" if bool(x) else "❌")
    view["article_chars"] = pd.to_numeric(view["article_chars"], errors="coerce").fillna(0).astype(int)

    cols = [
        "heure_paris",
        "prediction",
        "confidence",
        "ret_pred_pct",
        "prob_bull",
        "prob_neut",
        "prob_bear",
        "mag_pred_pct",
        "mag_bucket",
        "features_status",
        "article_status",
        "article_found",
        "article_chars",
        "text_source",
        "title",
    ]
    for c in cols:
        if c not in view.columns:
            view[c] = ""
    return view[cols]


def style_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    return df


st.set_page_config(page_title="BTC Live Bot", layout="wide")
st_autorefresh(interval=REFRESH_SEC * 1000, key="refresh")
st.title("BTC Live Bot - Paper Trading (Streamlit)")

state = load_state()
preds_df = load_predictions()

status_cols = st.columns(3)
rss_ok, rss_detail = check_rss_status()
model_ok, model_detail = check_model_status(preds_df)
trader_ok, trader_detail = check_trader_status(state)
render_status_badge(status_cols[0], "Flux RSS", rss_ok, rss_detail)
model_label = f"Modèle {MODEL_DIR.name}"
render_status_badge(status_cols[1], model_label, model_ok, model_detail)
render_status_badge(status_cols[2], "Trader fictif", trader_ok, trader_detail)

with st.sidebar:
    st.header("Model info")
    info: Dict[str, Any] = {}
    cfg_path = MODEL_DIR / "config.json"
    if cfg_path.exists():
        try:
            info = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            st.warning(f"Unable to read config.json: {exc}")
    trainer_path = MODEL_DIR / "trainer_state.json"
    if trainer_path.exists():
        try:
            trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
            info["best_metric"] = trainer.get("best_metric")
            info["best_model_checkpoint"] = trainer.get("best_model_checkpoint")
        except Exception as exc:
            st.warning(f"Unable to read trainer_state.json: {exc}")
    if info:
        st.json(info)
    else:
        st.info("No model metadata found.")

    if st.button("Reset session state"):
        if STATE_FILE.exists():
            STATE_FILE.unlink(missing_ok=True)
        state = load_state()
        st.success("bot_state.json has been reset.")

st.caption(f"Last refresh (Paris): **{now_paris_str()}** — auto refresh every {REFRESH_SEC}s")

col_left, col_right = st.columns([1.6, 1.4], gap="large")

with col_left:
    st.subheader("BTC/USDT chart")
    interval = st.selectbox("Interval", INTERVALS, index=0)
    try:
        df_klines = fetch_klines(interval=interval, limit=400 if interval in ("1m", "5m") else 300)
    except Exception as exc:
        st.error(f"Failed to fetch Binance klines: {exc}")
        df_klines = pd.DataFrame()

    ticker = fetch_btc_ticker()
    price = ticker.get("price") if isinstance(ticker, dict) else None

    if not df_klines.empty:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df_klines["open_time"],
                    open=df_klines["open"],
                    high=df_klines["high"],
                    low=df_klines["low"],
                    close=df_klines["close"],
                    name="BTCUSDT",
                )
            ]
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            uirevision="chart_lock",
        )
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True})
    else:
        st.info("Waiting for market data...")

with col_right:
    st.subheader("Market snapshot")
    if isinstance(ticker, dict) and ticker.get("price") is not None:
        pct = ticker.get("change_pct") or 0.0
        color = "#16c784" if pct >= 0 else "#ea3943"
        kpi_block("Price", f"{ticker['price']:,.2f} $", color=color, help_text=f"High {ticker['high']:.0f} / Low {ticker['low']:.0f}")
        kpi_block("Change 24h", f"{pct:+.2f} %")
        kpi_block("Volume 24h", f"{ticker['volume']:.2f}")
    else:
        st.error(f"Unable to fetch ticker: {ticker.get('error') if isinstance(ticker, dict) else 'unknown error'}")

    st.subheader("Dernier signal")
    last_signal = state.get("last_signal") if isinstance(state, dict) else None
    if last_signal:
        pred = str(last_signal.get("prediction", "")).upper()
        conf_raw = last_signal.get("confidence")
        try:
            conf = float(conf_raw) if conf_raw is not None else None
        except (TypeError, ValueError):
            conf = None
        ret_raw = last_signal.get("ret_pred")
        try:
            ret_pred = float(ret_raw) if ret_raw is not None else None
        except (TypeError, ValueError):
            ret_pred = None
        mag_pred = abs(ret_pred) if ret_pred is not None else None
        lev_raw = last_signal.get("planned_leverage")
        leverage = int(lev_raw) if isinstance(lev_raw, (int, float)) else None
        risk_raw = last_signal.get("risk_fraction")
        risk_fraction = float(risk_raw) if isinstance(risk_raw, (int, float)) else None
        article_status = last_signal.get("article_status", "")
        article_found = last_signal.get("article_found")
        text_source = last_signal.get("text_source", "")
        features_status = last_signal.get("features_status", "")
        icon = "🟢" if pred == "BULLISH" else ("🔴" if pred == "BEARISH" else "⚪")
        lines = [
            f"{icon} **{pred or '—'}** — confiance {conf:.2f}" if conf is not None else f"{icon} **{pred or '—'}**",
            f"Retour 60m: {ret_pred * 100:+.2f}% (|{mag_pred * 100:.2f}%|)" if ret_pred is not None else "Retour 60m: n/a",
            f"Levier planifié: x{int(leverage)}" if leverage else "Levier planifié: n/a",
            f"Risque engagé: {risk_fraction * 100:.2f}%" if isinstance(risk_fraction, (float, int)) else "Risque engagé: n/a",
            f"Article: {'✅' if article_found else '⚠️'} {article_status or 'inconnu'} ({text_source or 'n/a'})",
            f"Features: {features_status or 'n/a'}",
        ]
        st.markdown("\n".join(f"- {line}" for line in lines if line))
        if last_signal.get("title"):
            st.caption(last_signal.get("title"))
        if last_signal.get("url"):
            st.markdown(f"[Voir l'article]({last_signal['url']})")
    else:
        st.info("En attente d'un signal du modèle.")

    st.subheader("Live predictions")
    if not preds_df.empty:
        table_df = format_prediction_table(preds_df)
        st.dataframe(table_df, width="stretch", height=420)
    else:
        st.info("Waiting for rows in live_predictions.csv...")

bottom_left, bottom_right = st.columns([1.1, 1.3], gap="large")

with bottom_left:
    st.subheader("Bot state")
    equity = state.get("equity", 0.0)
    start_equity = state.get("starting_equity", 0.0)
    pnl_total = equity - start_equity
    pnl_pct = (pnl_total / start_equity * 100.0) if start_equity else 0.0

    position = state.get("position")
    if position:
        info = f"{position['side'].upper()} @ {position['entry']:.2f} $"
        if position.get("leverage"):
            info += f" — lev x{position['leverage']}"
        kpi_block("Current position", info, help_text=f"TP {position.get('tp',0):.2f} / SL {position.get('sl',0):.2f}")
    else:
        kpi_block("Current position", "None")

    kpi_block("Equity", f"{equity:,.2f} $", help_text=f"Start {start_equity:,.0f} $")
    kpi_block("Total PnL", f"{pnl_total:+,.2f} $", color="#16c784" if pnl_total >= 0 else "#ea3943", help_text=f"{pnl_pct:+.2f} %")

    st.markdown("**Recent trades**")
    trades = state.get("trades", [])
    if trades:
        trades_df = pd.DataFrame(trades).tail(20).iloc[::-1]
        st.dataframe(trades_df, width="stretch", height=320)
    else:
        st.info("No trades recorded yet.")

with bottom_right:
    st.subheader("Equity curve")
    curve = state.get("equity_curve", [])
    if len(curve) >= 2:
        ts = []
        values = []
        for t, v in curve:
            dt = pd.to_datetime(t, utc=True, errors="coerce")
            if pd.isna(dt):
                continue
            ts.append(dt.tz_convert(TZ_PARIS))
            values.append(float(v))
        if ts:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=values, mode="lines", name="Equity"))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", uirevision="equity_lock")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Equity curve not available yet.")
    else:
        st.info("Equity curve will appear after a few refreshes.")

st.caption("Paper trading dashboard — not financial advice.")






