"""Utilities for building live market features consistent with the training pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional

import numpy as np
import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _annualize_vol(vol: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        return vol
    scale = np.sqrt(1440.0 / window)
    return vol * scale


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


@dataclass
class FeatureSource:
    """Fetches 1m OHLCV candles from Binance and keeps a rolling cache."""

    symbol: str = "BTCUSDT"
    interval: str = "1m"
    lookback_minutes: int = 360
    session: Optional[requests.Session] = None

    def __post_init__(self) -> None:
        self._session = self.session or requests.Session()
        self._cache: pd.DataFrame = pd.DataFrame()
        self._cache_updated: Optional[datetime] = None

    def _fetch_klines(self, end_time: Optional[datetime]) -> pd.DataFrame:
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": min(self.lookback_minutes + 5, 1000),
        }
        if end_time is not None:
            params["endTime"] = int(_as_utc(end_time).timestamp() * 1000)
        resp = self._session.get(BINANCE_KLINES_URL, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data:
            ts = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df

    def get_price_window(self, event_time: datetime) -> pd.DataFrame:
        event_time_utc = _as_utc(event_time)
        need_start = event_time_utc - timedelta(minutes=self.lookback_minutes)

        refresh_needed = (
            self._cache.empty
            or self._cache.index.max() < event_time_utc
            or self._cache.index.min() > need_start
        )
        if refresh_needed:
            fetched = self._fetch_klines(event_time_utc)
            if fetched.empty:
                raise RuntimeError("Binance klines response empty")
            self._cache = fetched
            self._cache_updated = datetime.now(timezone.utc)

        window = self._cache.loc[(self._cache.index >= need_start) & (self._cache.index <= event_time_utc)]
        if window.empty:
            raise RuntimeError("Insufficient OHLCV history for feature computation")
        return window.copy()


class LiveFeatureBuilder:
    """Recomputes the training-time market & context features for a live event."""

    def __init__(
        self,
        source: FeatureSource,
        feat_stats: Optional[Dict[str, Dict[str, float]]] = None,
        history_size: int = 512,
    ) -> None:
        self.source = source
        self.feat_stats = feat_stats or {}
        self.history: Deque[tuple[datetime, float]] = deque(maxlen=history_size)

    def _prepare_price_frame(self, px: pd.DataFrame) -> pd.DataFrame:
        frame = px.copy()
        if frame.index.tzinfo is None:
            frame.index = frame.index.tz_localize(timezone.utc)
        else:
            frame.index = frame.index.tz_convert(timezone.utc)
        full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="1min", tz=timezone.utc)
        frame = frame.reindex(full_index).ffill()
        frame["ret1m"] = frame["close"].pct_change()
        frame["logret1m"] = np.log(frame["close"] / frame["close"].shift(1))

        for window in [30, 60, 120, 240]:
            rv = frame["logret1m"].rolling(window).std()
            frame[f"realized_vol_{window}m"] = rv
            frame[f"realized_vol_{window}m_annual"] = _annualize_vol(rv, window)

        frame["rsi_14"] = _rsi(frame["close"], period=14)

        tr = _true_range(frame)
        for window in [30, 60, 120]:
            atr = tr.rolling(window).mean()
            frame[f"atr_{window}m"] = atr
            frame[f"atr_{window}m_pct"] = atr / (frame["close"].rolling(1).mean() + 1e-12)

        mid = frame["close"].rolling(20).mean()
        std = frame["close"].rolling(20).std()
        frame["boll_width_20"] = (2 * std) / (mid + 1e-12)

        for window in [5, 15, 30, 60]:
            frame[f"momentum_{window}m"] = frame["close"].pct_change(window)

        frame["vol_z_60m"] = (
            (frame["volume"] - frame["volume"].rolling(60).mean())
            / (frame["volume"].rolling(60).std() + 1e-12)
        )
        frame["volume_rate_30m"] = frame["volume"].rolling(5).sum() / (
            frame["volume"].rolling(30).sum() + 1e-12
        )
        frame["volume_trend_120m"] = frame["volume"].ewm(span=120, adjust=False).mean()

        idx = frame.index
        hour_float = idx.hour + idx.minute / 60.0
        frame["intraday_sin"] = np.sin(2 * np.pi * hour_float / 24)
        frame["intraday_cos"] = np.cos(2 * np.pi * hour_float / 24)
        frame["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        frame["is_weekend"] = (idx.dayofweek >= 5).astype(float)

        return frame

    def _extract_market_features(self, frame: pd.DataFrame) -> Dict[str, float]:
        last = frame.iloc[-1]
        features: Dict[str, float] = {}
        for col in frame.columns:
            if col in {"open", "high", "low", "close", "volume", "ret1m", "logret1m"}:
                continue
            value = last[col]
            features[f"feat_{col}"] = float(value) if pd.notna(value) else np.nan
        return features

    def _compute_context_features(self, event_time: datetime, sentiment: float) -> Dict[str, float]:
        event_time_utc = _as_utc(event_time)
        horizon_60 = event_time_utc - timedelta(minutes=60)
        horizon_180 = event_time_utc - timedelta(minutes=180)

        # purge old events (> 4h)
        cutoff = event_time_utc - timedelta(hours=4)
        filtered: Deque[tuple[datetime, float]] = deque(maxlen=self.history.maxlen)
        for ts, sent in self.history:
            if ts >= cutoff:
                filtered.append((ts, sent))
        self.history = filtered

        last_ten = list(self.history)[-10:]
        sent_values = [s for _, s in last_ten]
        sent_mean = float(np.mean(sent_values)) if sent_values else 0.0
        sent_std = float(np.std(sent_values, ddof=1)) if len(sent_values) > 1 else 0.0

        count_60 = sum(1 for ts, _ in self.history if ts >= horizon_60)
        count_180 = sum(1 for ts, _ in self.history if ts >= horizon_180)

        return {
            "feat_sent_roll_mean_10": sent_mean,
            "feat_sent_roll_std_10": sent_std,
            "feat_news_count_60m": float(count_60),
            "feat_news_count_180m": float(count_180),
        }

    def build(self, event_time: datetime, sentiment: Optional[float] = None) -> Dict[str, float]:
        px = self.source.get_price_window(event_time)
        frame = self._prepare_price_frame(px)
        market = self._extract_market_features(frame)
        context = self._compute_context_features(event_time, float(sentiment or 0.0))

        features = {**market, **context}
        if self.feat_stats:
            for key, stats in self.feat_stats.items():
                if key not in features:
                    features[key] = stats.get("mean", 0.0)
        self.history.append((_as_utc(event_time), float(sentiment or 0.0)))
        return features
