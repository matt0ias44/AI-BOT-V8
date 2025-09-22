from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def default_state() -> Dict[str, Any]:
    """Return a fresh default bot state."""
    return {
        "starting_equity": 10000.0,
        "equity": 10000.0,
        "position": None,
        "trades": [],
        "equity_curve": [],
        "last_pred_id": None,
        "last_signal": None,
    }


def _clone_default(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_default(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_default(v) for v in value]
    return value


def ensure_state_defaults(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure the provided state dictionary has all required keys."""
    if not isinstance(state, dict):
        state = default_state()
    defaults = default_state()
    for key, value in defaults.items():
        if key not in state or state[key] is None:
            state[key] = _clone_default(value)
    # Equity should not fall below starting equity if missing or invalid
    try:
        equity = float(state.get("equity", defaults["equity"]))
    except (TypeError, ValueError):
        equity = defaults["equity"]
    try:
        starting_equity = float(state.get("starting_equity", defaults["starting_equity"]))
    except (TypeError, ValueError):
        starting_equity = defaults["starting_equity"]
    state["starting_equity"] = starting_equity
    if equity <= 0:
        equity = starting_equity
    state["equity"] = equity
    return state


def load_state_file(path: Path, *, strict: bool = False) -> Dict[str, Any]:
    """Load a bot state JSON file and normalise missing keys.

    When ``strict`` is True, parsing errors are re-raised so the caller can
    surface them to the user. By default the function falls back to the
    default state when a decoding issue occurs.
    """
    if not path.exists():
        return default_state()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if strict:
            raise
        return default_state()
    return ensure_state_defaults(raw)


def save_state_file(path: Path, state: Dict[str, Any]) -> None:
    """Persist the bot state to disk with UTF-8 encoding."""
    safe_state = ensure_state_defaults(state)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(safe_state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def build_last_signal(
    *,
    news_id: str,
    prediction: str,
    confidence: Optional[float],
    ret_pred: Optional[float],
    mag_pred: Optional[float],
    atr_pct: Optional[float],
    article_status: str = "",
    article_found: Optional[bool] = None,
    article_chars: Optional[int] = None,
    text_chars: Optional[int] = None,
    text_source: str = "",
    title: str = "",
    url: str = "",
    features_status: str = "",
    plan: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalised payload for the latest model signal."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).astimezone().isoformat()

    payload: Dict[str, Any] = {
        "time": timestamp,
        "news_id": news_id,
        "prediction": prediction,
        "confidence": None if confidence is None else float(confidence),
        "ret_pred": None if ret_pred is None else float(ret_pred),
        "mag_pred": None if mag_pred is None else float(mag_pred),
        "atr_pct": None if atr_pct is None else float(atr_pct),
        "article_status": article_status,
        "article_found": bool(article_found) if article_found is not None else None,
        "article_chars": None if article_chars is None else int(article_chars),
        "text_chars": None if text_chars is None else int(text_chars),
        "text_source": text_source,
        "title": title[:200] if title else "",
        "url": url,
        "features_status": features_status,
    }

    if plan:
        for key in ("leverage", "planned_leverage"):
            if key in plan and plan[key] is not None:
                payload["planned_leverage"] = int(plan[key])
                break
        if plan.get("risk_fraction") is not None:
            payload["risk_fraction"] = float(plan["risk_fraction"])
        if plan.get("risk_budget") is not None:
            payload["risk_budget"] = float(plan["risk_budget"])
        if plan.get("size_usd") is not None:
            payload["size_usd"] = float(plan["size_usd"])

    if extra_fields:
        for key, value in extra_fields.items():
            payload[key] = value

    return payload


def update_last_signal_file(path: Path, signal: Dict[str, Any]) -> Dict[str, Any]:
    """Update the last signal field in the bot state file and return the state."""
    state = load_state_file(path)
    state["last_signal"] = signal
    save_state_file(path, state)
    return state
