#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge inference v2.
- Watches live_raw.csv produced by rss_to_csv.js
- Builds live market features via LiveFeatureBuilder
- Runs the multimodal model (direction + magnitude)
- Appends consolidated rows to live_predictions.csv
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from live.feature_builder import LiveFeatureBuilder, FeatureSource

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path("./models/bert_v7_1_multi")
INPUT_CSV = Path("live_raw.csv")
OUTPUT_CSV = Path("live_predictions.csv")
PROCESSED_FILE = Path("live_processed_ids.json")

LABELS = ["bearish", "neutral", "bullish"]
TARGET_HORIZON = "60"  # use 60m head as primary signal

EXPECTED_RAW_COLUMNS = [
    "datetime_paris",
    "datetime_utc",
    "title",
    "url",
    "summary",
    "source",
    "news_id",
]


def _clean_field(value: str | float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _utc_to_paris(dt_utc: str) -> str:
    if not dt_utc:
        return ""
    ts = pd.to_datetime(dt_utc, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.tz_convert("Europe/Paris").isoformat()


def _ensure_news_id(dt_utc: str, title: str, existing: str) -> str:
    if existing:
        return existing
    base_dt = dt_utc.strip()
    base_title = title.strip()
    if base_dt and base_title:
        return f"{base_dt}|{base_title}"
    if base_title:
        return base_title
    return base_dt


def load_live_raw() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        return pd.DataFrame(columns=EXPECTED_RAW_COLUMNS)

    rows: List[Dict[str, str]] = []
    with INPUT_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for idx, raw in enumerate(reader, start=2):
            if not raw:
                continue
            try:
                if len(raw) >= len(EXPECTED_RAW_COLUMNS):
                    mapped = {
                        col: _clean_field(value)
                        for col, value in zip(EXPECTED_RAW_COLUMNS, raw)
                    }
                elif len(raw) == 5:
                    dt_utc, title, url, summary, source = (_clean_field(v) for v in raw)
                    mapped = {
                        "datetime_utc": dt_utc,
                        "datetime_paris": _utc_to_paris(dt_utc),
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "source": source,
                        "news_id": "",
                    }
                else:
                    print(
                        f"[WARN] skipping malformed row {idx}: "
                        f"expected 5 or 7 columns, got {len(raw)}"
                    )
                    continue

                mapped["news_id"] = _ensure_news_id(
                    mapped.get("datetime_utc", ""),
                    mapped.get("title", ""),
                    _clean_field(mapped.get("news_id")),
                )
                if not mapped.get("datetime_paris"):
                    mapped["datetime_paris"] = _utc_to_paris(mapped.get("datetime_utc", ""))
                rows.append(mapped)
            except Exception as exc:
                print(f"[WARN] unable to parse row {idx}: {exc}")

    if not rows:
        return pd.DataFrame(columns=EXPECTED_RAW_COLUMNS)

    df = pd.DataFrame(rows, columns=EXPECTED_RAW_COLUMNS)
    return df


class MultiModalHead(nn.Module):
    def __init__(self, model_name: str, feat_dim: int, hidden_drop: float = 0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, use_safetensors=True)
        hidden = self.backbone.config.hidden_size
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(hidden_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.GELU(),
            nn.Dropout(hidden_drop),
        )
        self.dir30 = nn.Linear(hidden, 3)
        self.dir60 = nn.Linear(hidden, 3)
        self.dir120 = nn.Linear(hidden, 3)
        self.mag30 = nn.Linear(hidden, 1)
        self.mag60 = nn.Linear(hidden, 1)
        self.mag120 = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        f = self.feat_mlp(feats)
        z = self.fuse(torch.cat([cls, f], dim=-1))
        o_dir30 = self.dir30(z)
        o_dir60 = self.dir60(z)
        o_dir120 = self.dir120(z)
        o_mag30 = self.mag30(z).squeeze(-1)
        o_mag60 = self.mag60(z).squeeze(-1)
        o_mag120 = self.mag120(z).squeeze(-1)
        return (o_dir30, o_dir60, o_dir120), (o_mag30, o_mag60, o_mag120)


@dataclass
class InferenceAssets:
    model: MultiModalHead
    tokenizer: AutoTokenizer
    feat_norm: Dict[str, Dict[str, float]]
    feat_cols: List[str]
    thresholds: Optional[Dict[str, Dict[str, float]]]


def load_model_assets(model_dir: Path) -> InferenceAssets:
    with open(model_dir / "feature_norm.json", "r", encoding="utf-8") as f:
        feat_norm = json.load(f)
    feat_cols = list(feat_norm.keys())

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = MultiModalHead(model_name=str(model_dir), feat_dim=len(feat_cols)).to(DEVICE)
    state_path = model_dir / "multimodal_heads.pt"
    model.load_state_dict(torch.load(state_path, map_location=DEVICE))
    model.eval()

    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    thresh_path = model_dir / "thresholds_mag.json"
    if thresh_path.exists():
        with open(thresh_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    return InferenceAssets(
        model=model,
        tokenizer=tokenizer,
        feat_norm=feat_norm,
        feat_cols=feat_cols,
        thresholds=thresholds,
    )


def normalize_features(feat_norm: Dict[str, Dict[str, float]], feats: Dict[str, float], feat_cols: List[str]):
    vec = []
    for col in feat_cols:
        stats = feat_norm[col]
        val = feats.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = stats.get("mean", 0.0)
        vec.append((val - stats["mean"]) / (stats["std"] or 1e-12))
    return torch.tensor([vec], dtype=torch.float32, device=DEVICE)


def fallback_features(feat_norm: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {col: stats.get("mean", 0.0) for col, stats in feat_norm.items()}


def magnitude_bucket(thresholds: Dict[str, Dict[str, float]] | None, value: float) -> str:
    if thresholds is None:
        return "unknown"
    bucket = thresholds.get(TARGET_HORIZON)
    if not bucket:
        return "unknown"
    q1 = bucket.get("q1", 0.0)
    q2 = bucket.get("q2", 0.0)
    if np.isnan(value):
        return "unknown"
    if value < q1:
        return "small"
    if value < q2:
        return "medium"
    return "large"


def ensure_output_header():
    if OUTPUT_CSV.exists():
        return
    header = [
        "news_id",
        "datetime_paris",
        "datetime_utc",
        "prediction",
        "prob_bear",
        "prob_neut",
        "prob_bull",
        "confidence",
        "mag_pred",
        "mag_bucket",
        "features_status",
        "title",
        "summary",
        "url",
        "source",
        "processed_at",
    ]
    OUTPUT_CSV.write_text(",".join(header) + "\n", encoding="utf-8")


def load_processed_ids() -> List[str]:
    if not PROCESSED_FILE.exists():
        return []
    try:
        ids = json.loads(PROCESSED_FILE.read_text(encoding="utf-8"))
        if isinstance(ids, list):
            return ids
    except Exception as exc:
        print(f"[WARN] could not read {PROCESSED_FILE}: {exc}")
    return []


def persist_processed_ids(ids: List[str]):
    keep = ids[-5000:]
    PROCESSED_FILE.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")


def tail_news(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 200:
        return df
    return df.tail(200)


def run_loop():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    assets = load_model_assets(MODEL_DIR)
    feature_builder = LiveFeatureBuilder(FeatureSource())
    processed_ids = load_processed_ids()
    processed_set = set(processed_ids)
    ensure_output_header()

    print("[INFO] Bridge inference v2 started. Watching", INPUT_CSV)

    while True:
        try:
            if not INPUT_CSV.exists():
                time.sleep(2)
                continue

            df = load_live_raw()
            if df.empty:
                time.sleep(2)
                continue

            df = tail_news(df)
            if "news_id" not in df.columns:
                time.sleep(2)
                continue

            for _, row in df.iterrows():
                news_id = str(row.get("news_id", "").strip())
                if not news_id or news_id in processed_set:
                    continue

                title = str(row.get("title", ""))
                summary = str(row.get("summary", ""))
                url = str(row.get("url", ""))
                source = str(row.get("source", ""))
                body_text = " ".join(part for part in [title, summary] if part)

                dt_paris = pd.to_datetime(row.get("datetime_paris"), utc=True, errors="coerce")
                dt_utc = pd.to_datetime(row.get("datetime_utc"), utc=True, errors="coerce")
                if pd.isna(dt_utc):
                    dt_utc = dt_paris.tz_convert("UTC") if dt_paris is not None else pd.Timestamp.utcnow().tz_localize("UTC")
                event_time = dt_utc.to_pydatetime()

                features_status = "live"
                try:
                    feats = feature_builder.build(event_time)
                except Exception as exc:
                    print(f"[WARN] feature builder failed for {news_id}: {exc}; fallback to means")
                    feats = fallback_features(assets.feat_norm)
                    features_status = "fallback_means"

                Xf = normalize_features(assets.feat_norm, feats, assets.feat_cols)

                enc = assets.tokenizer(
                    body_text,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(DEVICE)
                mask = enc["attention_mask"].to(DEVICE)

                with torch.no_grad():
                    (o30, o60, o120), (g30, g60, g120) = assets.model(ids, mask, Xf)

                logits = o60
                probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
                pred_idx = int(probs.argmax())
                label = LABELS[pred_idx]
                confidence = float(probs[pred_idx])
                mag_val = float(g60.cpu().numpy().reshape(-1)[0])
                bucket = magnitude_bucket(assets.thresholds, mag_val)

                processed_ids.append(news_id)
                processed_set.add(news_id)
                if len(processed_ids) > 6000:
                    processed_ids = processed_ids[-5000:]
                    processed_set = set(processed_ids)
                persist_processed_ids(processed_ids)

                out_row = {
                    "news_id": news_id,
                    "datetime_paris": row.get("datetime_paris"),
                    "datetime_utc": dt_utc.isoformat(),
                    "prediction": label,
                    "prob_bear": round(float(probs[0]), 6),
                    "prob_neut": round(float(probs[1]), 6),
                    "prob_bull": round(float(probs[2]), 6),
                    "confidence": round(confidence, 6),
                    "mag_pred": round(mag_val, 6),
                    "mag_bucket": bucket,
                    "features_status": features_status,
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "source": source,
                    "processed_at": pd.Timestamp.utcnow().isoformat(),
                }

                df_out = pd.DataFrame([out_row])
                df_out.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
                print(
                    f"[PRED] {news_id} | {label} conf={confidence:.2f} "
                    f"bull={probs[2]:.2f} bear={probs[0]:.2f} mag={mag_val:.4f}"
                )
        except Exception as exc:
            print("[ERROR]", exc)
            time.sleep(5)

        time.sleep(2)


if __name__ == "__main__":
    run_loop()
