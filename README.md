# Sentiment Trading AI Bot

## Overview

This repository contains the tooling used to collect crypto news, transform them into multimodal features, train a DeBERTa based classifier/regressor, and stream live predictions to a paper-trading loop.

### Live components

1. **News ingestion**: `node rss_to_csv.js` writes deduplicated items (Paris time) into `live_raw.csv`.
2. **Price pipe writer**: `python live/price_pipe_writer.py` keeps `price_pipe.csv` fresh with Binance BTCUSDT quotes (auto-launched by `run_live_stack.py`).
3. **Inference bridge**: `python bridge_inference.py` watches `live_raw.csv`, fetches full article bodies, rebuilds the same market/context features used during training, runs the multimodal model, and appends rows (with article/feature diagnostics) to `live_predictions.csv`.
4. **Paper trader**: `python live_trader.py` transforms predictions into positions, updates `bot_state.json`, and tracks equity.
5. **Dashboard**: `streamlit run app.py` shows the BTC chart, predictions, trades, and the equity curve.

All timestamps exposed to users are aligned on Europe/Paris.

### Training pipeline

Historic data preparation and training scripts remain under the root directory. The main entry point is `train_model_v7_1_multi.py`, which expects the dataset produced by `prepare_dataset_full.py` and the helpers in `attach_market_features.py`.

> **Export tip** — The trainer now writes to `models/bert_v7_1_plus/` by default so the live stack picks up fresh checkpoints automatically. Override the target with `OUTPUT_DIR=/path/to/export python train_model_v7_1_multi.py` if you need to keep multiple runs side by side.

## Quick start (live stack)

```
# 1. install dependencies
pip install -r requirements.txt

# 2. run the news collector (node >= 18)
node rss_to_csv.js

# 3. run the live stack (ingestion + inference + trader)\npython run_live_stack.py\n\n# 4. launch the dashboard (optional if --with-dashboard used)\nstreamlit run app.py
```

Optional: run `python live/price_pipe_writer.py` standalone if you need to feed `price_pipe.csv` manually (two columns: `timestamp,price`). `run_live_stack.py` launches it for you, and `live_trader.py` still falls back to Binance public prices if the file is absent.

> **Model switcher** — Set `MODEL_DIR=/path/to/models/bert_v7_1_plus` (or your preferred export) before starting the stack to point both the bridge, trader, and dashboard to the right checkpoint. If unset, the stack now auto-detects `models/bert_v7_1_plus/` and falls back to `models/bert_v7_1_multi/` when the newer export is absent.

## Repository structure

- `live/feature_builder.py`: reusable helper that computes live market features for the model.
- `bridge_inference.py`: unified inference loop (direction + magnitude).
- `live_trader.py`: paper trading engine with dynamic sizing.
- `app.py`: Streamlit dashboard.
- `live/state_utils.py`: shared helpers to load/save the bot state and normalise last-signal payloads.
- `legacy/`: previous live scripts kept for reference.
- `models/`: trained artifacts.
- `data/`, `strategies/`, `notebooks/`: historical data and experiments.

## State files

- `live_raw.csv`: raw news feed (Paris + UTC time, deduplicated).
- `live_predictions.csv`: enriched predictions (probas, magnitude, feature status).
- `bot_state.json`: persistent trader state (trades, equity, current position).

Each component is idempotent and can be restarted independently.

