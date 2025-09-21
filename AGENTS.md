# Repository Guidelines

## Project Structure & Module Organization
- `live/feature_builder.py` holds the live Binance feature cache used during inference.
- `bridge_inference.py`, `live_trader.py`, and `app.py` drive the live stack; `legacy/` keeps superseded scripts.
- Dataset prep and training scripts (e.g., `train_model_v7_1_multi.py`, `prepare_dataset_full.py`) remain at the repo root with historical CSV assets under `data/`.
- Model weights live in `models/bert_v7_1_multi/`; runtime artefacts (`live_raw.csv`, `live_predictions.csv`, `bot_state.json`) sit in the project root.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` - install Python dependencies.
- `node rss_to_csv.js` - run the CryptoPanic RSS ingestor (Node =18).
- `python live/price_pipe_writer.py` - keep `price_pipe.csv` populated with Binance BTC quotes (auto-started by the live stack).
- `python bridge_inference.py` - watch `live_raw.csv`, compute features, emit predictions.
- `python live_trader.py` - execute the paper-trading loop (reads `price_pipe.csv`, falls back to Binance REST).
- `streamlit run app.py` - launch the dashboard at http://localhost:8501.
- `python -m py_compile bridge_inference.py live_trader.py live/feature_builder.py app.py` - quick syntax gate for core modules.


## Coding Style & Naming Conventions
- Target Python 3.9+, PEP 8 spacing (4-space indent, snake_case functions, PascalCase classes).
- Keep files ASCII unless data requires otherwise; avoid emojis or smart punctuation in code.
- Place new runtime helpers under `live/`; keep offline or training utilities near existing training scripts.

## Testing Guidelines
- Add unit tests under `tests/` (create if missing) named `test_<module>.py`; prefer pytest for consistency.
- Replay a captured `live_raw.csv` slice through `bridge_inference.py` and check appended rows in `live_predictions.csv` before shipping feature changes.
- Document mock price feeds or fixtures inside `tests/fixtures/` to ensure deterministic runs.

## Commit & Pull Request Guidelines
- Use short, imperative commit messages (e.g., `Add magnitude bucket helper`). Reference issues in the body (`Refs #123`) when applicable.
- Pull requests should summarise the change, list manual test commands, and include dashboard screenshots whenever UI output changes.
- Keep diffs focused; split infrastructure and model-training adjustments into separate PRs for easier review.

## Security & Configuration Tips
- Never commit secrets; store API keys in `.env` files ignored by Git.
- Validate any external writer that updates `price_pipe.csv` to prevent malformed prices from entering the trader.
- Large datasets should stay in `data/` and remain untracked—update `.gitignore` if new raw exports are added.
