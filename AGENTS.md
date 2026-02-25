# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python-based cryptocurrency trend reversal detection and prediction system ("Market Top Detector"). It is a single-product repo with 5 layers of analysis scripts in `scripts/`. See `README.md` for full architecture details.

### Hardcoded paths

All scripts use hardcoded paths to `/Users/king/quant/market-top-detector/`. On the cloud VM, a symlink is created during setup:
```
/Users/king/quant/market-top-detector -> /workspace
```
This symlink must exist for any script to run. The update script handles this automatically.

### Dependencies

- Python 3.12, NumPy, Pandas (<3.0), PyTorch (CPU), Matplotlib
- **Pandas must be < 3.0** — scripts use `iloc` to assign float values into int64 columns, which pandas 3.x rejects with `LossySetitemError`.
- PyTorch runs on CPU in this environment (no MPS/CUDA). Scripts auto-detect and fall back to CPU.

### Running scripts

All scripts are standalone and run from the repo root:
```bash
python3 scripts/top_risk_model.py          # Layer 2: multi-factor risk scoring (~5s)
python3 scripts/transformer_risk_model.py  # Layer 3: single-asset Transformer (~4min on CPU)
python3 scripts/multi_asset_transformer.py # Layer 4: multi-asset Transformer (~long, CPU intensive)
python3 scripts/cfx_predict_vs_ema.py      # Layer 5: CFX backtest vs EMA (~long, CPU intensive)
python3 scripts/cfx_strict_eval.py         # Layer 5b: strict no-leakage eval (~long)
```

- `scripts/plot_two_layer_peaks.py` (Layer 1) requires an external `btcdata` package not in this repo — it cannot run in isolation.
- Font warnings ("Arial Unicode MS not found") are harmless; charts render with fallback fonts.
- Output images go to `img/`. Model weights go to `models/`.

### No lint/test framework

This project has no linter configuration, no test framework, and no build system. It is a collection of standalone Python data science scripts.
