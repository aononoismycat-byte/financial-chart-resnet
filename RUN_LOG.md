# Experiment & Reproducibility Log

Chronological record of protocol registration, training runs, frozen test evaluation, and repository milestones. Future experimental phases (e.g., cost-aware backtesting, tabular baselines, and calibration analysis) should be appended in chronological order with exact commands, commit/manifest hashes, and confirmatory vs. exploratory status.

## Reference Environment

- **Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`)
- **Runtime**: Python `3.11.4`, PyTorch `2.8.0+cu126`, torchvision `0.23.0+cu126`
- **Global seed**: `42`
- **Artifact policy**: Raw Yahoo Finance CSVs (`data/`), local experiment logs, and `.pt` checkpoints are excluded from Git tracking via `.gitignore`. Curated metrics, figures, and SHA-256 verification hashes are tracked in `results/` and `experiments/research_protocol.json`.

---

## Phase 1 — Confirmatory Image-Classification Study (2026-10-28 to 2026-10-29)

- **2026-10-28 — SPY Classifier-Only Run**
  - Command: `python scripts/train_spy_resnet.py --output experiments/spy_resnet18_d2_seed42`
  - Setup: ImageNet-pretrained ResNet-18, frozen backbone + BatchNorm statistics, Adam (`lr=1e-3`), batch size `32`, `10` epochs.
  - Outcome: Minimum validation cross-entropy `0.698535` at epoch `4` (`51.30%` validation accuracy). Checkpoint reload verified exact prediction and loss parity (`sha256: 17756ae6...`).

- **2026-10-29 — Pre-Test Protocol Registration & SPY Full Fine-Tuning**
  - Recorded pre-test selection and evaluation rules in `experiments/research_protocol.json` prior to any test-set inference.
  - Command: `python scripts/train_spy_full_finetune.py --output experiments/spy_resnet18_full_finetune_seed42`
  - Setup: Adam (`backbone lr=1e-5`, `classifier lr=1e-3`), batch size `32`, `10` epochs; no test evaluation performed.
  - Outcome: Minimum validation cross-entropy `0.733958` at epoch `1` (`48.70%` validation accuracy), followed by sharp training-loss decrease and rising validation loss (overfitting). Per protocol, `classifier_only` was selected for pooled multi-asset training.

- **2026-10-29 — Five-Asset Dataset Construction & Pooled Training**
  - Command: `python scripts/build_multi_asset_dataset.py`
  - Verified all `4,276` NYSE sessions (`2009-01-02` to `2025-12-31`) across `SPY`, `QQQ`, `IWM`, `TLT`, and `GLD`, yielding `15,105` train (`2010–2021`), `2,505` validation (`2022–2023`), and `2,510` test (`2024–2025`) chart samples.
  - Command: `python scripts/train_multi_asset.py`
  - Setup: Pooled `classifier_only` ResNet-18, seed `42`, batch size `32`, Adam (`lr=1e-3`), `10` epochs.
  - Outcome: Minimum overall validation cross-entropy `0.698657` at epoch `9` (`50.06%` validation accuracy). Reloaded checkpoint verified exact validation parity (`sha256: 0e0fb7df...`).

- **2026-10-29 — Frozen Final Test Evaluation & Summary Figures**
  - Command: `python scripts/evaluate_final.py`
  - Locked protocol, manifest, and checkpoint SHA-256 hashes prior to a single held-out test pass comparing ResNet-18 against training-majority and 20-day trend baselines on identical samples.
  - Command: `python scripts/summarize_results.py`
  - Generated aggregate tables and figures in `results/`, and verified unit tests (`python -m unittest discover -s tests -v`).

---

## Phase 2 — Public Release, Inference CLI & Exploratory Numeric Baseline (2026-10-30)

- **2026-10-30 — Initial Public Repository Release**
  - Published [aononoismycat-byte/financial-chart-resnet](https://github.com/aononoismycat-byte/financial-chart-resnet) at commit `342bebe`, confirming exclusion of raw price files, caches, and binary weights.

- **2026-10-30 — Single-Window Prediction CLI**
  - Added `scripts/predict.py` and synthetic example `examples/sample_20_closes.csv`; verified inference output and input/checkpoint validation rules without retraining.

- **2026-10-30 — Post-Hoc Exploratory Numeric Baseline**
  - Command: `python scripts/numeric_baseline_posthoc.py` (protocol recorded in `results/numeric_baseline_protocol.json`, commit `2eef8e7`).
  - Setup: `StandardScaler` + `LogisticRegression(C=1.0)` fit strictly on pooled training data using the same normalized 20-day close vectors, after cryptographic verification of raw CSV and manifest hashes.
  - Outcome: Overall held-out test accuracy `54.70%` vs. `48.17%` for pooled ResNet-18 and `55.26%` for the pooled majority baseline. Explicitly labeled **post-hoc exploratory** because test labels had already been evaluated in Phase 1.
  - Verification: All CPU and GPU unit tests passed (`python -m unittest discover -s tests -v`).
