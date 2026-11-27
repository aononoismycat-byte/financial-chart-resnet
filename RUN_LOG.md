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

---

## Phase 3 — P1 Extension, Stage 1: Probabilistic Evaluation (2026-11-27, post-hoc exploratory)

- **2026-11-27 — P1 Protocol Registration**
  - Recorded `experiments/p1_protocol.json` (Stages 1–8: metrics, bootstrap, backtest, tabular baselines, walk-forward, seeds, explainability, optional 2026 holdout) before computing any P1 result. No 2026 price data had been downloaded by the pipeline.

- **2026-11-27 — Verified Prediction Export**
  - Command: `python scripts/p1_export_predictions.py`
  - Regenerated per-sample validation/test scores for the frozen pooled and SPY ResNet-18 checkpoints, the fixed logistic regression, majority/climatology and 20-day trend. Reproduction gate passed: all 64 frozen confusion-matrix and loss checks matched exactly. Output `results/p1/predictions.csv` (no prices) with hashes in `predictions_meta.json`.

- **2026-11-27 — Stage 1 Metrics**
  - Command: `python scripts/p1_stage1_metrics.py` (CPU only, reads the committed predictions).
  - Pooled test: ResNet-18 accuracy `48.17%` but balanced accuracy `50.30%` [48.63, 52.01] and ROC-AUC `0.516` [0.495, 0.537], statistically indistinguishable from logistic regression (AUC difference `+0.002` [−0.027, +0.029]). Its accuracy deficit versus majority (`−7.09 pp` [−10.72, −3.35]) comes from probabilities shifted below 0.5 (miscalibrated intercept) in an up-trending test period, not from worse ranking. No model achieved positive Brier skill against climatology.

- **2026-11-27 — Stage 2: Cost-Aware Long/Flat Backtest (post-hoc exploratory)**
  - Command: `python scripts/p1_backtest.py --overwrite`
  - Evaluated equal-weight five-asset (`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`) long/flat portfolios on dividend-adjusted `Adj Close` returns after verifying raw CSV hashes and session alignment against `multi_asset_samples.csv`.
  - Test (2024–2025, 5 bp one-way, delay 0): Buy & hold (= Majority) Sharpe `1.493` [0.396, 2.878] (CAGR `19.80%`, Max DD `-12.84%`); Logistic regression (closes) Sharpe `1.381` [0.310, 2.711] (`36.7x` turnover/yr); 20-day trend Sharpe `0.833` [-0.293, 1.987]; Pooled ResNet-18 Sharpe `0.607` [-0.438, 1.838] (`69.2x` turnover/yr, CAGR drops from `7.65%` at 0 bp to `3.99%` at 5 bp, Sharpe diff vs Buy & hold `-0.886` [-1.807, -0.087], `p=0.03`).

- **2026-11-27 — Stage 3: Engineered Tabular Features & LightGBM Baselines (post-hoc exploratory)**
  - Commands: `python scripts/p1_build_features.py --overwrite` and `python scripts/p1_train_tabular.py --overwrite`
  - Gates passed: `20,120` feature rows matched `multi_asset_samples.csv` row-for-row; `(ret_19 > 0) == trend_pred` exact match on all `5,015` val/test rows; `logreg_closes` reproduced Stage 1 `logreg_pred` identically (`max |prob diff| = 4.39e-11`); serialized LightGBM/LogReg artifacts verified.
  - Test (2024–2025): `LightGBM (features)` stopped at `best_iter=21`, ROC-AUC `0.513` [0.493, 0.533], Brier skill `+0.0002`, staying anchored near the training up-ratio (`100%` exposure, 5 bp Sharpe `1.493`). `LightGBM (extended, exploratory)` (`best_iter=28`) gave ROC-AUC `0.511` [0.488, 0.536] (diff vs primary `-0.0017`, `p=0.802`). `Logistic regression (features)` achieved the highest test ROC-AUC `0.532` [0.509, 0.555], Brier skill `+0.0035`, and 5 bp Sharpe `1.594` [0.574, 2.790] (vs Buy & hold `+0.101`, `p=0.593`), though its 2022–2023 validation Sharpe was `-0.425`.

- **2026-11-27 — Stage 4: Frozen-Backbone Embeddings & Expanding Walk-Forward Validation (2014–2025)**
  - Commands: `python scripts/p1_extract_embeddings.py --overwrite` and `python scripts/p1_walk_forward.py --overwrite`
  - Embedding gates passed: all `100` non-fc ResNet-18 backbone tensors matched ImageNet weights exactly; `fc(embeddings)` reproduced Stage 1 `resnet_pooled_pred` identically on all `5,015` val/test samples with `max |prob diff| = 1.79e-7 < 1e-4` (using `batch_size=32` per split to match original cuDNN kernels).
  - Fixed-split sanity check (`ResNet-18 embedding + LR`): test accuracy `52.59%`, ROC-AUC `0.510`, Pearson `r = 0.620` vs `resnet_pooled_prob`.
  - Unseen 2014–2023 OOS (`12,580` samples): all models clustered tightly around chance (`ROC-AUC`: `LR closes` `0.507` [0.494, 0.520], `LR features` `0.507` [0.495, 0.519], `LightGBM` `0.500` [0.488, 0.511], `ResNet-18 emb + LR` `0.509` [0.498, 0.520]). Note: pooled fold-varying climatology yields a spurious `0.480` AUC artifact; walk-forward model AUCs must be evaluated against `0.500` rather than against the pooled `Majority` AUC row (caveat appended to `results/p1/stage4_walk_forward.md`).
  - Full 2014–2025 OOS backtest (5 bp, delay 0): Buy & hold Sharpe `0.950` [0.384, 1.534]; LightGBM `0.939` (`7.8x` turnover/yr); LR closes `0.792`; LR features `0.765` (diff vs Buy & hold `-0.185`, `p=0.0368`); ResNet-18 embedding + LR `0.336` [-0.168, 0.863] (`97.1x` turnover/yr, diff `-0.614`, `p=0.0002`).

- **2026-11-27 — Stage 5: Multi-Seed Robustness & Holm-Corrected Family (post-hoc exploratory)**
  - Amendment recorded in `experiments/p1_protocol.json` before running; command: `python scripts/p1_stage5_seeds.py --overwrite` (seeds `42, 0, 1, 2, 3, 4`).
  - Equivalence gate passed: full-model vs `fc(embedding)` `max |logit diff| = 2.38e-7 < 1e-4`.
  - Across 6 seeds (2024–2025 test): ResNet-18 head accuracy `50.67% ± 2.84%` (`47.89%`–`55.78%`), ROC-AUC `0.5177 ± 0.0098` (6-seed ensemble `0.5197 [0.5007, 0.5391]`); LightGBM accuracy `55.27% ± 0.02%` (balanced accuracy `50.01% ± 0.03%`), ROC-AUC `0.5118 ± 0.0066` (ensemble `0.5121 [0.4906, 0.5346]`).
  - Across the pre-specified 10-test comparison family (`roc_auc` and `log_loss`), all Holm-adjusted p-values were `>= 0.14` (no statistically significant pairwise separation).

- **2026-11-27 — Stage 6: Grad-CAM, Native TreeSHAP & SR 11-7 Model Card (post-hoc exploratory)**
  - Command: `python scripts/p1_explain.py --overwrite`
  - Gates passed: Grad-CAM pass matched Stage 1 `resnet_pooled_pred` identically (`max |prob diff| = 5.00e-13`); native TreeSHAP additivity error `1.67e-16 < 1e-6`.
  - Grad-CAM (`2,510` test charts): mean line lift `1.633x`, rightmost quarter share `0.2762` (vs `0.250` uniform), with background margin artifacts visible on low-volatility charts.
  - TreeSHAP (`lightgbm_seed42.txt`): base log-odds `+0.1613`, top drivers `ret_1` (`0.0222`), `ret_19` (`0.0058`), `max_dd_20` (`0.0057`), showing mild short-term mean reversion outweighed by the positive base-rate intercept (explaining `100%` test exposure / `50.00%` balanced accuracy).
  - Authored `docs/MODEL_CARD.md` under SR 11-7 governance headings.






