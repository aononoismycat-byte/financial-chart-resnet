# Model Risk Management & Governance Card (SR 11-7 Aligned)

> **Scope & Status**: This card documents the empirical evaluation and governance controls for the **Financial Chart ResNet** model family (`ResNet-18` image classifier) and its tabular comparators (`Logistic Regression`, `LightGBM`). All 2024–2025 P1 evaluations are **post-hoc exploratory** extensions following the frozen Phase 1/2 study. This card follows Federal Reserve **SR 11-7** (*Supervisory Guidance on Model Risk Management*) conventions.

---

## 1. Model Details

| Component | Specification | Frozen Artifact / Hash (SHA-256) |
|---|---|---|
| **Primary Image Model** | ImageNet-pretrained `ResNet-18` (`classifier_only`, frozen backbone + frozen BatchNorm, trainable 512→2 linear head) | `experiments/multi_asset_resnet18_seed42/best_model.pt` (`0e0fb7df7e351b388e64861400d99bd38163927996b784ce6d01000ff5dbab10`) |
| **Linear Numeric Comparator** | `StandardScaler` + `LogisticRegression(C=1.0, lbfgs)` on 20 normalized closes | `results/p1/models/logreg_closes.json` (`8e56423f3d9e71305689049bd64571724d0f8b42edce2fc2f870d3598fcb1c87`) |
| **Linear Feature Comparator** | `StandardScaler` + `LogisticRegression(C=1.0, lbfgs)` on 14 price features + 5 asset one-hots | `results/p1/models/logreg_features.json` (`ce9619b7e5d614e05528c1881013680a4a69596732ad28e2a21928d366e36588`) |
| **Gradient-Boosted Tree** | `LightGBM` binary classifier (`num_leaves=15`, `lr=0.02`, early-stopped at iteration `21` on 2022–2023 log loss) | `results/p1/models/lightgbm_seed42.txt` (`e6c083176f850ef3cd59863479aaa61ed10a14a74bad4592bfae7127fa229181`) |
| **Training Horizon** | Pooled 5 ETFs, label dates `2010–2021` (`15,105` samples); validation `2022–2023` (`2,505` samples) | — |
| **Target Definition** | Binary next-session direction: `1` if `Close[t+1] > Close[t]`, else `0` (`Non-up`) | — |

---

## 2. Intended Use & Out-of-Scope Use

- **Intended Use**: Empirical, leakage-controlled benchmarking of 2D line-chart representations vs. direct numerical and engineered tabular features for short-horizon asset return direction.
- **Out-of-Scope / Prohibited Use**:
  - **Not a trading system or investment recommendation.** None of the models demonstrate statistically robust out-of-sample alpha net of realistic transaction costs.
  - **Uncalibrated probabilities.** Raw softmax / tree outputs must not be used directly for position sizing, Kelly allocation, or VaR/risk pricing without explicit probability recalibration and cost optimization.
  - **Out-of-universe assets.** Models are scoped to daily sessions of five liquid US ETFs (`SPY`, `QQQ`, `IWM`, `TLT`, `GLD`); do not apply to single stocks, intraday horizons, or illiquid instruments.

---

## 3. Data & Lineage Controls

- **Source & Calendar**: Daily Yahoo Finance history (`yfinance`, `auto_adjust=False`) over `2009-01-02` to `2025-12-31` (`4,276` verified NYSE sessions per asset, `0` missing/duplicate dates).
- **Split Integrity & Look-Ahead Prevention**: Chronological split strictly by `label_date`: `train` (`2010–2021`, `15,105`), `val` (`2022–2023`, `2,505`), `test` (`2024–2025`, `2,510`). Input windows never contain `Close[t+1]`.
- **Cryptographic Verification**:
  - Frozen sample manifest SHA-256: `e18bf1ed3225ccf88931f59472dabccd78b269a2f62229e1aebcb610d9fcf78b` (`data/processed/multi_asset_samples.csv`).
  - Engineered feature dataset SHA-256: `765ecb8f04e9ea9e0244bd139f12e5efbba5e2fad7b1632a568907a38086a1d1`.
  - Raw CSV hashes per ticker recorded in `results/summary.json` (`asset_audit`).
  - Prediction exports verified via automated reproduction gates (`64` Stage 1 checks, `Stage 3` logreg/trend checks, `Stage 4` ImageNet & `fc(embedding)` checks with `max |prob diff| = 1.79e-7`).

---

## 4. Conceptual Soundness

- **Image Representation Critique**: Converting a 1D vector of 20 unadjusted closing prices into a 224×224 1-pixel monochrome line chart discards exact numeric precision, clips returns beyond `[0.80, 1.20]`, omits OHLC range and volume, and introduces spatial empty-space artifacts when processed by ImageNet-pretrained convolutional kernels.
- **Class Imbalance & Calibration Trap**: Equity and gold ETFs exhibit long-run positive upward drift (`54.04%` training Up rate; `55.26%` in 2024–2025 test). Raw classification accuracy rewards always-Up classifiers during bull markets even when ranking skill (`ROC-AUC`) is near `0.50`.
- **LightGBM Conservatism / Base-Rate Behaviour**: Under strict chronological validation (`2022–2023` bear/sideways regime) with strong regularization (`min_child_samples=100`, `num_leaves=15`, `lr=0.02`), LightGBM early-stopped at **iteration 21**. Because its base log-odds prior (`+0.1613`, `P ≈ 54.0%`) dominates small tree leaf updates (`|SHAP|` mostly `< 0.05`), the model predicts `Up` on essentially **100% of test observations** (`balanced_accuracy = 50.00%`, `exposure = 100.00%`). Thus, LightGBM effectively learned the **training base rate** rather than an actionable directional timing signal; this is documented explicitly so raw accuracy (`55.26%`) is not misinterpreted as active predictive edge.

---

## 5. Performance & Outcomes Analysis (2024–2025 Test, 2,510 Samples)

Intervals are 95% circular moving-block bootstrap percentiles (`block = 20` sessions, `5,000` replicates, keeping all 5 assets per resampled date together).

| Model | Accuracy | Balanced Acc | ROC-AUC [95% CI] | Brier Skill | Log Loss | 5 bp Sharpe [95% CI] | 5 bp Max DD | Turnover/yr |
|---|---:|---:|---|---:|---:|---|---:|---:|
| Majority / Buy & Hold | 55.26% | 50.00% | 0.500 [0.500, 0.500] | 0.0000 | 0.6879 | 1.493 [0.396, 2.878] | -12.84% | 0.5x |
| 20-day Trend | 50.12% | 48.25% | 0.482 [0.462, 0.503] | — | — | 0.833 [-0.293, 1.987] | -5.91% | 25.0x |
| Pooled ResNet-18 | 48.17% | 50.30% | 0.516 [0.495, 0.537] | -0.0251 | 0.7004 | 0.607 [-0.438, 1.838] | -7.68% | 69.2x |
| LR (20 closes) | 54.70% | 50.26% | 0.515 [0.487, 0.543] | -0.0000 | 0.6879 | 1.381 [0.310, 2.711] | -13.19% | 36.7x |
| LR (14 features) | 55.26% | 50.84% | 0.532 [0.509, 0.555] | +0.0035 | 0.6862 | 1.594 [0.574, 2.790] | -11.79% | 32.1x |
| LightGBM (14 features) | 55.26% | 50.00% | 0.513 [0.493, 0.533] | +0.0002 | 0.6878 | 1.493 [0.396, 2.878] | -12.84% | 0.5x |
| LightGBM (OHLCV ext.) | 55.22% | 49.97% | 0.511 [0.488, 0.536] | +0.0003 | 0.6877 | 1.495 [0.405, 2.860] | -12.84% | 1.1x |

Key outcome findings:
1. **Discrimination vs. Calibration**: `Pooled ResNet-18` ranks nearly identically to `LR (closes)` (`AUC 0.516 vs 0.515`, difference `+0.0016 [-0.0267, +0.0289]`, `p = 0.90`). Its `-7.09 pp` accuracy gap vs. majority stems from uncalibrated probabilities shifted below `0.50` (`ECE = 0.0839`), producing `29.80%` market exposure and high daily turnover (`69.2x/yr`) that halves CAGR at 5 bp (`7.65% → 3.99%`, Sharpe diff vs Buy & Hold `-0.886 [-1.807, -0.087]`, `p = 0.03`).
2. **Engineered Features vs. Extended OHLCV**: `LR (features)` achieved `AUC 0.532` on 2024–2025, though its 2022–2023 validation Sharpe was `-0.425` and its 12-year walk-forward Sharpe was `0.765` (worse than Buy & Hold `0.950`). Adding OHLCV + 60-day history (`LightGBM ext`) gave no improvement (`ΔAUC = -0.0017`, `p = 0.802`).

---

## 6. Robustness: Multi-Seed & Expanding Walk-Forward Validation

- **Multi-Seed Sensitivity (`seeds = [42, 0, 1, 2, 3, 4]`, Stage 5)**:
  - `ResNet-18 head` (retrained on verified frozen embeddings; equivalence gate `max |logit diff| = 2.38e-7`): test accuracy varied widely (`50.67% ± 2.84%`, range `47.89%–55.78%`) while ranking remained tight (`AUC 0.5177 ± 0.0098`, 6-seed ensemble `0.5197 [0.5007, 0.5391]`).
  - `LightGBM`: accuracy `55.27% ± 0.02%`, `AUC 0.5118 ± 0.0066` (ensemble `0.5121 [0.4906, 0.5346]`).
  - Across the pre-specified 10-test family (`roc_auc` and `log_loss`), **all Holm-Bonferroni adjusted p-values were `≥ 0.14`**, indicating no statistically significant separation across model families.
- **Expanding Walk-Forward Validation (`2014–2025`, 12 annual folds, Stage 4)**:
  - On the **unseen 2014–2023 OOS period** (`12,580` samples), every model's pooled AUC sat within noise of `0.500`: `LR (closes) 0.507 [0.494, 0.520]`, `LR (features) 0.507 [0.495, 0.519]`, `LightGBM 0.500 [0.488, 0.511]`, `ResNet-18 emb + LR 0.509 [0.498, 0.520]`. *(Note: walk-forward models must be judged against `0.500` rather than the pooled fold-varying climatology row, whose constant shift induces a spurious `0.480` AUC.)*
  - Over the full `2014–2025` OOS backtest at `5 bp`, `ResNet-18 emb + LR` generated `97.1x/yr` turnover and a `0.336 [-0.168, 0.863]` Sharpe vs. `0.950` for Buy & Hold (`diff = -0.614`, `p = 0.0002`).

---

## 7. Explainability (Stage 6)

- **ResNet-18 Grad-CAM (`layer4`, `2,510` test charts)**:
  - Exact prediction verification (`max |prob diff| = 5.00e-13`).
  - Mean line lift is `1.633x` (`1.742x` for Predicted Up vs `1.587x` for Predicted Non-up), and the rightmost quarter (`x ≥ 168`, most recent ~5 sessions) carries `27.62%` of Grad-CAM mass vs. `25.00%` uniform baseline. Visual inspection (`results/p1/stage6_gradcam_examples.png`) shows spurious corner activations in blank background margins on low-volatility charts.
- **LightGBM Native TreeSHAP (`lightgbm_seed42.txt`, `2,510` test samples)**:
  - Additivity verification gate passed (`max |sum(SHAP) + bias - raw_score| = 1.67e-16 < 1e-6`).
  - Top drivers: `ret_1` (mean `|SHAP| = 0.0222` log-odds), `ret_19` (`0.0058`), `max_dd_20` (`0.0057`), `ret_10` (`0.0053`), `asset_QQQ` (`0.0050`), `ma_dev_5` (`0.0042`), `rsi_14` (`0.0040`). Dependence plots reveal short-term mean reversion (positive `ret_1` pushes `P(Up)` downward), though total contribution magnitudes remain small relative to the `+0.1613` base log-odds.

---

## 8. Key Limitations

1. **No multi-seed full fine-tuning**: Stage 5 varied seeds only on frozen-backbone heads and LightGBM due to compute budget; full end-to-end CNN fine-tuning (which severely overfit on SPY) was not repeated across seeds.
2. **Close-only chart & unadjusted signal inputs**: Charts omit intraday range, volume, and dividend adjustments (`auto_adjust=False` for signal inputs, while backtest returns use `Adj Close`).
3. **Single-market / 5-ETF concentration**: Dependent overlapping 20-day windows and correlated equity ETFs (`SPY`, `QQQ`, `IWM`) limit effective sample size.
4. **Post-hoc exploratory status of 2024–2025**: P1 Stages 1–6 evaluate the already-observed 2024–2025 window and therefore cannot serve as confirmatory evidence (addressed via pre-registered Stage 8 2026 holdout).

---

## 9. Ongoing Monitoring Plan (Production / Governance Controls)

If any successor model in this family were deployed in a paper-trading or monitoring environment, quarterly model risk reviews must track:
1. **Calibration Drift**: Rolling 60-session `ECE` and `Brier skill score` vs. trailing 5-year climatology; alert if `ECE > 0.05` or `Brier skill < -0.02`.
2. **Annual Out-of-Sample Discrimination**: Rolling 252-session `ROC-AUC` with block-bootstrap confidence intervals; escalate if trailing 1-year AUC falls below `0.48` or diverges from linear tabular baselines.
3. **Base-Rate & Exposure Regime Shift**: Track trailing `up_ratio` vs. training prior (`0.5404`) and realised `annual_turnover`; trigger recalibration review if effective exposure collapses below `40%` (false-bearish intercept drift) or saturates at `100%` (base-rate collapse) or if turnover exceeds `50x/yr` without positive net Sharpe.

---

## 10. Confirmatory 2026 Holdout (Stage 8)

- **Pre-registration Proof**: Four WF2026 models (`2010–2024` train, `2025` early stopping) and the confirmatory hypothesis family `H1–H4` were frozen, hashed, committed, and pushed at commit [`5aa00e4785af31c2f5acfd9c9b5685676bd10482`](https://github.com/aononoismycat-byte/financial-chart-resnet/commit/5aa00e4785af31c2f5acfd9c9b5685676bd10482) (GitHub Actions CI run `45969724476`, passed) **prior to** downloading any 2026 price history.
- **Holdout Window & Audit**: `2026-01-02` through `2026-11-25` inclusive (`227` verified NYSE sessions per asset, `1,135` total samples, `51.01%` realized Up rate). Every asset matched its `2025-10-01..2025-12-31` `Close` overlap (`64` sessions) within `rtol = 1e-6`.
- **Confirmatory Family (Holm-Bonferroni corrected, α = 0.05, block = 20, B = 5,000)**:

| ID | Hypothesis | Estimate / Diff | 95% CI (diff vs null) | Raw p | Holm p | Significant? |
|---|---|---:|---|---:|---:|---|
| H1 | `AUC(lightgbm_wf2026) != 0.5` | 0.5055 | [-0.0186, +0.0340] | 0.6656 | 1.0000 | No |
| H2 | `AUC(resnet_pooled) != 0.5` | 0.4855 | [-0.0505, +0.0218] | 0.4708 | 1.0000 | No |
| H3 | `AUC(logreg_closes_wf2026) != 0.5` | 0.4722 | [-0.0745, +0.0178] | 0.2336 | 0.9344 | No |
| H4 | `AUC(resnet_pooled) - AUC(logreg_closes_wf2026) != 0` | +0.0133 | [-0.0333, +0.0706] | 0.6376 | 1.0000 | No |

- **Secondary 2026 Backtest (5 bp one-way, delay 0)**: Buy & Hold CAGR `0.48%`, Sharpe `0.108 [-1.500, 1.866]`; `LR (features, WF2026)` CAGR `1.13%`, Sharpe `0.150 [-1.441, 1.917]`; `LightGBM (WF2026)` CAGR `-1.79%`, Sharpe `-0.048 [-1.567, 1.586]`; `Pooled ResNet-18` CAGR `-0.75%`, Sharpe `-0.054 [-1.509, 1.695]` (`83.5x/yr` turnover wiping out `+3.48%` gross CAGR); `ResNet-18 emb + LR (WF2026)` CAGR `-4.95%`, Sharpe `-0.431`. Full tables in `results/p1/stage8_holdout.md` and `results/p1/stage8_backtest.md`.

