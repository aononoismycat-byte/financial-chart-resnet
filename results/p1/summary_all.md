# P1 Unified Summary Across All Evaluation Stages

> **Scope**: Stages 1–6 cover the `2024–2025` test period (`2,510` samples, **post-hoc exploratory**) and `2014–2023` unseen walk-forward OOS (`12,580` samples). Stage 8 (`2026-01-02` to `2026-11-25`, `1,135` samples) is a **pre-registered confirmatory holdout** (`5aa00e4785af31c2f5acfd9c9b5685676bd10482`).

## Table A — 2024–2025 Held-Out Test (2,510 samples, 5 bp one-way cost, delay 0)

| Model | Accuracy | Balanced Acc | ROC-AUC [95% CI] | Brier Skill | 5 bp Sharpe [95% CI] | Max DD | Turnover/yr |
|---|---:|---:|---|---:|---|---:|---:|
| Majority / Buy & hold | 55.26% | 50.00% | 0.500 [0.500, 0.500] | 0.0000 | 1.493 [0.396, 2.878] | -12.84% | 0.5x |
| 20-day trend | 50.12% | 48.25% | 0.482 [0.462, 0.503] | — | 0.833 [-0.293, 1.987] | -5.91% | 25.0x |
| Pooled ResNet-18 | 48.17% | 50.30% | 0.516 [0.495, 0.537] | -0.0251 | 0.607 [-0.438, 1.838] | -7.68% | 69.2x |
| Logistic regression (closes) | 54.70% | 50.26% | 0.515 [0.487, 0.543] | -0.0000 | 1.381 [0.310, 2.711] | -13.19% | 36.7x |
| Logistic regression (features) | 55.26% | 50.84% | 0.532 [0.509, 0.555] | 0.0035 | 1.594 [0.574, 2.790] | -11.79% | 32.1x |
| LightGBM (features) | 55.26% | 50.00% | 0.513 [0.493, 0.533] | 0.0002 | 1.493 [0.396, 2.878] | -12.84% | 0.5x |
| LightGBM (extended, exploratory) | 55.22% | 49.97% | 0.511 [0.488, 0.536] | 0.0003 | 1.495 [0.405, 2.860] | -12.84% | 1.1x |

## Table B — Walk-Forward, Multi-Seed & 2026 Confirmatory Holdout Robustness

| Model | WF 2014–2023 AUC [95% CI] | Stage 5 Seed AUC (mean ± sd) | 2026 Holdout AUC [95% CI] | 2026 5 bp Sharpe |
|---|---|---|---|---:|
| Majority / Buy & hold | 0.500 (constant per fold; compare vs 0.500) | — | 0.500 [0.500, 0.500] | 0.108 [-1.500, 1.866] |
| 20-day trend | 0.503 [0.493, 0.513] | — | 0.492 [0.464, 0.519] | -0.352 [-2.409, 1.666] |
| ResNet-18 (pooled / emb head) | 0.509 [0.498, 0.520] | 0.5177 ± 0.0098 | 0.485 [0.449, 0.522] | -0.054 [-1.509, 1.695] |
| Logistic regression (closes) | 0.507 [0.494, 0.520] | — | 0.494 [0.454, 0.538] | -0.229 [-1.922, 1.543] |
| Logistic regression (features) | 0.507 [0.495, 0.519] | — | 0.519 [0.484, 0.557] | 0.334 [-1.253, 2.089] |
| LightGBM (features) | 0.500 [0.488, 0.511] | 0.5118 ± 0.0066 | 0.495 [0.471, 0.521] | 0.108 [-1.500, 1.866] |
| LR (closes, WF2026) | — | — | 0.472 [0.426, 0.518] | -0.207 [-1.751, 1.441] |
| LR (features, WF2026) | — | — | 0.523 [0.485, 0.562] | 0.150 [-1.441, 1.917] |
| LightGBM (features, WF2026) | — | — | 0.506 [0.481, 0.534] | -0.048 [-1.567, 1.586] |
| ResNet-18 emb + LR (WF2026) | — | — | 0.493 [0.462, 0.524] | -0.431 [-1.875, 1.205] |

> **Walk-forward climatology note**: Across pooled walk-forward folds, fold-varying `climatology_prob` induces a spurious `0.480` pooled AUC artifact; walk-forward model AUCs must be evaluated against `0.500` rather than against the pooled majority row.

## Table C — Stage 8 Pre-Registered Confirmatory Family (2026 Holdout, Holm α = 0.05)

| ID | Hypothesis | Estimate / Diff | 95% CI (diff vs null) | Raw p | Holm p | Significant? |
|---|---|---:|---|---:|---:|---|
| H1 | `AUC(lightgbm_wf2026) != 0.5` | 0.5055 | [-0.0186, +0.0340] | 0.6656 | 1 | No |
| H2 | `AUC(resnet_pooled) != 0.5` | 0.4855 | [-0.0505, +0.0218] | 0.4708 | 1 | No |
| H3 | `AUC(logreg_closes_wf2026) != 0.5` | 0.4722 | [-0.0745, +0.0178] | 0.2336 | 0.9344 | No |
| H4 | `AUC(resnet_pooled) - AUC(logreg_closes_wf2026) != 0` | 0.0133 | [-0.0333, +0.0706] | 0.6376 | 1 | No |
