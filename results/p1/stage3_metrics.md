# P1 Stage 3 — Tabular feature baselines vs. ResNet-18

> Post-hoc exploratory analysis on the frozen 2024–2025 test split. Primary features use only the exact 20 unadjusted closes seen by ResNet-18 plus asset identity; `LightGBM (extended, exploratory)` uses OHLCV and 60-session history and is reported separately.

Climatology reference probability (training up-ratio): **0.5404**.
Uncertainty intervals are 95% percentile circular moving-block bootstrap over label dates (block 20 sessions, 5000 replicates).

## Overall (2,510 samples)

| Model | Accuracy [95% CI] | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier | Brier skill | Log loss | ECE | Binomial p |
|---|---|---|---|---:|---:|---:|---:|---:|
| Majority / climatology | 55.26% [53.11%, 57.41%] | 50.00% [50.00%, 50.00%] | 0.500 [0.500, 0.500] | 0.2474 | 0.0000 | 0.6879 | 0.0349 | 1.49e-07 |
| 20-day trend | 50.12% [47.97%, 52.19%] | 48.25% [46.18%, 50.30%] | 0.482 [0.462, 0.503] | — | — | — | — | 0.921 |
| Logistic regression (closes) | 54.70% [52.31%, 57.13%] | 50.26% [49.00%, 51.54%] | 0.515 [0.487, 0.543] | 0.2474 | -0.0000 | 0.6879 | 0.0281 | 2.68e-06 |
| Pooled ResNet-18 | 48.17% [46.02%, 50.36%] | 50.30% [48.63%, 52.01%] | 0.516 [0.495, 0.537] | 0.2536 | -0.0251 | 0.7004 | 0.0839 | 0.0693 |
| Logistic regression (features) | 55.26% [53.15%, 57.41%] | 50.84% [49.79%, 51.91%] | 0.532 [0.509, 0.555] | 0.2465 | 0.0035 | 0.6862 | 0.0286 | 1.49e-07 |
| LightGBM (features) | 55.26% [53.11%, 57.41%] | 50.00% [50.00%, 50.00%] | 0.513 [0.493, 0.533] | 0.2473 | 0.0002 | 0.6878 | 0.0345 | 1.49e-07 |
| LightGBM (extended, exploratory) | 55.22% [53.11%, 57.33%] | 49.97% [49.85%, 50.10%] | 0.511 [0.488, 0.536] | 0.2473 | 0.0003 | 0.6877 | 0.0219 | 1.85e-07 |

## Per-asset metrics

| Asset | Up rate | Model | Accuracy | Balanced accuracy [95% CI] | ROC-AUC [95% CI] | Brier skill |
|---|---:|---|---:|---|---|---:|
| SPY | 57.77% | Majority / climatology | 57.77% | 50.00% [50.00%, 50.00%] | 0.500 [0.500, 0.500] | 0.0000 |
| SPY | 57.77% | 20-day trend | 52.79% | 48.61% [44.94%, 52.12%] | 0.486 [0.449, 0.521] | — |
| SPY | 57.77% | Logistic regression (closes) | 55.78% | 48.78% [46.29%, 50.95%] | 0.515 [0.470, 0.563] | -0.0015 |
| SPY | 57.77% | Pooled ResNet-18 | 49.40% | 52.27% [48.26%, 56.47%] | 0.537 [0.491, 0.586] | -0.0235 |
| SPY | 57.77% | Logistic regression (features) | 57.77% | 50.13% [49.72%, 50.67%] | 0.547 [0.500, 0.595] | 0.0082 |
| SPY | 57.77% | LightGBM (features) | 57.77% | 50.00% [50.00%, 50.00%] | 0.512 [0.462, 0.564] | 0.0003 |
| SPY | 57.77% | LightGBM (extended, exploratory) | 57.77% | 50.00% [50.00%, 50.00%] | 0.506 [0.457, 0.559] | 0.0003 |
| QQQ | 57.97% | Majority / climatology | 57.97% | 50.00% [50.00%, 50.00%] | 0.500 [0.500, 0.500] | 0.0000 |
| QQQ | 57.97% | 20-day trend | 52.59% | 48.62% [44.95%, 52.07%] | 0.486 [0.450, 0.521] | — |
| QQQ | 57.97% | Logistic regression (closes) | 56.77% | 50.34% [47.89%, 52.68%] | 0.519 [0.469, 0.573] | -0.0028 |
| QQQ | 57.97% | Pooled ResNet-18 | 48.21% | 50.77% [47.66%, 53.96%] | 0.514 [0.476, 0.556] | -0.0415 |
| QQQ | 57.97% | Logistic regression (features) | 57.97% | 50.20% [49.36%, 51.05%] | 0.525 [0.484, 0.572] | 0.0083 |
| QQQ | 57.97% | LightGBM (features) | 57.97% | 50.00% [50.00%, 50.00%] | 0.498 [0.451, 0.546] | 0.0012 |
| QQQ | 57.97% | LightGBM (extended, exploratory) | 57.97% | 50.00% [50.00%, 50.00%] | 0.503 [0.456, 0.550] | 0.0008 |
| IWM | 52.19% | Majority / climatology | 52.19% | 50.00% [50.00%, 50.00%] | 0.500 [0.500, 0.500] | -0.0000 |
| IWM | 52.19% | 20-day trend | 48.01% | 47.39% [43.94%, 50.94%] | 0.474 [0.439, 0.509] | — |
| IWM | 52.19% | Logistic regression (closes) | 52.39% | 50.68% [48.11%, 53.26%] | 0.541 [0.497, 0.585] | 0.0050 |
| IWM | 52.19% | Pooled ResNet-18 | 47.81% | 48.86% [45.54%, 52.39%] | 0.493 [0.444, 0.540] | -0.0175 |
| IWM | 52.19% | Logistic regression (features) | 52.19% | 50.35% [48.10%, 52.56%] | 0.547 [0.510, 0.588] | 0.0062 |
| IWM | 52.19% | LightGBM (features) | 52.19% | 50.00% [50.00%, 50.00%] | 0.494 [0.460, 0.530] | -0.0000 |
| IWM | 52.19% | LightGBM (extended, exploratory) | 52.19% | 50.02% [49.44%, 50.63%] | 0.511 [0.479, 0.546] | 0.0010 |
| TLT | 50.20% | Majority / climatology | 50.20% | 50.00% [50.00%, 50.00%] | 0.500 [0.500, 0.500] | -0.0000 |
| TLT | 50.20% | 20-day trend | 46.61% | 46.62% [42.70%, 50.51%] | 0.466 [0.427, 0.505] | — |
| TLT | 50.20% | Logistic regression (closes) | 49.60% | 49.41% [48.20%, 50.63%] | 0.471 [0.429, 0.516] | -0.0063 |
| TLT | 50.20% | Pooled ResNet-18 | 48.80% | 48.88% [45.50%, 52.36%] | 0.512 [0.470, 0.556] | 0.0003 |
| TLT | 50.20% | Logistic regression (features) | 50.60% | 50.42% [48.56%, 52.31%] | 0.482 [0.443, 0.520] | 0.0009 |
| TLT | 50.20% | LightGBM (features) | 50.20% | 50.00% [50.00%, 50.00%] | 0.508 [0.458, 0.560] | 0.0001 |
| TLT | 50.20% | LightGBM (extended, exploratory) | 50.20% | 50.00% [50.00%, 50.00%] | 0.488 [0.431, 0.550] | -0.0004 |
| GLD | 58.17% | Majority / climatology | 58.17% | 50.00% [50.00%, 50.00%] | 0.500 [0.500, 0.500] | 0.0000 |
| GLD | 58.17% | 20-day trend | 50.60% | 46.64% [43.17%, 49.75%] | 0.466 [0.432, 0.497] | — |
| GLD | 58.17% | Logistic regression (closes) | 58.96% | 52.56% [49.89%, 55.53%] | 0.534 [0.476, 0.591] | 0.0056 |
| GLD | 58.17% | Pooled ResNet-18 | 46.61% | 50.70% [46.60%, 54.76%] | 0.527 [0.476, 0.581] | -0.0441 |
| GLD | 58.17% | Logistic regression (features) | 57.77% | 53.54% [49.88%, 57.16%] | 0.540 [0.491, 0.589] | -0.0063 |
| GLD | 58.17% | LightGBM (features) | 58.17% | 50.00% [50.00%, 50.00%] | 0.533 [0.487, 0.578] | -0.0005 |
| GLD | 58.17% | LightGBM (extended, exploratory) | 57.97% | 49.83% [49.48%, 50.00%] | 0.527 [0.481, 0.578] | -0.0003 |

## Paired differences on overall (A − B)

Positive differences favour A for accuracy, balanced accuracy and AUC; negative differences favour A for Brier and log loss.

| A − B | Metric | Difference | 95% CI | Bootstrap p |
|---|---|---:|---|---:|
| LightGBM (features) − Majority / climatology | accuracy | +0.00 pp | [+0.00, +0.00] pp | 1 |
| LightGBM (features) − Majority / climatology | balanced_accuracy | +0.00 pp | [+0.00, +0.00] pp | 1 |
| LightGBM (features) − Majority / climatology | roc_auc | +0.0126 | [-0.0069, +0.0330] | 0.218 |
| LightGBM (features) − Majority / climatology | brier | -0.0001 | [-0.0004, +0.0003] | 0.756 |
| LightGBM (features) − Majority / climatology | log_loss | -0.0001 | [-0.0008, +0.0006] | 0.752 |
| LightGBM (features) − Logistic regression (closes) | accuracy | +0.56 pp | [-0.68, +1.91] pp | 0.399 |
| LightGBM (features) − Logistic regression (closes) | balanced_accuracy | -0.26 pp | [-1.54, +1.00] pp | 0.68 |
| LightGBM (features) − Logistic regression (closes) | roc_auc | -0.0021 | [-0.0289, +0.0264] | 0.875 |
| LightGBM (features) − Logistic regression (closes) | brier | -0.0001 | [-0.0014, +0.0013] | 0.964 |
| LightGBM (features) − Logistic regression (closes) | log_loss | -0.0001 | [-0.0029, +0.0026] | 0.978 |
| LightGBM (features) − Pooled ResNet-18 | accuracy | +7.09 pp | [+3.35, +10.72] pp | 0.0002 |
| LightGBM (features) − Pooled ResNet-18 | balanced_accuracy | -0.30 pp | [-2.01, +1.37] pp | 0.722 |
| LightGBM (features) − Pooled ResNet-18 | roc_auc | -0.0036 | [-0.0328, +0.0265] | 0.81 |
| LightGBM (features) − Pooled ResNet-18 | brier | -0.0063 | [-0.0098, -0.0027] | 0.0002 |
| LightGBM (features) − Pooled ResNet-18 | log_loss | -0.0126 | [-0.0197, -0.0055] | 0.0002 |
| Logistic regression (features) − Logistic regression (closes) | accuracy | +0.56 pp | [-1.00, +2.31] pp | 0.515 |
| Logistic regression (features) − Logistic regression (closes) | balanced_accuracy | +0.58 pp | [-0.96, +2.32] pp | 0.48 |
| Logistic regression (features) − Logistic regression (closes) | roc_auc | +0.0170 | [-0.0046, +0.0400] | 0.134 |
| Logistic regression (features) − Logistic regression (closes) | brier | -0.0009 | [-0.0018, +0.0001] | 0.0868 |
| Logistic regression (features) − Logistic regression (closes) | log_loss | -0.0017 | [-0.0036, +0.0003] | 0.0912 |
| LightGBM (extended, exploratory) − LightGBM (features) | accuracy | -0.04 pp | [-0.16, +0.08] pp | 0.779 |
| LightGBM (extended, exploratory) − LightGBM (features) | balanced_accuracy | -0.03 pp | [-0.15, +0.10] pp | 0.779 |
| LightGBM (extended, exploratory) − LightGBM (features) | roc_auc | -0.0017 | [-0.0126, +0.0106] | 0.802 |
| LightGBM (extended, exploratory) − LightGBM (features) | brier | -0.0000 | [-0.0003, +0.0002] | 0.871 |
| LightGBM (extended, exploratory) − LightGBM (features) | log_loss | -0.0000 | [-0.0005, +0.0004] | 0.872 |

## Validation context (point estimates)

| Model | Accuracy | Balanced accuracy | ROC-AUC | Brier skill | Log loss |
|---|---:|---:|---:|---:|---:|
| Majority / climatology | 49.90% | 50.00% | 0.500 | -0.0000 | 0.6966 |
| 20-day trend | 50.46% | 50.46% | 0.505 | — | — |
| Logistic regression (closes) | 50.42% | 50.50% | 0.500 | -0.0060 | 0.6997 |
| Pooled ResNet-18 | 50.06% | 50.03% | 0.500 | -0.0040 | 0.6987 |
| Logistic regression (features) | 49.18% | 49.26% | 0.504 | -0.0062 | 0.6998 |
| LightGBM (features) | 49.90% | 50.00% | 0.516 | 0.0009 | 0.6962 |
| LightGBM (extended, exploratory) | 49.82% | 49.92% | 0.527 | 0.0022 | 0.6955 |
