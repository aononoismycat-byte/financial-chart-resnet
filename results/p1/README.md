# P1 Evaluation Artifacts Index

This directory contains all committed outputs from the P1 evaluation extension (`experiments/p1_protocol.json`). Per-sample prediction CSVs contain dates, labels, and model scores only — **no price columns** (`close_t`, `Close`, `Adj Close`) are stored in committed CSVs.

## Key Summaries & Governance

- [summary_all.md](summary_all.md) — Unified tables across 2024–2025 test, 2014–2023 walk-forward, Stage 5 multi-seed spread, and 2026 confirmatory holdout.
- [../../docs/MODEL_CARD.md](../../docs/MODEL_CARD.md) — SR 11-7 aligned model risk governance card.
- [../../experiments/p1_protocol.json](../../experiments/p1_protocol.json) — Pre-registered Stage 1–8 protocol and timestamped amendments.

## Stage Index & Regeneration Commands

| Stage | Description | Committed Files | Command |
|---|---|---|---|
| **Stage 1** | Export frozen predictions (64 reproduction checks), probabilistic metrics, calibration & block-bootstrap CIs | `predictions.csv`, `predictions_meta.json`, `stage1_metrics.{json,md}`, `stage1_calibration_test.png` | `python scripts/p1_export_predictions.py` then `python scripts/p1_stage1_metrics.py --overwrite` |
| **Stage 2** | Cost-aware long/flat backtest (`Adj Close`, `0/2/5/10` bp, delay `0/1`) | `stage2_backtest.{json,md}`, `stage2_equity_test.png`, `stage2_cost_sensitivity.png` | `python scripts/p1_backtest.py --overwrite` |
| **Stage 3** | 14 primary price features + 4 exploratory OHLCV features, LogReg & LightGBM + backtest | `stage3_features_meta.json`, `stage3_predictions.csv`, `stage3_predictions_meta.json`, `stage3_metrics.{json,md}`, `stage3_backtest.{json,md}`, `stage3_equity_test.png`, `stage3_cost_sensitivity.png`, `models/{lightgbm_seed42,lightgbm_ext_seed42}.txt`, `models/{logreg_closes,logreg_features}.json` | `python scripts/p1_build_features.py --overwrite` then `python scripts/p1_train_tabular.py --overwrite` |
| **Stage 4** | Frozen-backbone embedding extraction (`batch=32` parity gate) + 2014–2025 expanding walk-forward OOS & backtest | `stage4_embeddings_meta.json`, `stage4_walk_forward_predictions.csv`, `stage4_walk_forward_meta.json`, `stage4_walk_forward.{json,md}`, `stage4_auc_by_year.png`, `stage4_backtest.{json,md}`, `stage4_equity_oos.png`, `stage4_cost_sensitivity.png` | `python scripts/p1_extract_embeddings.py --overwrite` then `python scripts/p1_walk_forward.py --overwrite` |
| **Stage 5** | 6-seed robustness (`42, 0..4`) for ResNet-18 head & LightGBM + Holm-corrected 10-test family | `stage5_seed_predictions.csv`, `stage5_seed_predictions_meta.json`, `stage5_seeds.{json,md}`, `stage5_seed_spread.png` | `python scripts/p1_stage5_seeds.py --overwrite` |
| **Stage 6** | Grad-CAM (`layer4`) & native TreeSHAP explainability | `stage6_explainability.{json,md}`, `stage6_gradcam_examples.png`, `stage6_gradcam_mean.png`, `stage6_shap_bar.png`, `stage6_shap_beeswarm.png`, `stage6_shap_dependence.png` | `python scripts/p1_explain.py --overwrite` |
| **Stage 8** | Pre-registered confirmatory 2026 holdout (`2026-01-02..2026-11-25`, commit `5aa00e4`) | `stage8_freeze_meta.json`, `models/*_wf2026.{txt,json}`, `stage8_holdout_download_audit.json`, `stage8_holdout_predictions.csv`, `stage8_holdout_meta.json`, `stage8_holdout.{json,md}`, `stage8_backtest.{json,md}`, `stage8_equity_2026.png`, `stage8_cost_sensitivity.png` | `python scripts/p1_freeze_holdout_models.py --overwrite`, commit/push, `python scripts/p1_download_holdout.py`, `python scripts/p1_holdout_evaluate.py --pre-reg-commit 5aa00e4785af31c2f5acfd9c9b5685676bd10482 --overwrite` |
| **Stage 7** | Unified summary generation | `summary_all.md`, `README.md` | `python scripts/p1_summary.py --overwrite` |
