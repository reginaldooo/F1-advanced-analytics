# F1 Advanced Analytics: make it a real ML project (free tools only)

## Context
Project is one notebook (`f1-advanced-analytics.ipynb`, 13 cells) plus README. Current "ML" is K-Means with hardcoded k=4 on 2 features (cell 6), a keyword-matching DNF classifier labelled "NLP" (cell 8), and raw lap-time std. No validation, no supervised model, no train/test split, no tests, no pinned env. A friend also found a real methodology flaw: `Points Per Race` uses raw points across scoring eras (cell 4/6), which skews the x-axis and the clusters.

Goal: turn it into a credible, reproducible ML project. Free tools only. Discussion/plan only for now; user executes later.

## Recommended framework: CRISP-DM style pipeline
Business understanding -> data understanding -> preparation -> modeling -> evaluation -> deployment. Maps cleanly onto the folders below.

## Phase 0: Fix the foundation (do first, results change)
- Recompute points from `positionOrder` under one scale (2010+ 25-18-15-12-10-8-6-4-2-1, DNF = 0). Decide: fastest-lap point, shared drives.
- Replace raw lap-time std with coefficient of variation (std / mean lap time per driver-race), so circuits and eras are comparable.
- Add an era-aware lap filter instead of fixed `< 150000` ms.
- Rerun clustering and charts; update README insights, which will likely change.

## Phase 1: Project structure
```
f1-advanced-analytics/
  data/{raw,processed}     # gitignored, DVC-tracked
  notebooks/               # EDA + storytelling only
  src/f1/{data,features,models,viz}.py
  tests/
  models/ reports/figures/
  pyproject.toml (or requirements.txt + lockfile)
  Makefile, .pre-commit-config.yaml, .github/workflows/ci.yml
```
Move logic out of cells into `src/` functions, notebooks import them.

## Phase 2: Modeling upgrades (the "looks like ML" core)
1. **Clustering done properly:** choose k with silhouette + elbow + Davies-Bouldin, compare K-Means vs GMM vs HDBSCAN, check stability across seeds, add PCA/UMAP 2D projection plot.
2. **Supervised task (needs a real target):** predict podium / points finish / DNF from grid position, driver form, constructor form, circuit, era.
   - Models: logistic regression baseline, then LightGBM / XGBoost.
   - **Time-based split** (train on past seasons, test on recent) to avoid leakage; `TimeSeriesSplit` CV.
   - Metrics: ROC-AUC, PR-AUC, log loss, calibration curve; compare against naive baseline (grid position only).
3. **Real NLP for DNF status:** replace keyword rules with TF-IDF + logistic regression, then optionally a small sentence-transformer embedding, evaluated against hand-labelled sample.
4. **Cross-era driver rating:** Elo or Bradley-Terry / mixed-effects model with driver and constructor effects, which separates driver skill from car, directly answers the README question (Fangio vs Verstappen).
5. **Tuning + explainability:** Optuna for hyperparameters, SHAP for feature importance.

## Phase 3: MLOps layer (all free)
- **MLflow** (local) for experiment tracking and model registry.
- **DVC** (local or free DagsHub remote) for data versioning + `dvc.yaml` pipeline stages.
- **pytest** for feature/leakage tests, **ruff** + **pre-commit**, **GitHub Actions** CI.
- **Evidently** for a drift/data-quality report (e.g. era drift).
- Optional: **Docker** image for reproducibility.

## Phase 4: Delivery
- **Streamlit** app (free on Streamlit Community Cloud): driver explorer, cluster view, podium predictor.
- **Model card** and **data sheet** in `docs/`, README rewritten with methodology, metrics table, limitations, reproduce steps.
- Optional extra data: **FastF1** (free, telemetry/weather from 2018+) and **Jolpica-F1** API (Ergast successor) to extend past the Kaggle snapshot.

## Suggested order and effort
Phase 0 (small) -> Phase 1 (small) -> Phase 2.1 and 2.2 (medium, biggest credibility gain) -> Phase 3 tracking + CI (small) -> Phase 2.4 / 2.3 (medium) -> Phase 4 (medium).

## Decisions (2026-10-19)
- Goal: **comparison only, no prediction** for now. Drop the supervised podium/DNF model (2.2), SHAP, and time-based split. Keep clustering (2.1), DNF text classifier (2.3), Elo/Bradley-Terry rating (2.4), Optuna only if a tuned model remains.
- Add **criteria analysis**: build a multi-criteria driver profile (normalized points/race, win and podium rate, CV consistency, positions gained, driver-error DNF rate, mechanical DNF rate), check redundancy with PCA/correlation, and add teammate-relative versions to separate driver from car.
- Data: **Kaggle CSVs only** for now.
- Delivery: Streamlit Community Cloud is fine (runs from GitHub, local machine need not stay on). App reads a small precomputed driver-level table, not the raw CSVs. Check the Kaggle license before committing derived data.
- Baseline of the current state is in `BASELINE.md`, `baseline/`, and `images/baseline-v1/`.

## Verification (when executed)
- `make test` and CI green; `dvc repro` rebuilds features and models from raw CSVs.
- MLflow UI shows tracked runs with metrics; supervised model beats the grid-position baseline on the held-out recent seasons.
- Notebook runs top to bottom from a clean env using the lockfile.
- README numbers match regenerated figures in `reports/figures/`.
