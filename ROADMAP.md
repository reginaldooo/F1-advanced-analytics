# F1 Advanced Analytics: consolidated plan (supersedes `synchronous-meandering-newell.md` and the old Phase 0 draft)

## Context
Project: `/Users/reginaldo/Documents/Projects/F1-advanced-analytics/F1-advanced-analytics`. Goal: a credible, reproducible, portfolio-grade **driver-comparison** project (Kaggle CSVs, comparison only, no prediction, free tools, Streamlit Cloud delivery). Decisions of 2026-10-19 in `ROADMAP.md` still hold.

Cleanup of the earlier plans, which collided or repeated each other:
- `ROADMAP.md` (repo) = copy of the first plan + decisions; its Phase 0 (single modern points scale, CV) is **superseded** by the era-neutral features below, and its Verification still mentions a supervised model that was dropped. At execution, replace `ROADMAP.md` with this file's content so there is one source of truth.
- Dropped as redundant/overkill for a comparison-only project on small public data: supervised podium/DNF model, SHAP, time-based split, Optuna, **MLflow, DVC, Evidently**, Docker. Revisit only if a predictive model is added later.
- README rewrite appeared twice (Phase 0 and Phase 4); now done once per phase, final polish at the end.
- Criteria analysis (old decision) overlaps the Phase 0 features: it **extends** them, it does not redo them.

## Phase 0: era-neutral normalization — DONE in working tree, uncommitted
Notebook cells 0, 4, 6, 8, 10, 12 now use finish percentile, podium/top-5/win rate, pace-vs-race-median consistency (laps > 107% dropped), `RobustScaler`, k=4. README and `images/*.png` regenerated; baseline preserved in `baseline/`, `images/baseline-v1/`, tag `v1-baseline`.
Remaining:
- Tick the before/after checklist in `BASELINE.md` (legends: Moss in, Vettel out; Midfielders 14 -> 51, Backmarkers 201 -> 169; Hamilton consistency rank 80/107, Verstappen 63/107; lap data only from 1996).
- Commit: `refactor: era-neutral performance and pace-based consistency`, then `docs: update README and figures`.

## Phase 1: portfolio hygiene + structure (do now, before the LinkedIn post)
Replaces old Phase 1, the "Phase 0.5" idea, and the Phase 3 tooling.
- Layout: `src/f1/{data,features,dnf}.py`, `tests/`, `notebooks/` -> keep a single `f1-advanced-analytics.ipynb` at root for now (EDA/storytelling only, imports from `src/f1`), `reports/figures/`, `docs/`.
- `src/f1/data.py`: `load_tables()` using `kagglehub` (cache dir) with `data/` override; removes the manual download step and the symlink workaround. `features.py`: `add_finish_pct`, `driver_performance_table`, `pace_consistency` (reuse the exact logic now in cells 4/6/8). `dnf.py`: current keyword classifier (replaced in Phase 2).
- Environment: `pyproject.toml` (runtime deps + `dev` extra: pytest, ruff, nbstripout, pre-commit); drop the ad-hoc `pip install` line in README.
- Tests (pytest, written first per TDD rule): `finish_pct` in [0, 1] and = 1 for winner; 107% filter drops only slow laps; pace consistency independent of circuit scale; driver filter (>20 races); cluster naming ordered by win rate. Target 80% coverage on `src/`.
- Tooling: `Makefile` (`make test`, `make lint`, `make figures`), ruff + pre-commit with nbstripout, GitHub Actions CI (lint + tests on a tiny synthetic fixture, no Kaggle download in CI).
- Figures: `make figures` regenerates `images/*.png` from `src/`, so README numbers match.
- Colab badge **skipped**: needs `src/` install + Kaggle credentials, not worth the friction.
- README: add reproduce steps (`pip install -e .[dev]`, `make test`), keep the 1996+ lap-data caveat.

## Phase 2: modeling for comparison
1. **[DONE] Clustering properly:** choose k via silhouette + elbow + Davies-Bouldin; compare K-Means vs GMM vs HDBSCAN; seed stability; PCA/UMAP 2D plot.
2. **[DONE] Criteria profile (extends Phase 0 features):** add positions gained, driver-error DNF rate, mechanical DNF rate; PCA/correlation redundancy check; teammate-relative versions to separate driver from car.
3. **Real DNF NLP:** TF-IDF + logistic regression replacing keywords, scored on a hand-labelled sample.
4. **Cross-era rating:** Elo or Bradley-Terry with driver + constructor effects (answers Fangio vs Verstappen).

## Phase 3: delivery
- Streamlit app on Streamlit Community Cloud reading a small precomputed driver-level table (check Kaggle license before committing derived data).
- `docs/METHODOLOGY.md` (one doc: data sheet + limitations + model notes) instead of separate model card and data sheet.
- Final README polish; portfolio post(s).

## Order
Phase 0 commit -> Phase 1 -> LinkedIn post (English draft already written; add repo link) -> Phase 2.1 -> 2.2 -> 2.4 -> 2.3 -> Phase 3.

## Verification
- `make test` green, coverage >= 80% on `src/`; `make lint` clean; CI green on GitHub.
- Notebook runs top to bottom in a fresh venv created from `pyproject.toml`.
- `make figures` output matches README numbers (legends list, Hamilton 80/107, Verstappen 63/107).
- Tag `v1-baseline` untouched; new work tagged `v2-era-neutral`.
