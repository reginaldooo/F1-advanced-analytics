# Roadmap and plan (final state)

## Status
| Step | Status | Commit |
|---|---|---|
| Era-neutral features, pace consistency | done | `74b693e` |
| Package, tests, CI, tooling | done | `56fb74e` |
| 2.1 k validation, GMM / HDBSCAN comparison | done | `441b8c8` |
| 2.2 criteria profile (teammate delta, DNF rates) | done | `2b5c811` |
| 2.0 exclude non-starters | done | `12868d2` |
| 2.3 hand-labelled DNF statuses + trained fallback | done | `9026d81` |
| 2.4 cross-era Bradley-Terry rating | done | `2acba7b` |
| Phase 3 Streamlit app (built, tested locally) | done | `9a43031`, `22686a8` |
| Rescored-points (single points system) sensitivity check | done | - |
| Deploy on Streamlit Community Cloud | **manual**, see `docs/DEPLOY.md` | - |
| Add app URL to README and LinkedIn post | after deploy | - |

Not planned (comparison-only project): supervised podium/DNF model, SHAP, MLflow, DVC, Evidently, Docker, Optuna.
Deviations from the plan below: one `app/data/drivers.csv` instead of two files (PCA coordinates are columns in it); the rating's naive baseline is a constant 0.5 plus a driver-only model; UMAP skipped (PCA only).

---

# F1-advanced-analytics: finish Phase 2 (data fix, DNF model, cross-era rating) and ship the Streamlit app

## Context
Repo: `/Users/reginaldo/Documents/Projects/me/F1-advanced-analytics` (`main` at `73b9257`, CI green, 37 tests, 97% coverage; `.venv` inside). Done so far: era-neutral features, k validation, criteria profile. Remaining roadmap items: 2.4 cross-era rating, 2.3 DNF NLP, Phase 3 delivery. User wants all of it through "Streamlit up". Everything below is TDD (tests first), ruff clean, one commit per step, push after each, CI green before moving on.

Read-only exploration found two things that change the order of work:
1. **Data-quality bug in the existing features.** `results.csv` contains 1,025 "Did not qualify" and 331 "Did not prequalify" rows (plus "Withdrew" 248, "Not classified" 172). `add_finish_pct` counts them in `starters` and `Total Races`, so they act as last-place finishes for cars that never started. This biases `finish_pct`, `Avg Finish Pct`, the >20-race filter and possibly cluster membership.
2. **The keyword DNF classifier is incomplete.** `+6 Laps` and above, `Transmission` (321), `Turbo` (146), `Fuel system` (155), `Did not qualify` all fall into "Other". Mechanical DNF rates in the criteria profile are understated, and a rating model that drops mechanical retirements would be wrong without a better classifier.

So the order is: 2.0 data fix -> 2.3 DNF classifier -> re-run profile -> 2.4 rating -> app. The LinkedIn post (`me/Linkedin/4-...md`, still `Post on: TBD`) must be re-checked against the final numbers before publishing; the 10-legends / 80th / 63rd claims may shift.

## Step 2.0: exclude non-starters (small, do first)
- Add `f1.dnf.is_non_start(status)` for: Did not qualify, Did not prequalify, Withdrew, Not classified? (decide: Not classified = started, classified last; keep). Non-starters = DNQ, DNPQ, Withdrew, Excluded, Did not start-type statuses; also require `grid` not NaN where available.
- `build_master` drops non-starters before `add_finish_pct`.
- Tests: synthetic race with 3 starters + 2 DNQ -> `starters` = 3, winner pct 1, last 0, DNQ rows absent.
- Re-run pipeline and **quantify impact** before/after: legends list, cluster sizes, Hamilton/Verstappen consistency ranks, top-10 consistency (consistency uses laps, unaffected except driver filter). Record in `BASELINE.md`. If claims change, update README, METHODOLOGY, and the draft LinkedIn post (do not publish stale numbers).

## Step 2.3: DNF status classifier (trained, evaluated)
- Only 139 distinct `status` strings exist, so the work is labelling, not big NLP. Create `data/labels/status_labels.csv` equivalent committed as `src/f1/status_labels.csv` (not gitignored): all 139 statuses hand-labelled into `Finished`, `Accident/Driver Error`, `Car/Mechanical Failure`, `Did Not Start`, `Other`. I draft labels from F1 knowledge (Puncture, Wheel, Tyre -> mechanical/tyre decision documented; Spun off, Collision, Accident -> driver error; Disqualified, Illness, etc. -> Other). **Labels are model-written; user spot-checks the ambiguous ones** (list them in the PR/commit message).
- `f1/dnf.py`: keep `classify_dnf` as the keyword baseline; add `train_status_classifier()` (TF-IDF char n-grams + LogisticRegression) and `predict_reason()`. Evaluate with stratified cross-validation on distinct statuses (macro-F1, per-class report) and **frequency-weighted accuracy over `results`** for baseline vs trained.
- Decision rule: use the trained model in the pipeline only if it beats the keyword baseline on weighted accuracy and macro-F1; otherwise keep rules plus the corrected label table. Either way fix the missing keywords (laps-down finishes, transmission, turbo, fuel).
- Persist nothing binary: model retrains in seconds from the CSV; app never needs it.
- Re-run criteria profile with the new reasons; update README insight 3 (Hunt/Mansell) and insight 5 numbers.

## Step 2.4: cross-era driver rating (Bradley-Terry with car effect)
- Model: per race, every pair (i, j) of starters where i finished ahead of j. Features = onehot(driver i) - onehot(driver j) + onehot(constructor-season i) - onehot(constructor-season j). L2-regularised logistic regression without intercept (scikit-learn, sparse matrices; ~280k pairs). Driver coefficient = skill (constant over career), constructor-season coefficient = car that year. Teammates and drivers who change teams identify the two effects.
- Pair rules: exclude non-starters and mechanical retirements (from 2.3), keep driver-error DNFs as losses, ties impossible (positionOrder). No pairs for races with fewer than 2 starters.
- Uncertainty: bootstrap over races (30-50 resamples), report rating with 5-95% interval; drivers with few races get wide intervals. Only drivers with more than 20 races appear in the comparison table.
- Validation (still "comparison only", no prediction product): GroupKFold by race, held-out pairwise log-loss and accuracy for (a) constructor-season only, (b) driver + constructor, (c) naive baseline (higher finish percentile history). The driver term must improve held-out log-loss, otherwise report that driver skill is not identifiable and say so.
- Files: `src/f1/rating.py` (`build_pairs`, `fit_rating`, `bootstrap_ratings`, `cross_validate`), `tests/test_rating.py` with synthetic data where a known strong driver in a weak car must come out on top and a weak driver in a dominant car must not.
- Notebook section 9, figure `images/rating.png` (top drivers with intervals; Fangio vs Verstappen highlighted), README insight, METHODOLOGY (era linkage is weak for the 1950s: ratings across eras rest on drivers whose careers overlap, so intervals are wide).

## Phase 3: Streamlit app (Community Cloud)
- Build step: `src/f1/app_data.py` + `make app-data` produce one small committed table `app/data/drivers.csv` (profile, cluster, pace consistency, rating with interval, teammate delta) and `app/data/pca.csv` (2D coordinates). `.gitignore` currently ignores `*.csv` and `data/`, so add explicit `!app/data/*.csv`. Tests for the builder (columns, no NaN in key fields, row count equals filtered drivers).
- `app/streamlit_app.py` in the style of `me/pit-stop-risk-engine/app.py` (wide layout, sidebar with title/filters/"Built by Reginaldo Ahnaf" + links, Plotly, `st.cache_data`): tabs **Driver Explorer** (filters by era/cluster/min races, sortable table), **Cluster View** (PCA scatter coloured by tier), **Compare** (two drivers: criteria bars, rating with interval, teammate delta), **Methodology** (method, caveats: 1996+ laps, tiers = continuum, teammate delta compression).
- `app/requirements.txt` with pinned `streamlit`, `pandas`, `plotly` only (the app reads precomputed CSVs, no `f1` package or Kaggle access needed).
- Local check: `streamlit run app/streamlit_app.py --server.headless true`, `curl` the health endpoint, and a headless smoke test with `streamlit.testing.v1.AppTest` (no exceptions on each tab, both compare selections).
- Licence: the source data derives from Ergast (CC BY-NC-SA 3.0 via the Kaggle dataset). Verify the Kaggle licence text before committing derived data; add attribution in the app footer and README; the project stays non-commercial.
- **Deployment cannot be done from here**: Streamlit Community Cloud needs the user to sign in with GitHub in the browser (share.streamlit.io, New app, repo `reginaldooo/F1-advanced-analytics`, branch `main`, main file `app/streamlit_app.py`). I will prepare everything, verify locally, and hand over exact steps; after the user deploys, I add the app URL to README and the LinkedIn post and re-check the live page.

## Order and commits
2.0 `fix: exclude non-starters from finish percentile` -> 2.3 `feat: trained DNF status classifier` -> `docs: refresh profile numbers` -> 2.4 `feat: cross-era Bradley-Terry rating` -> Phase 3 `feat: precomputed app data` -> `feat: streamlit app` -> `docs: app link, licence, final numbers`. Update `ROADMAP.md` ticks and the LinkedIn draft at the end.

## Verification
- `pytest` green with coverage >= 80%, `ruff check .` clean, CI green after every push (check with `gh run list`).
- 2.0: before/after table in `BASELINE.md`; no DNQ/DNPQ rows in `master`.
- 2.3: classification report printed in the notebook; weighted accuracy of trained vs keyword baseline recorded in METHODOLOGY.
- 2.4: held-out log-loss table (constructor-only vs driver+constructor vs naive); synthetic recovery test passes; ratings table sanity (Clark/Fangio/Schumacher/Hamilton/Verstappen in plausible order with intervals).
- App: AppTest passes, local server answers, all numbers in README, app and `me/Linkedin/4-...md` match `app/data/drivers.csv`.
- Final: user deploys on Streamlit Cloud; live URL loads all four tabs.
