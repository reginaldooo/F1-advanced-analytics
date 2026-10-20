# Baseline (v1): state before the upgrade

Snapshot of the project as of commit `78f7c3b` (2026-01-29). Use this as the "before" when writing the article about the update. Frozen copies live in `baseline/` (notebook, README) and `images/baseline-v1/` (charts).

## What v1 is
- One notebook, `f1-advanced-analytics.ipynb`, 13 cells (7 code, 6 markdown), plus README and 3 charts.
- Data: Kaggle F1 CSVs (`drivers`, `results`, `status`, `lap_times`). Lap times loaded: 589,081 rows.
- No tests, no `src/`, no pinned environment, no experiment tracking.

## Methodology in v1
| Area | What v1 does | Cell |
|---|---|---|
| Points | Raw `points` from `results.csv`, summed per driver. Not normalized across scoring eras. | 4, 6 |
| Driver filter | More than 20 races | 6 |
| Features | `Points Per Race` = total points / races; `Win Rate` = wins / races | 6 |
| Clustering | `StandardScaler` + K-Means, k=4 hardcoded, `n_init=10`, `random_state=42`. No silhouette/elbow check. | 6 |
| Cluster names | Ranked by mean Win Rate: Dominant Legends, Podium Contenders, Midfielders, Backmarkers | 6 |
| Consistency | Std of lap `milliseconds` per driver-race (laps under 150,000 ms only), averaged per driver. Raw ms, not scaled by lap time or circuit. | 8 |
| DNF "NLP" | Keyword rules on `status`: Finished / Accident-Driver Error / Car-Mechanical Failure / Other. No trained model, no evaluation. | 8 |
| Charts | Human Metronome (top 10 consistent, >50 races), Reliability stacked bar (top clusters), Performance Matrix scatter (Points Per Race vs Win Rate) | 10, 12 |

## Known limitations (what the update fixes)
1. Points are raw across eras (8-10 for a win early on, 25 from 2010), which inflates the modern x-axis and skews clusters. Raised by a reader.
2. Consistency is raw ms, so track length and era affect it.
3. k=4 is a choice, not validated.
4. DNF "NLP" is keyword matching, not NLP.
5. Driver and car effects are mixed together (no teammate comparison).
6. README says "millions of rows" for lap times; the loaded file has 589,081.
7. README run step names `F1_Advanced_Analytics.ipynb`, but the file is `f1-advanced-analytics.ipynb`.

## v1 headline claims (README, to re-check after the update)
- Speed is not consistency; Hamilton and Verstappen appear as high-speed, high-consistency outliers.
- Berger and Hakkinen show the lowest lap-time std.
- Only a few drivers land in "Dominant Legends".
- James Hunt: high share of accident DNFs. Nigel Mansell: high share of mechanical DNFs.
- Highest-consistency drivers are rarely multi-title champions.

## Before/after checklist for the article
- [x] Re-run with era-neutral performance (finish percentile, podium/win rate) instead of raw points; compare against `images/baseline-v1/performance-matrix.png`.
- [x] Which v1 claims survive, which flip:
  - Survives: only 10 of 246 drivers are "Dominant Legends" (Moss in, Vettel out); Hunt/Mansell DNF reading unchanged.
  - Changes: Midfielders 14 -> 51, Backmarkers 201 -> 169 (clusters more balanced).
  - Flips: Berger/Hakkinen are no longer the most consistent (Berger is 7th among 50+ race drivers, Hakkinen is 93rd of 107, Brundle 92nd); Hamilton ranks 80/107 and Verstappen 63/107 on consistency.
  - Caveat: lap times exist only from 1996, so consistency covers 107 drivers.

## Update 2: non-starters excluded (data-quality fix)
`results.csv` includes rows for drivers who never started (Did not qualify 1,025, Did not prequalify 331, 107% rule, grid-less withdrawals). They were counted as last place. They are now dropped in `build_master` (`f1.dnf.non_starter_mask`).

| | Before | After |
|---|---|---|
| Result rows | 26,759 | 25,211 |
| Drivers with 20+ races | 257 | 246 |
| Dominant Legends | 10 (same names) | 10 (same names) |
| Backmarkers / Midfielders / Podium Contenders | 169 / 51 / 27 | 151 / 54 / 31 |
| Hamilton / Verstappen consistency rank | 80 / 63 of 107 | unchanged |
| Top-10 most consistent | Jos Verstappen ... Coulthard | unchanged |
| k=4 silhouette / seed stability | 0.56 / 1.00 | 0.54 / 0.98 (k=5: 0.43 / 0.56) |
| HDBSCAN noise | 129 of 257 | 66 of 246 |

## Update 3: DNF status labels
Hand-labelled all 139 statuses; the old keyword rules matched 32% of them (missing Transmission, Turbo, Fuel system, Puncture, +6 Laps and above). Mechanical DNF rate: Hunt 15.1% -> 29.0%, Mansell 24.5% -> 32.8%. The v1 claim "Mansell suffered disproportionately from mechanical failures" no longer holds as an outlier (Collins, Andretti and others are higher); the Hunt claim holds.
