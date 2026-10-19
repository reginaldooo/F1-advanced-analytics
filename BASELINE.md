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
- [ ] Re-run with points normalized to one scale; compare cluster membership and the scatter plot against `images/baseline-v1/performance-matrix.png`.
- [ ] Which v1 claims survive, which flip.
- [ ] Silhouette-chosen k vs the hardcoded 4.
- [ ] Coefficient-of-variation ranking vs raw-std ranking.
- [ ] Teammate-relative comparison vs raw comparison.

## Optional: tag it
`git tag v1-baseline 78f7c3b` marks this commit locally (not run).
