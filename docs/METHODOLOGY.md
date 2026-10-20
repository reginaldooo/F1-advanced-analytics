# Methodology, data and limitations

## Question
How can drivers from different eras (e.g. Fangio vs Verstappen) be compared fairly?

## Data
Kaggle "Formula 1 World Championship (1950-2020)" by Rohan Rao (`drivers`, `results`, `status`, `lap_times`).
Not redistributed in this repo; `f1.data.load_tables()` reads `./data`, `$F1_DATA_DIR`, or downloads it with `kagglehub`.
Lap times start in 1996 (589,081 rows), so the consistency metric covers 107 modern drivers only.

## Features (era-neutral)
- **Finish percentile:** `1 - (positionOrder - 1) / (starters - 1)` per race, averaged per driver. Independent of the points system and grid size.
- **Win, podium and top-5 rate.**
- **Pace consistency:** each lap divided by its race median; laps above 107% of the median are dropped; std per driver-race, averaged per driver. Independent of circuit length.
- Raw points are not used because scoring changed many times (a win was 8 points in 1950, 25 today).

## Model
K-Means (k=4, `RobustScaler`) on finish percentile, podium rate and win rate for drivers with more than 20 races. Clusters are named by mean win rate. k=4 was validated in `f1.clustering`: silhouette, Davies-Bouldin, elbow and seed stability (mean ARI over 10 seeds) for k=2..10, plus GMM and HDBSCAN comparisons. Silhouette/DB favour k=2 (too coarse); k=4 is the finest split with silhouette > 0.5 (0.54) and ARI = 0.98 (k=5: silhouette 0.43, ARI 0.56). GMM scores lower; HDBSCAN marks 66 of 246 drivers as noise, so the tiers segment a continuum rather than reveal natural groups. UMAP was skipped; PCA is used for the 2D view.

## Criteria profile (`f1.profile`)
Per driver (more than 20 races): finish percentile, win and podium rate, positions gained (grid minus finish, classified finishes from a real grid slot only), driver-error and mechanical DNF rate (share of starts, from the keyword classifier), and teammate delta (finish percentile minus the other driver(s) in the same car and race, kept only with 20+ shared races; 237 of 246 drivers qualify).
Redundancy check: win, podium and finish percentile are highly correlated (r 0.66-0.86; PC1 = 66% of variance). Positions gained is negatively correlated with podium rate (r = -0.50) because front-runners start near the front, and positively with mechanical DNF rate (r = 0.56) because retirements let finishers gain places. Mechanical DNF rate is almost independent of performance. Teammate delta correlates moderately (0.43-0.52) with results. The profile is for analysis only; clustering still uses the three features above.

## DNF reasons (`f1.dnf`)
All 139 distinct `status` values are hand-labelled (`src/f1/status_labels.csv`) as Finished, Accident/Driver Error, Car/Mechanical Failure, Did Not Start or Other. The pipeline uses this table. The earlier keyword rules matched only 32% of the distinct statuses (missing Transmission, Turbo, Fuel system, Puncture and every +6 Laps or more); they are kept as a baseline. A TF-IDF (character n-grams) + logistic regression model trained on the table is the fallback for statuses not in it: leave-one-out accuracy 91% vs 32% for keywords. Its frequency-weighted accuracy (67% vs 89%) is lower only because leave-one-out removes the single `Finished` status (7,674 results), so the table, not the model, is used for known statuses. Effect: mechanical DNF rates rise (e.g. Mansell 24.5% -> 32.8%, Hunt 15.1% -> 29.0%).

## Limitations
- Finish percentile mixes driver and car; teammate delta reduces this but is compressed in dominant cars (both drivers finish near the top) and mixes shared drives and more than two drivers per team in the 1950s.
- Positions gained mostly reflects starting position, not pure racecraft.
- DNF labels are hand-written by the author (open to review); ambiguous ones (Damage, Debris, Physical, Retired, Stalled, seat-related) are labelled Other. The trained fallback was only validated with leave-one-out on the 139 known statuses.
- A driver with few races at a top team can be over-rated; the 20-race filter only partly helps.
- Results describe the dataset; they are not causal claims about driver skill.
