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
K-Means (k=4, `RobustScaler`) on finish percentile, podium rate and win rate for drivers with more than 20 races. Clusters are named by mean win rate. k=4 was validated in `f1.clustering`: silhouette, Davies-Bouldin, elbow and seed stability (mean ARI over 10 seeds) for k=2..10, plus GMM and HDBSCAN comparisons. Silhouette/DB favour k=2 (too coarse); k=4 is the finest split with silhouette > 0.5 and ARI = 1.0. GMM scores lower; HDBSCAN marks 129 of 257 drivers as noise, so the tiers segment a continuum rather than reveal natural groups. UMAP was skipped; PCA is used for the 2D view.

## Limitations
- Finish percentile still mixes driver and car; teammate-relative metrics are planned.
- The DNF classifier is keyword matching, not a trained model.
- A driver with few races at a top team can be over-rated; the 20-race filter only partly helps.
- Results describe the dataset; they are not causal claims about driver skill.
