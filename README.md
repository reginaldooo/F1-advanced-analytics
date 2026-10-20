# 🏎️ F1 Advanced Analytics: Speed, Consistency & Reliability (1950-2024)

[![GitHub](https://img.shields.io/badge/GitHub-reginaldooo-181717?logo=github&style=flat-square)](https://github.com/reginaldooo)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-farrel--reginaldo-0A66C2?logo=linkedin&style=flat-square)](https://linkedin.com/in/farrel-reginaldo)
[![Kaggle](https://img.shields.io/badge/Data-Kaggle%20F1-20BEFF?logo=kaggle&style=flat-square)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&style=flat-square)](https://python.org)
[![CI](https://img.shields.io/github/actions/workflow/status/reginaldooo/F1-advanced-analytics/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/reginaldooo/F1-advanced-analytics/actions/workflows/ci.yml)

> **"How do you compare F1 drivers across eras without letting points systems and cars decide?"**

## 📖 Project Background & Motivation
Formula 1 is one of the most data rich sports in the world. However, historic analysis often suffers from **Recency Bias** (favoring current drivers) or **Nostalgia Bias** (favoring legends without context). Furthermore, looking solely at "Championship Titles" ignores the nuance of car reliability and consistency.

**The Problem:** How do we objectively compare drivers across different eras (e.g., Fangio vs. Verstappen) while accounting for car failures and driving styles?

**The Solution:** This project utilizes an advanced data driven approach:
1.  **Unsupervised Learning (K-Means)** to segment drivers based on performance metrics, not opinions.
2.  **Telemetry Analysis** (Lap Times) to measure driver consistency ("The Human Metronome").
3.  **Status classification**: a hand-labelled table of all 139 race statuses (with a trained TF-IDF fallback for unseen ones) separates "Bad Luck" (Mechanical Failure) from "Driver Error" (Crashes).

---

## 📊 Key Insights & Conclusions

After analyzing over 70+ years of race data, here are the key findings:

### 1. Speed $\neq$ Consistency
Consistency is measured as the standard deviation of each lap time relative to that race's median pace (laps above 107% of the median are dropped), so circuits are comparable.
* The most consistent drivers with 50+ races are **Jos Verstappen, Christian Klien, Alexander Wurz, Johnny Herbert** and **Mika Salo** (Gerhard Berger is 7th). Almost all are midfielders or backmarkers.
* **Mika Häkkinen** (93rd of 107), **Martin Brundle** (92nd) and **Michael Schumacher** (94th) rank near the bottom, the opposite of the earlier version of this analysis.
* **Hamilton** (rank 80 of 107) and **Max Verstappen** (rank 63 of 107) sit in the lower half on consistency but top the era-neutral performance ranking.
* **Insight:** A driver can be extremely consistent without being fast, and the fastest drivers are not the most metronomic.
* *Lap time data only exists from 1996, so this metric covers 107 drivers, not the full history.*

### 2. The "Legend" Tier is Exclusive
Using K-Means Clustering on *Average Finish Percentile*, *Podium Rate* and *Win Rate* (all era-neutral, scaled with `RobustScaler`), only 10 of 246 drivers fall into the **"Dominant Legends"** category: Fangio, Ascari, Moss, Clark, Stewart, Prost, Senna, Schumacher, Hamilton and Verstappen.
* **Why k=4:** silhouette and Davies-Bouldin favour k=2 (backmarkers vs everyone else), which is too coarse. k=4 is the finest split with silhouette above 0.5 (0.54) and near-perfect seed stability (ARI = 0.98); at k=5 silhouette drops to 0.43 and stability to 0.56. GMM scores lower, and HDBSCAN leaves 66 of 246 drivers (27%) unassigned, so the tiers are a segmentation of a continuum, not natural groups.
* Raw points were dropped because the scoring system changed many times (a win was worth 8 points in 1950 and 25 today), which skewed the earlier clusters.
* The model successfully identified high performing drivers who never won a championship as **"Podium Contenders,"** validating their skill level statistically despite the lack of titles.

### 3. Reliability Analysis: Bad Luck vs. Aggression
By classifying DNF (Did Not Finish) reasons, we tested historical reputations:
* **James Hunt:** Confirmed. Among top-tier drivers he has the highest share of races ending in **Accidents/Collisions** (21.5%).
* **Nigel Mansell:** *Not* confirmed as an outlier. About a third of his races (33%) ended in mechanical failure, but Peter Collins, Mario Andretti and others are higher. Mechanical attrition mostly tracks the era: older cars broke more, whoever drove them.
* *Correction:* the earlier keyword rules missed statuses such as Transmission, Turbo and Fuel system, which understated mechanical failures (Mansell 24.5% before, 32.8% now).

### 4. The Consistency Paradox
Interestingly, the drivers with the absolute highest statistical consistency (lowest pace variance) are rarely World Champions.
* **The "Safe Zone" Effect:** Extreme consistency often correlates with driving within a comfort zone (e.g. 95% capacity). To win titles like **Hamilton** or **Verstappen**, drivers must push the car to the absolute physical limit, which naturally introduces slight variance in lap times.
* **Strategic Variation:** Elite drivers frequently alter their pace intentionally to manage tire degradation or fuel loads. This strategic adaptability negatively impacts their "consistency score" in the raw data but is essential for winning races.

### 5. Which Criteria Actually Add Information?
Beyond finish percentile and win/podium rate, each driver now has racecraft (positions gained), reliability (driver-error and mechanical DNF rates) and a teammate-relative score (finish percentile minus the other driver in the same car, 20+ shared races).
* **Redundancy:** win rate, podium rate and finish percentile correlate at r = 0.66 to 0.86, and the first principal component explains 66% of the variance. They measure mostly the same thing.
* **Positions gained** correlates *negatively* with podium rate (r = -0.50): front-runners start near the front and have little to gain, so it reflects grid position as much as racecraft. It also correlates with mechanical DNF rate (r = 0.56): in unreliable eras many cars retire, so finishers gain places.
* **Mechanical DNF rate** is nearly independent of performance (|r| <= 0.11), a clean "car luck" axis.
* **Teammate delta** adds car-adjusted information (r = 0.43 to 0.52 with results). Top of 237 drivers: Jim Clark, Juan Fangio, Christian Fittipaldi, Emerson Fittipaldi, Pedro Rodriguez; Max Verstappen is 8th.
* *Caveat: in a dominant car both teammates finish near the top, which compresses the delta (Hamilton ranks 49th of 237). A low delta is not proof of a weaker driver.*

---

## 📈 Visualizations
*(Note: These images are generated by the notebook)*

| Performance Matrix | Reliability Analysis | Human Metronome |
| :---: | :---: | :---: |
| ![Matrix](images/performance-matrix.png) | ![Reliability](images/reliability-analysis.png) | ![Human Metronome](images/human-metronome.png) |

| Choosing k | Clusters in 2D (PCA) | Criteria correlation |
| :---: | :---: | :---: |
| ![k selection](images/k-selection.png) | ![PCA](images/clusters-pca.png) | ![Correlation](images/criteria-correlation.png) |

---

## 🛠️ Tech Stack & Methodology

* **Data Cleaning:** Handling 70 years of inconsistent data (e.g., treating `\N` values, normalizing constructor names).
* **Feature Engineering:**
    * *Average Finish Percentile*, *Podium Rate* & *Win Rate* for clustering (era-neutral, no raw points).
    * *Standard Deviation* of lap pace relative to the race median (~589k lap records, laps > 107% of median removed).
    * *Status classification*: 139 hand-labelled statuses (`src/f1/status_labels.csv`) plus a TF-IDF + logistic regression fallback for unseen statuses (leave-one-out accuracy 91% vs 32% for the old keyword rules).
* **Machine Learning:** K-Means Clustering (`sklearn`, `RobustScaler`, k=4) to create driver personas.
* **Visualization:** Seaborn & Matplotlib for high dimensional data representation.

---

## 🚀 How to Run Locally

1.  **Clone and install**
    ```bash
    git clone https://github.com/reginaldooo/F1-advanced-analytics.git
    cd F1-advanced-analytics
    pip install -e ".[dev]"
    ```
2.  **Data**
    * Nothing to download by hand: `f1.data.load_tables()` reads `./data`, then `$F1_DATA_DIR`, then downloads the [Kaggle dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) with `kagglehub`.
    * *(Data is not included in this repo.)*
3.  **Run**
    ```bash
    jupyter lab f1-advanced-analytics.ipynb   # explore
    make figures                              # re-run the notebook and regenerate images/
    make test                                 # pytest + coverage (>= 80%)
    make lint                                 # ruff
    ```

## 🗂️ Project Structure

```
src/f1/        data loading, features, clustering + model selection, criteria profile, DNF classifier, charts
tests/         pytest suite (synthetic fixtures, no downloads)
docs/          METHODOLOGY.md (data, features, limitations)
baseline/      frozen v1 notebook and README (tag v1-baseline)
f1-advanced-analytics.ipynb   storytelling notebook, imports from src/f1
```

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for features and limitations and [`ROADMAP.md`](ROADMAP.md) for what is next.

---

*Project by Reginaldo Ahnaf*