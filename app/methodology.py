"""Methodology tab: what each step does, the formula, the reason and how good it is.

Every number comes from app/data/metrics.json (built by `make app-data` from src/f1).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import ARCTIC_WHITE, PAPAYA, RACE_BLUE

SUB_TABS = ["Overview", "Features", "Tiers", "Reliability", "Rating", "Limits & Data"]
REPO_URL = "https://github.com/reginaldooo/F1-advanced-analytics"
DATA_URL = "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020"


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _section(title: str) -> None:
    st.markdown(f"##### {title}")


def render(metrics: dict) -> None:
    tabs = st.tabs(SUB_TABS)
    for tab, draw in zip(
        tabs, (overview, features, tiers, reliability, rating, limits_and_data)
    ):
        with tab:
            draw(metrics)


def overview(metrics: dict) -> None:
    data = metrics["data"]
    st.markdown(
        "**Question.** How do you compare F1 drivers across eras without letting points systems "
        "and cars decide?"
    )
    _section("What happens")
    st.markdown(
        f"""
1. **Features.** Every result row is turned into era-neutral numbers (finish percentile, win and
   podium rate, pace consistency, teammate delta).
2. **Tiers.** K-Means groups the {data['drivers_profiled']} drivers with more than 20 starts
   into four tiers.
3. **Reliability.** Every race status is labelled driver error or mechanical failure.
4. **Rating.** A Bradley-Terry model scores drivers across eras with the car effect removed.
"""
    )
    _section("Which number means what")
    st.markdown(
        "Only two steps can be scored against a ground truth, so only they get an **accuracy**:\n\n"
        "- **Reliability** is a classifier checked against hand labels (leave-one-out accuracy).\n"
        "- **Rating** predicts which of two drivers finishes ahead, checked on races it never "
        "saw.\n\n"
        "**Tiers are unsupervised.** There is no right answer to compare with, so they get "
        "*validity* scores (silhouette, Davies-Bouldin, seed stability) instead. "
        "A high silhouette says the groups are well separated, not that they are *true*."
    )
    _section("Data")
    cols = st.columns(3)
    cols[0].metric("Race results (incl. non-starters)", f"{data['results_rows']:,}")
    cols[1].metric("Lap times (1996+)", f"{data['lap_rows']:,}")
    cols[2].metric("Drivers profiled", data["drivers_profiled"])


def features(metrics: dict) -> None:
    data = metrics["data"]
    corr = metrics["profile"]["correlation"]
    st.markdown(
        "Every feature is built so that a 1950 race and a 2024 race are comparable."
    )
    _section("Finish percentile")
    st.latex(r"\text{finish\_pct}=1-\frac{\text{position}-1}{\text{starters}-1}")
    st.markdown(
        "1 for the winner, 0 for the last starter, averaged per driver. "
        "**Why:** it ignores the points system and the grid size. "
        "Drivers who never started (DNQ, DNPQ, 107% rule) are removed first, otherwise they "
        "would count as last place in races they never ran."
    )
    _section("Win, podium and top-5 rate")
    st.latex(r"\text{rate}=\frac{\text{wins (or podiums, top-5s)}}{\text{races}}")
    st.markdown(
        "**Why not points:** a win paid 8 points in 1950 and 25 today, and the scale changed many "
        "times in between. Raw points would reward the era, not the driver."
    )
    _section("Sensitivity check: one points system")
    rescored = metrics["profile"]["rescored_correlation"]
    st.markdown(
        "Suggested by Ian Wright: put every race on one points system. Every race in the dataset "
        "(1950-2024) is rescored with the same modern scale (25-18-15-12-10-8-6-4-2-1, no "
        "fastest-lap point, no sprints), then averaged per race. It is **not** used for the "
        "tiers. Correlation with the clustering features: "
        f"finish percentile **{rescored['Avg Finish Pct']:.2f}**, "
        f"podium rate **{rescored['Podium Rate']:.2f}**, "
        f"win rate **{rescored['Win Rate']:.2f}**. "
        "The podium figure is partly by construction, since both reward the top three. "
        "Rank order agrees closely (Spearman 0.94, 0.93 and 0.84), so the ranking is not an "
        "artefact of dropping points. The tier *edges* are less stable. K-Means on rescored "
        "points alone gives an adjusted Rand index of 0.62 against the published tiers. Nine of "
        "the ten Dominant Legends stay. Moss drops out, and Farina, Gonzalez, Hawthorn and "
        "Vettel move in."
    )
    _section("Pace consistency (1996 onwards)")
    st.latex(
        r"\text{pace}_{lap}=\frac{t_{lap}}{\operatorname{median}_{race}(t)},\quad "
        r"\text{drop laps with pace}>1.07,\quad "
        r"\text{Pace\_StdDev}=\operatorname{mean}_{races}\big(\operatorname{std}_{laps}(\text{pace})\big)"
    )
    st.markdown(
        "**Why:** dividing by the race median makes circuits of different length comparable. "
        "The 107% cut removes pit-in/out laps, safety cars and slow laps after incidents. "
        "Lap times only exist from 1996, so this covers "
        f"**{data['drivers_with_consistency']}** drivers."
    )
    _section("Positions gained and teammate delta")
    st.latex(r"\text{gained}=\text{grid}-\text{finish}")
    st.latex(
        r"\Delta_{team}=\text{finish\_pct}_{driver}-\operatorname{mean}(\text{finish\_pct}_{teammates})"
    )
    st.markdown(
        "Teammate delta compares a driver with the other driver(s) in the same car in the same "
        "race, so the car cancels out. It is kept only with 20+ shared races "
        f"(**{data['drivers_with_teammate_delta']}** of {data['drivers_profiled']} drivers). "
        "Positions gained is measured on classified finishes from a real grid slot."
    )
    _section("How good: do these criteria add information?")
    share = metrics["profile"]["pc1_share"]
    st.markdown(
        f"The first principal component explains **{_pct(share)}** of the variance of the seven "
        f"criteria, so some are redundant. Win rate vs finish percentile: "
        f"r = {corr['Win Rate']['Avg Finish Pct']:.2f}; podium rate vs finish percentile: "
        f"r = {corr['Podium Rate']['Avg Finish Pct']:.2f}. Mechanical DNF rate is almost "
        "independent of "
        f"performance (|r| up to {_max_abs(corr, 'Mechanical DNF Rate'):.2f}), a clean "
        "\"car luck\" axis."
    )


def _max_abs(corr: dict, name: str) -> float:
    return max(abs(value) for key, value in corr[name].items() if key != name)


def tiers(metrics: dict) -> None:
    clustering = metrics["clustering"]
    k_table = pd.DataFrame(clustering["k_table"]).set_index("k")
    chosen = clustering["chosen_k"]
    row = k_table.loc[chosen]
    st.markdown(
        "Drivers are grouped into four tiers named by mean win rate: Dominant Legends, "
        "Podium Contenders, Midfielders, Backmarkers."
    )
    _section("Formula")
    st.latex(r"x'=\frac{x-\operatorname{median}(x)}{\operatorname{IQR}(x)}\quad\text{(RobustScaler)}")
    st.latex(r"\min_{C}\sum_{k=1}^{K}\sum_{x\in C_k}\lVert x-\mu_k\rVert^2\quad\text{(K-Means)}")
    st.latex(r"s=\frac{b-a}{\max(a,b)}\quad\text{(silhouette)}")
    st.markdown(
        "Features: average finish percentile, podium rate, win rate. "
        "**Why robust scaling:** a handful of extreme drivers (Fangio, Schumacher) would "
        "otherwise stretch the scale. **Why these three:** all are era-neutral. "
        "Raw points were dropped for the reason in *Features*."
    )
    _section("How good (validity, not accuracy)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Silhouette (k={chosen})", f"{row['silhouette']:.2f}", help="-1 to 1, higher is better"
    )
    c2.metric("Davies-Bouldin", f"{row['davies_bouldin']:.2f}", help="Lower is better")
    c3.metric(
        "Seed stability (ARI)",
        f"{row['stability']:.2f}",
        help="1 = identical groups for every random seed",
    )
    st.plotly_chart(_k_chart(k_table, chosen), use_container_width=True)
    _section("Why k = 4")
    st.markdown(
        "Silhouette and Davies-Bouldin favour k = 2 (backmarkers vs everyone else), which is too "
        f"coarse. k = {chosen} is the finest split with silhouette above 0.5 "
        f"({row['silhouette']:.2f}) and near-perfect stability (ARI {row['stability']:.2f}). "
        f"At k = 5 silhouette drops to {k_table.loc[5, 'silhouette']:.2f} and stability to "
        f"{k_table.loc[5, 'stability']:.2f}."
    )
    algorithms = pd.DataFrame(clustering["algorithms"]).set_index("method")
    noise = int(algorithms.loc["HDBSCAN", "noise"])
    share = noise / clustering["n_drivers"]
    _section("Other algorithms")
    st.dataframe(
        algorithms.rename(columns={"davies_bouldin": "Davies-Bouldin", "silhouette": "Silhouette"})
        .style.format({"Silhouette": "{:.2f}", "Davies-Bouldin": "{:.2f}"}, na_rep="n/a"),
        width="stretch",
    )
    st.markdown(
        f"GMM scores lower than K-Means. HDBSCAN scores well only because it leaves **{noise} of "
        f"{clustering['n_drivers']} drivers ({_pct(share)})** unassigned and is scored on the "
        "rest. "
        "**So the tiers segment a continuum; they are not natural groups.**"
    )


def _k_chart(k_table: pd.DataFrame, chosen: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_table.index, y=k_table["silhouette"], name="Silhouette",
                             mode="lines+markers", line={"color": PAPAYA}))
    fig.add_trace(go.Scatter(x=k_table.index, y=k_table["stability"], name="Seed stability (ARI)",
                             mode="lines+markers", line={"color": RACE_BLUE}))
    fig.add_vline(x=chosen, line_dash="dash", line_color=ARCTIC_WHITE)
    fig.update_layout(height=320, xaxis_title="k (number of tiers)", yaxis_range=[0, 1.05],
                      margin={"l": 10, "r": 10, "t": 20, "b": 10})
    return fig


def reliability(metrics: dict) -> None:
    rel = metrics["reliability"]
    models = pd.DataFrame(rel["classifiers"]).set_index("method")
    keywords, trained = models.loc["Keywords"], models.loc["Trained (leave-one-out)"]
    counts = rel["label_counts"]
    st.markdown(
        "Every result has a status such as *Engine*, *Collision* or *+2 Laps*. "
        "To separate **bad luck** (mechanical failure) from **driver error**, each status needs "
        "a class."
    )
    _section("What happens")
    st.markdown(
        f"1. All **{rel['n_statuses']}** distinct statuses are labelled by hand: "
        + ", ".join(f"{name} {n}" for name, n in sorted(counts.items()))
        + ". The pipeline uses this table.\n"
        "2. For a status the table has never seen, a model trained on the table predicts the class."
    )
    st.latex(
        r"\text{TF-IDF}(\text{character 2-4 grams})\;\rightarrow\;"
        r"\text{logistic regression}\ (C=10,\ \text{balanced})"
    )
    _section("Why")
    st.markdown(
        "Simple keyword rules (\"engine\", \"collision\") miss statuses such as Transmission, "
        "Turbo, Fuel system, Puncture and every \"+6 Laps\" or more. The effect is large: "
        "Mansell's mechanical DNF rate rises from 24.5% to 32.8% once they are caught."
    )
    _section("How good (leave-one-out against the hand labels)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trained model accuracy", _pct(trained["accuracy"]),
              delta=f"{(trained['accuracy'] - keywords['accuracy']) * 100:+.0f} pts vs keywords")
    c2.metric("Keyword rules accuracy", _pct(keywords["accuracy"]))
    c3.metric("Trained macro-F1", f"{trained['macro_f1']:.2f}", help="Averages all classes equally")
    st.dataframe(
        models.rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1",
                               "weighted_accuracy": "Frequency-weighted accuracy"})
        .style.format("{:.1%}", subset=["Accuracy", "Frequency-weighted accuracy"])
        .format("{:.2f}", subset=["Macro-F1"]),
        width="stretch",
    )
    st.markdown(
        "**Why weighted accuracy is lower for the trained model:** leave-one-out removes the "
        "single *Finished* status, which covers thousands of results, so the model has never seen "
        "it. That is why known statuses always use the table, not the model. "
        "**Caveat:** the labels are the author's own and open to review; the model was only "
        "validated against those same labels."
    )


def rating(metrics: dict) -> None:
    rate = metrics["rating"]
    cv = pd.DataFrame(rate["cv"]).set_index("model")
    best = cv.loc["Driver + car"]
    st.markdown(
        "A rating that separates driver skill from the car, so drivers from different eras can "
        "be placed on one scale."
    )
    _section("Formula (Bradley-Terry)")
    st.latex(r"P(i\ \text{finishes ahead of}\ j)=\sigma(s_i-s_j)")
    st.latex(r"s=\text{driver skill}+\text{constructor-season effect}")
    st.latex(
        r"\min\ \frac{1}{2C}\lVert w\rVert^2+\sum_{\text{pairs}}\log\!\big(1+e^{-y(s_i-s_j)}\big)"
        r"\quad\text{(L2-regularised logistic)}"
    )
    st.latex(r"\text{rating}=\text{skill}\times\frac{400}{\ln 10}\quad\text{(Elo-like)}")
    _section("What happens")
    st.markdown(
        f"- Every pair of starters in a race is one observation: **{rate['n_pairs']:,} pairs**, "
        f"**{rate['n_drivers']}** drivers, **{rate['n_car_seasons']:,}** constructor-seasons.\n"
        "- Each car-season gets its own effect, so teammates and drivers who change teams "
        "separate skill from car.\n"
        "- Mechanical retirements are removed (not a skill signal). Driver-error retirements "
        "lose to every finisher ahead.\n"
        "- Intervals are 5-95% from 30 resamples of whole races."
    )
    _section("Why")
    st.markdown(
        "Skill is assumed constant over a career, which is what links 1950s and modern drivers "
        "through overlapping careers. "
        f"The regularisation C = {rate['c']} was chosen by held-out log-loss over C in "
        "{0.03, 0.1, 0.3, 1}."
    )
    _section(f"How good ({rate['folds']}-fold, split by race, pairs the model never saw)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pairwise accuracy", _pct(best["accuracy"]), help="Chance is 50%")
    c2.metric("Log-loss", f"{best['log_loss']:.3f}", help="Chance is 0.693; lower is better")
    c3.metric("Car only accuracy", _pct(cv.loc["Car only", "accuracy"]))
    st.plotly_chart(_cv_chart(cv), use_container_width=True)
    st.markdown(
        f"The driver term adds information on top of the car "
        f"(log-loss {cv.loc['Car only', 'log_loss']:.3f} to {best['log_loss']:.3f}), but "
        f"**{_pct(best['accuracy'])} is not high**: a car-only model already gets "
        f"{_pct(cv.loc['Car only', 'accuracy'])}. Finishing order has a lot of luck in it."
    )
    st.markdown(
        "**Robust:** Fangio, Stewart and Clark stay in the top 3 for every C tested, and "
        "Verstappen in the top 6. **Not robust:** Hamilton (3rd to 28th), Schumacher "
        "(5th to 31st) and Senna (11th to 50th) across C, because drivers who always had dominant "
        "cars cannot be separated from them. Read the list as tiers."
    )


def _cv_chart(cv: pd.DataFrame) -> go.Figure:
    colors = [ARCTIC_WHITE, RACE_BLUE, RACE_BLUE, PAPAYA]
    fig = go.Figure(go.Bar(x=cv.index, y=cv["log_loss"], marker_color=colors,
                           text=[f"{v:.3f}" for v in cv["log_loss"]], textposition="outside"))
    fig.update_layout(height=320, yaxis_title="Held-out log-loss (lower is better)",
                      yaxis_range=[0.4, 0.75], margin={"l": 10, "r": 10, "t": 20, "b": 10})
    return fig


def limits_and_data(metrics: dict) -> None:
    _section("Limitations")
    st.markdown(
        """
- Finish percentile mixes driver and car. Teammate delta reduces this, but is compressed in
  dominant cars (both drivers finish near the top) and mixes shared drives in the 1950s.
- Positions gained mostly reflects starting position, not pure racecraft.
- Ratings assume constant skill and link eras only through overlapping careers (wide 1950s
  intervals). Elite drivers in dominant cars are weakly identified.
- DNF labels are hand-written by the author. Ambiguous ones (Damage, Debris, Physical, Retired,
  Stalled, seat-related) are labelled *Other*.
- A driver with few races at a top team can be over-rated; the 20-race filter only partly helps.
- The rescored-points check scores only the top 10. About half of 1950s races have fewer than 10
  classified finishers (median 11), and the 1960s are lower (median 9), so low places score zero
  through attrition. Shared drives each receive the full rescored points.
- Tier edges are sensitive to the feature choice (see the rescored-points check). Treat the
  tiers as a segmentation of a continuum, not as fixed groups.
- Lap times start in 1996, so pace consistency covers modern drivers only.
- Results describe the dataset. They are not causal claims about driver skill.
"""
    )
    _section("Data and licence")
    st.markdown(
        f"[Formula 1 World Championship 1950-2024 on Kaggle]({DATA_URL}) by Vopani (listed as CC0 "
        "on Kaggle; the underlying Ergast data is CC BY-NC-SA 3.0, so credit is given and this app "
        f"is non-commercial). Code, tests and the long-form write-up: [{REPO_URL}]({REPO_URL})."
    )
