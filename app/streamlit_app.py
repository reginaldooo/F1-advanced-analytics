"""
F1 Driver Comparison: era-neutral profiles, tiers and a cross-era rating.
Reads the precomputed table app/data/drivers.csv (built by `make app-data`).
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "drivers.csv"
REPO_URL = "https://github.com/reginaldooo/F1-advanced-analytics"

TIER_ORDER = ["Dominant Legends", "Podium Contenders", "Midfielders", "Backmarkers"]
TIER_COLOR = {
    "Dominant Legends": "#E10600",
    "Podium Contenders": "#FF8700",
    "Midfielders": "#4C72B0",
    "Backmarkers": "#9AA5B1",
}
# (column, label, higher_is_better) for the percentile radar in the Compare tab.
RADAR_CRITERIA = [
    ("Avg Finish Pct", "Finish percentile", True),
    ("Win Rate", "Win rate", True),
    ("Podium Rate", "Podium rate", True),
    ("Teammate Delta", "Beats teammate", True),
    ("Driver-Error DNF Rate", "Fewer crashes", False),
    ("Pace_StdDev", "Pace consistency", False),
]

st.set_page_config(page_title="F1 Driver Comparison", page_icon="🏎", layout="wide")


@st.cache_data
def load_drivers() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def season_label(row: pd.Series) -> str:
    return f"{int(row['First Season'])}-{int(row['Last Season'])}"


def percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """0-100 percentile of every radar criterion across all drivers (higher = better)."""
    out = pd.DataFrame({"Driver Name": df["Driver Name"]})
    for col, _, higher in RADAR_CRITERIA:
        out[col] = df[col].rank(pct=True, ascending=higher) * 100
    return out


def rating_chart(df: pd.DataFrame, selected: list, top: int = 20) -> go.Figure:
    rated = df.dropna(subset=["Rating"])
    shown = pd.concat(
        [rated.nsmallest(top, "Rating Rank"), rated[rated["Driver Name"].isin(selected)]]
    ).drop_duplicates("Driver Name")
    shown = shown.sort_values("Rating")
    colors = ["#E10600" if n in selected else "#4C72B0" for n in shown["Driver Name"]]
    fig = go.Figure(
        go.Scatter(
            x=shown["Rating"],
            y=shown["Driver Name"],
            mode="markers",
            marker={"color": colors, "size": 10},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": shown["Rating High"] - shown["Rating"],
                "arrayminus": shown["Rating"] - shown["Rating Low"],
                "color": "#888888",
            },
            hovertemplate="%{y}<br>rating %{x:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(420, 22 * len(shown) + 120),
        xaxis_title="Driver skill rating (Elo-like points, car effect removed, 5-95% interval)",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig


drivers = load_drivers()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏎 F1 Driver Comparison")
st.sidebar.caption("Era-neutral profiles, 1950-2024")

min_season, max_season = int(drivers["First Season"].min()), int(drivers["Last Season"].max())
season_range = st.sidebar.slider("Active between", min_season, max_season, (min_season, max_season))
tiers = st.sidebar.multiselect("Tier", TIER_ORDER, default=TIER_ORDER)
min_races = st.sidebar.slider("Minimum races", 21, int(drivers["Total Races"].max()), 21)
search = st.sidebar.text_input("Search driver")

mask = (
    (drivers["Last Season"] >= season_range[0])
    & (drivers["First Season"] <= season_range[1])
    & drivers["Category"].isin(tiers)
    & (drivers["Total Races"] >= min_races)
)
if search:
    mask &= drivers["Driver Name"].str.contains(search, case=False, regex=False)
filtered = drivers[mask]

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(filtered)} of {len(drivers)} drivers shown")
st.sidebar.markdown("**Built by**")
st.sidebar.markdown("Reginaldo Ahnaf")
st.sidebar.markdown(f"[GitHub]({REPO_URL}) · [LinkedIn](https://linkedin.com/in/farrel-reginaldo)")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Driver Explorer", "Cluster View", "Compare", "How to Read This"]
)

# ── Tab 1: Driver Explorer ───────────────────────────────────────────────────
with tab1:
    st.subheader("Driver Explorer")
    st.caption("Drivers with more than 20 race starts. Use the sidebar to filter.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Drivers shown", len(filtered))
    c2.metric("Dominant Legends", int((filtered["Category"] == "Dominant Legends").sum()))
    c3.metric("Median races", int(filtered["Total Races"].median()) if len(filtered) else 0)

    table = filtered.copy()
    table["Seasons"] = [season_label(row) for _, row in filtered.iterrows()]
    columns = [
        "Driver Name", "Nationality", "Seasons", "Total Races", "Category", "Avg Finish Pct",
        "Win Rate", "Podium Rate", "Teammate Delta", "Pace_StdDev", "Rating", "Rating Rank",
    ]
    st.dataframe(
        table[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "Avg Finish Pct": st.column_config.NumberColumn("Finish pct", format="%.2f"),
            "Win Rate": st.column_config.NumberColumn("Win rate", format="percent"),
            "Podium Rate": st.column_config.NumberColumn("Podium rate", format="percent"),
            "Teammate Delta": st.column_config.NumberColumn("Teammate delta", format="%.3f"),
            "Pace_StdDev": st.column_config.NumberColumn("Pace std (1996+)", format="%.4f"),
            "Rating": st.column_config.NumberColumn("Rating", format="%.0f"),
            "Rating Rank": st.column_config.NumberColumn("Rank", format="%d"),
        },
    )

# ── Tab 2: Cluster View ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Cluster View")
    view = st.radio("Projection", ["PCA of scaled features", "Performance matrix"], horizontal=True)
    if filtered.empty:
        st.info("No drivers match the current filters.")
    else:
        x, y = ("PC1", "PC2") if view.startswith("PCA") else ("Avg Finish Pct", "Win Rate")
        fig = px.scatter(
            filtered,
            x=x,
            y=y,
            color="Category",
            color_discrete_map=TIER_COLOR,
            category_orders={"Category": TIER_ORDER},
            hover_name="Driver Name",
            hover_data={"Total Races": True, "Avg Finish Pct": ":.2f", "Win Rate": ":.1%"},
            height=620,
        )
        fig.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "white"}})
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Four tiers from K-Means (k=4) on finish percentile, podium rate and win rate. "
        "k was validated, but HDBSCAN leaves 27% of drivers unassigned: the tiers segment a "
        "continuum, they are not natural groups."
    )

# ── Tab 3: Compare ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Compare two drivers")
    names = sorted(drivers["Driver Name"])
    left_col, right_col = st.columns(2)
    default_left, default_right = names.index("Juan Fangio"), names.index("Max Verstappen")
    left = left_col.selectbox("Driver A", names, index=default_left)
    right = right_col.selectbox("Driver B", names, index=default_right)
    pair = drivers.set_index("Driver Name").loc[[left, right]]

    cards = st.columns(2)
    for card, name in zip(cards, [left, right]):
        row = pair.loc[name]
        card.markdown(f"### {name}")
        card.caption(f"{row['Category']} · {row['Nationality']} · "
                     f"{int(row['First Season'])}-{int(row['Last Season'])} · "
                     f"{int(row['Total Races'])} races")
        if pd.notna(row["Rating"]):
            card.metric(
                "Skill rating (car removed)",
                f"{row['Rating']:.0f}",
                help=f"5-95% interval {row['Rating Low']:.0f} to {row['Rating High']:.0f}; "
                     f"rank {int(row['Rating Rank'])} among rated drivers.",
            )
            card.caption(
                f"Interval {row['Rating Low']:.0f} to {row['Rating High']:.0f}, "
                f"rank {int(row['Rating Rank'])}"
            )
        else:
            card.info("Not rated (too few eligible races).")

    pct = percentiles(drivers).set_index("Driver Name").loc[[left, right]]
    usable = [(c, label) for c, label, _ in RADAR_CRITERIA if pct[c].notna().all()]
    skipped = [label for c, label, _ in RADAR_CRITERIA if not pct[c].notna().all()]
    radar = go.Figure()
    for name, color in zip([left, right], ["#E10600", "#4C72B0"]):
        values = [pct.loc[name, c] for c, _ in usable]
        labels = [label for _, label in usable]
        radar.add_trace(
            go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], name=name,
                            fill="toself", line={"color": color}, opacity=0.6)
        )
    radar.update_layout(polar={"radialaxis": {"range": [0, 100]}}, height=480)
    st.plotly_chart(radar, use_container_width=True)
    note = "Percentile among all rated drivers (higher is better)."
    if skipped:
        note += f" Not shown (no data for one driver): {', '.join(skipped)}. Lap data is 1996+."
    st.caption(note)

    detail = pair[
        ["Avg Finish Pct", "Win Rate", "Podium Rate", "Positions Gained", "Driver-Error DNF Rate",
         "Mechanical DNF Rate", "Teammate Delta", "Pace_StdDev"]
    ].T
    st.dataframe(detail.style.format("{:.3f}", na_rep="n/a"), width="stretch")

    st.markdown("#### Cross-era rating, top 20 plus selected drivers")
    st.plotly_chart(rating_chart(drivers, [left, right]), use_container_width=True)
    st.caption(
        "Robust: Fangio, Stewart and Clark stay in the top 3 for every regularisation strength "
        "tested and Verstappen in the top 6. Drivers who always had dominant cars (Hamilton, "
        "Schumacher, Senna) cannot be cleanly separated from them. Read the list as tiers."
    )

# ── Tab 4: How to Read This ──────────────────────────────────────────────────
with tab4:
    st.subheader("How to read this")
    st.markdown(
        f"""
**Question.** How do you compare F1 drivers across eras without letting points systems and cars
decide?

**Era-neutral performance.** Finish percentile (1 = win, 0 = last starter), win, podium and top-5
rate. Raw points are not used because a win paid 8 points in 1950 and 25 today. Drivers who never
started (DNQ, DNPQ) are excluded.

**Tiers.** K-Means (k=4, robust scaling), named by mean win rate. k was checked with silhouette,
Davies-Bouldin, elbow and seed stability.

**Reliability.** All 139 race statuses are hand-labelled as driver error or mechanical; labels
are the author's and open to review.

**Teammate delta.** Finish percentile minus the other driver(s) in the same car (20+ shared
races). In a dominant car both finish near the top, which compresses it: a low value is not
proof of a weaker driver.

**Pace consistency.** Std of each lap relative to the race median pace (laps above 107% of the
median removed). Lap times exist only from 1996, so older drivers show n/a.

**Cross-era rating.** Bradley-Terry on finishing order with a separate effect for every
constructor-season. Skill is assumed constant over a career, eras are linked only through
overlapping careers (wide 1950s intervals) and drivers with dominant cars are weakly identified.

**Data.** [Formula 1 World Championship 1950-2024 on Kaggle](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
by Vopani (listed as CC0 on Kaggle; the underlying Ergast data is CC BY-NC-SA 3.0, so credit is
given and this app is non-commercial).
Code, tests and methodology: [{REPO_URL}]({REPO_URL}).
"""
    )
