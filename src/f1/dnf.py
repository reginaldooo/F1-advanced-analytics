"""DNF reason classification and non-starter detection."""

import pandas as pd

FINISHED = {"finished", "+1 lap", "+2 laps", "+3 laps", "+4 laps", "+5 laps"}
ACCIDENT_KEYWORDS = ("collision", "accident", "spun", "run off")
MECHANICAL_KEYWORDS = (
    "engine",
    "gearbox",
    "brakes",
    "suspension",
    "hydraulics",
    "clutch",
    "electrical",
)


def classify_dnf(status: object) -> str:
    text = str(status).lower()
    if text in FINISHED:
        return "Finished"
    if any(word in text for word in ACCIDENT_KEYWORDS):
        return "Accident/Driver Error"
    if any(word in text for word in MECHANICAL_KEYWORDS):
        return "Car/Mechanical Failure"
    return "Other"

# Entries that never took the start: always excluded, whatever the grid says.
NON_START_ALWAYS = {
    "did not qualify",
    "did not prequalify",
    "107% rule",
    "safety concerns",
    "underweight",
}
# Ambiguous statuses: a non-start only when the driver had no grid slot.
NON_START_IF_NO_GRID = {"withdrew", "excluded", "injured", "injury"}


def non_starter_mask(df: pd.DataFrame) -> pd.Series:
    """True for result rows that never started (DNQ, DNPQ, 107% rule, grid-less withdrawals)."""
    status = df["status"].str.lower()
    grid = df["grid"] if "grid" in df else pd.Series(0, index=df.index)
    no_grid = grid.fillna(0) == 0
    return status.isin(NON_START_ALWAYS) | (status.isin(NON_START_IF_NO_GRID) & no_grid)
