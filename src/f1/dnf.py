"""DNF reason classification (hand-labelled status table + trained fallback) and non-starters."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline

LABELS_PATH = Path(__file__).with_name("status_labels.csv")
VALID_LABELS = (
    "Finished",
    "Accident/Driver Error",
    "Car/Mechanical Failure",
    "Did Not Start",
    "Other",
)

# Keyword baseline (kept for comparison and as the last-resort fallback).
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


def classify_dnf_keywords(status: object) -> str:
    text = str(status).lower()
    if text in FINISHED:
        return "Finished"
    if any(word in text for word in ACCIDENT_KEYWORDS):
        return "Accident/Driver Error"
    if any(word in text for word in MECHANICAL_KEYWORDS):
        return "Car/Mechanical Failure"
    return "Other"


@lru_cache(maxsize=1)
def _label_table() -> pd.DataFrame:
    return pd.read_csv(LABELS_PATH)


def load_status_labels() -> dict[str, str]:
    """Hand-labelled `status -> reason` for every status in the Kaggle `status.csv`."""
    table = _label_table()
    return dict(zip(table["status"], table["label"]))


def train_status_classifier() -> Pipeline:
    """TF-IDF character n-grams + logistic regression, trained on the label table."""
    table = _label_table()
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)),
            ("clf", LogisticRegression(C=10, max_iter=1000, class_weight="balanced")),
        ]
    )
    return model.fit(table["status"], table["label"])


def predict_reasons(model: Pipeline, statuses: list[str]) -> list[str]:
    return list(model.predict(pd.Series(statuses)))


def classify_dnf(status: object, model: Pipeline | None = None) -> str:
    """Label-table lookup; unseen statuses use `model` if given, else the keyword rules."""
    labels = load_status_labels()
    if status in labels:
        return labels[status]
    if model is not None:
        return predict_reasons(model, [str(status)])[0]
    return classify_dnf_keywords(status)


def _collapse(label: str) -> str:
    """Non-starters are removed upstream, so score 'Did Not Start' as 'Other'."""
    return "Other" if label == "Did Not Start" else label


def evaluate_classifiers(counts: pd.Series | None = None) -> pd.DataFrame:
    """Keyword rules vs trained model (leave-one-out) against the hand labels.

    `counts` maps status -> number of result rows; it weights `weighted_accuracy` by how often
    each status occurs. Scored on four classes ('Did Not Start' counted as 'Other').
    """
    table = _label_table()
    truth = table["label"].map(_collapse)
    weights = (
        table["status"].map(counts).fillna(0)
        if counts is not None
        else pd.Series(1.0, index=table.index)
    )
    keyword = table["status"].map(classify_dnf_keywords)
    loo = cross_val_predict(
        train_status_classifier(), table["status"], table["label"], cv=LeaveOneOut()
    )
    trained = pd.Series(loo, index=table.index).map(_collapse)
    rows = []
    for method, pred in (("Keywords", keyword), ("Trained (leave-one-out)", trained)):
        correct = pred == truth
        rows.append(
            {
                "method": method,
                "accuracy": correct.mean(),
                "macro_f1": f1_score(truth, pred, average="macro"),
                "weighted_accuracy": (correct * weights).sum() / weights.sum()
                if weights.sum()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


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
