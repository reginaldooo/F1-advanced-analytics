"""Keyword-based DNF reason classifier (to be replaced by a trained model)."""

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
