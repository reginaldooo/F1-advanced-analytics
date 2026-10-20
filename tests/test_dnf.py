import pandas as pd
import pytest

from f1.dnf import classify_dnf, non_starter_mask


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Finished", "Finished"),
        ("+2 Laps", "Finished"),
        ("Collision", "Accident/Driver Error"),
        ("Spun off", "Accident/Driver Error"),
        ("Engine", "Car/Mechanical Failure"),
        ("Gearbox", "Car/Mechanical Failure"),
        ("Disqualified", "Other"),
    ],
)
def test_classify_dnf(status, expected):
    assert classify_dnf(status) == expected


def test_non_starter_mask_flags_qualifying_failures_regardless_of_grid():
    df = pd.DataFrame(
        {
            "status": ["Did not qualify", "Did not prequalify", "107% Rule", "Finished"],
            "grid": [5, 0, 3, 1],
        }
    )
    assert non_starter_mask(df).tolist() == [True, True, True, False]


def test_non_starter_mask_flags_withdrawals_only_without_a_grid_slot():
    df = pd.DataFrame(
        {"status": ["Withdrew", "Withdrew", "Excluded", "Collision"], "grid": [0, 7, 0, 0]}
    )
    assert non_starter_mask(df).tolist() == [True, False, True, False]


def test_non_starter_mask_treats_missing_grid_as_no_slot():
    df = pd.DataFrame({"status": ["Withdrew"], "grid": [None]})
    assert non_starter_mask(df).tolist() == [True]
