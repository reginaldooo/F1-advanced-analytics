import pytest

from f1.dnf import classify_dnf


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
