import pandas as pd
import pytest


@pytest.fixture
def results():
    """Two races: race 1 has 4 starters, race 2 has 3. Driver 1 wins both."""
    return pd.DataFrame(
        {
            "raceId": [1, 1, 1, 1, 2, 2, 2],
            "driverId": [1, 2, 3, 4, 1, 2, 3],
            "statusId": [1, 1, 1, 2, 1, 1, 2],
            "positionOrder": [1, 2, 3, 4, 1, 2, 3],
            "points": ["25", "18", "15", "0", "10", "6", "\\N"],
        }
    )


@pytest.fixture
def drivers():
    return pd.DataFrame(
        {
            "driverId": [1, 2, 3, 4],
            "forename": ["Ann", "Bob", "Cy", "Di"],
            "surname": ["A", "B", "C", "D"],
        }
    )


@pytest.fixture
def status():
    return pd.DataFrame({"statusId": [1, 2], "status": ["Finished", "Engine"]})
