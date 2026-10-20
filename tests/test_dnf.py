import pandas as pd
import pytest

from f1.dnf import (
    VALID_LABELS,
    classify_dnf,
    classify_dnf_keywords,
    evaluate_classifiers,
    load_status_labels,
    non_starter_mask,
    predict_reasons,
    train_status_classifier,
)


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


def test_label_table_is_complete_and_valid():
    labels = load_status_labels()
    assert set(labels.values()) <= set(VALID_LABELS)
    assert len(labels) >= 139


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Transmission", "Car/Mechanical Failure"),
        ("Turbo", "Car/Mechanical Failure"),
        ("Fuel system", "Car/Mechanical Failure"),
        ("Puncture", "Car/Mechanical Failure"),
        ("+9 Laps", "Finished"),
        ("+46 Laps", "Finished"),
        ("Did not qualify", "Did Not Start"),
        ("Collision damage", "Accident/Driver Error"),
        ("Retired", "Other"),
    ],
)
def test_classify_dnf_uses_label_table_for_known_statuses(status, expected):
    assert classify_dnf(status) == expected


def test_classify_dnf_falls_back_to_keywords_for_unseen_status():
    assert classify_dnf("Brand new engine problem") == "Car/Mechanical Failure"
    assert classify_dnf("Totally unknown") == "Other"


def test_keyword_baseline_misses_statuses_the_table_covers():
    assert classify_dnf_keywords("Transmission") == "Other"
    assert classify_dnf_keywords("+9 Laps") == "Other"


def test_trained_classifier_predicts_valid_labels_for_unseen_statuses():
    model = train_status_classifier()
    preds = predict_reasons(model, ["+31 Laps", "Wheel hub", "Gearbox oil"])
    assert set(preds) <= set(VALID_LABELS)
    assert preds[0] == "Finished"


def test_evaluate_classifiers_reports_both_methods_and_trained_beats_keywords():
    counts = pd.Series({"Transmission": 300, "Engine": 2000, "Finished": 7000, "+9 Laps": 40})
    out = evaluate_classifiers(counts).set_index("method")
    assert {"Keywords", "Trained (leave-one-out)"} == set(out.index)
    assert {"accuracy", "macro_f1", "weighted_accuracy"} <= set(out.columns)
    assert out.loc["Trained (leave-one-out)", "accuracy"] > out.loc["Keywords", "accuracy"]
