
import pandas as pd
import pytest

from src.preprocessing import validate_data, split_data


def make_sample_data(rows=1000):
    """Create a small numeric dataset with both target classes."""
    data = {
        "Time": list(range(rows)),
        "V1": [float(i % 10) for i in range(rows)],
        "Amount": [float(i % 100) for i in range(rows)],
        "Class": [1 if i % 20 == 0 else 0 for i in range(rows)],
    }
    return pd.DataFrame(data)


def test_validate_data_accepts_valid_data():
    df = make_sample_data()
    validate_data(df)


def test_validate_data_rejects_missing_values():
    df = make_sample_data()
    df.loc[0, "Amount"] = float("nan")

    with pytest.raises(ValueError, match="Missing values"):
        validate_data(df)


def test_validate_data_rejects_invalid_target():
    df = make_sample_data()
    df["Class"] = 2

    with pytest.raises(ValueError):
        validate_data(df)


def test_split_data_preserves_all_rows():
    df = make_sample_data()
    splits = split_data(df)

    assert len(splits["train"]) + len(splits["validation"]) + len(
        splits["test"]
    ) == len(df)

    assert set(splits.keys()) == {"train", "validation", "test"}


def test_split_data_contains_both_classes():
    df = make_sample_data()
    splits = split_data(df)

    for split_df in splits.values():
        assert set(split_df["Class"].unique()) == {0, 1}
