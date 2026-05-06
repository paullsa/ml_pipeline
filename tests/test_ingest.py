import pandas as pd
import numpy as np
import pytest

from ml_pipeline.ingest import prepare_data, split_data


def _make_df():
    return pd.DataFrame({
        "feat_a": [1, 2, 3, 4, 5],
        "feat_b": [5, 4, 3, 2, 1],
        "id": [10, 11, 12, 13, 14],
        "label": [0, 1, 0, 1, 0],
    })


def test_prepare_data_drops_target_and_excludes():
    df = _make_df()
    X, y = prepare_data(df, target_column="label", exclude_columns=["id"])
    assert "label" not in X.columns
    assert "id" not in X.columns
    assert list(X.columns) == ["feat_a", "feat_b"]
    assert len(y) == 5


def test_split_data_sizes():
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(100))
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=0)
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_split_data_reproducible():
    X = pd.DataFrame({"a": range(50)})
    y = pd.Series(range(50))
    _, X_test_1, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    _, X_test_2, _, _ = split_data(X, y, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
