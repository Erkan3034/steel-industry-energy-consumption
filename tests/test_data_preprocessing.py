import numpy as np
import pytest

from src.config import CONFIG
from src.data_preprocessing import DataPreprocessor, cyclical_hour_features


def test_cyclical_encoding_wraps_midnight():
    s1, c1 = cyclical_hour_features(23.75 * 3600)  # 23:45
    s2, c2 = cyclical_hour_features(0.25 * 3600)  # 00:15
    s3, c3 = cyclical_hour_features(12 * 3600)  # 12:00
    near = np.hypot(s1 - s2, c1 - c2)
    far = np.hypot(s1 - s3, c1 - c3)
    assert near < 0.2 < far


def test_cyclical_encoding_on_unit_circle():
    s, c = cyclical_hour_features(np.arange(0, 86400, 900))
    np.testing.assert_allclose(s**2 + c**2, 1.0)


def test_transform_drops_leaky_and_raw_time_columns(raw_df):
    out = DataPreprocessor().transform(raw_df)
    assert "CO2(tCO2)" not in out.columns
    assert "date" not in out.columns
    assert "NSM" not in out.columns
    assert {"hour_sin", "hour_cos"} <= set(out.columns)


def test_split_xy_matches_model_schema(raw_df):
    pre = DataPreprocessor()
    X, y = pre.split_xy(pre.transform(raw_df))
    assert list(X.columns) == CONFIG.features.model_features
    assert y.name == CONFIG.features.target
    assert len(X) == len(y) == len(raw_df)


def test_validate_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        DataPreprocessor().transform(raw_df.drop(columns=["Load_Type"]))


def test_validate_rejects_nulls(raw_df):
    raw_df.loc[0, "Usage_kWh"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        DataPreprocessor().transform(raw_df)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataPreprocessor.load(tmp_path / "nope.csv")


def test_run_split_sizes(raw_csv):
    X_train, X_test, y_train, y_test = DataPreprocessor().run(raw_csv)
    assert len(X_test) == pytest.approx(0.2 * (len(X_train) + len(X_test)), abs=1)
    assert len(X_train) == len(y_train)
