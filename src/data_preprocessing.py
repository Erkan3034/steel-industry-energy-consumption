"""Data loading, cleaning and feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CONFIG, FeatureConfig, TrainConfig

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 24 * 3600


def cyclical_hour_features(nsm: pd.Series | np.ndarray | float) -> tuple:
    """Encode seconds-from-midnight as (sin, cos) on the 24h circle.

    23:45 and 00:15 end up close to each other, unlike the raw NSM value.
    """
    angle = 2 * np.pi * np.asarray(nsm, dtype=float) / SECONDS_PER_DAY
    return np.sin(angle), np.cos(angle)


class DataPreprocessor:
    """Turns the raw Kaggle/UCI CSV into model-ready X, y."""

    def __init__(self, features: FeatureConfig = CONFIG.features) -> None:
        self.features = features

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. See data/README.md for download instructions."
            )
        df = pd.read_csv(path)
        logger.info("Loaded %s rows x %s cols from %s", *df.shape, path)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        required = {
            self.features.target,
            self.features.time_column,
            *self.features.categorical_features,
            *(c for c in self.features.numeric_features if not c.startswith("hour_")),
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df[list(required)].isnull().any().any():
            raise ValueError("Dataset contains missing values in required columns.")

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hour_sin"], df["hour_cos"] = cyclical_hour_features(df[self.features.time_column])
        return df.drop(columns=[self.features.time_column])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, drop leaky columns and engineer features."""
        self.validate(df)
        df = df.drop(columns=[c for c in self.features.drop_columns if c in df.columns])
        return self.add_time_features(df)

    def split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[self.features.model_features], df[self.features.target]

    def run(
        self, path: str | Path, train_cfg: TrainConfig = CONFIG.train
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Full flow: load -> transform -> train/test split."""
        X, y = self.split_xy(self.transform(self.load(path)))
        return train_test_split(
            X, y, test_size=train_cfg.test_size, random_state=train_cfg.random_state
        )
