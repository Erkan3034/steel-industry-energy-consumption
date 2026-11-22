"""Model pipeline definition."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from src.config import CONFIG, FeatureConfig, TrainConfig


def build_pipeline(
    features: FeatureConfig = CONFIG.features, train_cfg: TrainConfig = CONFIG.train
) -> Pipeline:
    """StandardScaler -> PolynomialFeatures(2) on numerics, OneHot on categoricals, Ridge."""
    numeric = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=train_cfg.poly_degree, include_bias=False)),
        ]
    )
    categorical = Pipeline(
        steps=[("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, list(features.numeric_features)),
            ("cat", categorical, list(features.categorical_features)),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=train_cfg.ridge_alpha)),
        ]
    )
