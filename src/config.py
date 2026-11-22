"""Central configuration: paths, feature schema and hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    raw_data: Path = PROJECT_ROOT / "data" / "raw" / "Steel_industry_data.csv"
    model_dir: Path = PROJECT_ROOT / "models"
    model_file: Path = PROJECT_ROOT / "models" / "steel_energy_pipeline_v1.pkl"
    metrics_file: Path = PROJECT_ROOT / "models" / "metrics.json"


@dataclass(frozen=True)
class FeatureConfig:
    target: str = "Usage_kWh"
    # CO2 is ~0.99 correlated with the target (computed from it) -> leakage.
    # `date` is replaced by the cyclical NSM encoding.
    drop_columns: tuple[str, ...] = ("CO2(tCO2)", "date")
    time_column: str = "NSM"
    numeric_features: tuple[str, ...] = (
        "Lagging_Current_Reactive.Power_kVarh",
        "Leading_Current_Reactive_Power_kVarh",
        "Lagging_Current_Power_Factor",
        "Leading_Current_Power_Factor",
        "hour_sin",
        "hour_cos",
    )
    categorical_features: tuple[str, ...] = ("WeekStatus", "Day_of_week", "Load_Type")

    @property
    def model_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    poly_degree: int = 2
    ridge_alpha: float = 10.0  # selected with GridSearchCV in the notebook


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
