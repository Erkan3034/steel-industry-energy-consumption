"""Inference helpers shared by the Streamlit app and any other consumer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import CONFIG
from src.data_preprocessing import cyclical_hour_features

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
LOAD_TYPES = ("Light_Load", "Medium_Load", "Maximum_Load")


@dataclass
class SensorReading:
    lagging_reactive_power: float  # kVarh
    leading_reactive_power: float  # kVarh
    lagging_power_factor: float  # %
    leading_power_factor: float  # %
    nsm: int  # seconds from midnight
    day_of_week: str
    load_type: str

    @property
    def week_status(self) -> str:
        return "Weekend" if self.day_of_week in ("Saturday", "Sunday") else "Weekday"

    def to_frame(self) -> pd.DataFrame:
        if self.day_of_week not in DAYS:
            raise ValueError(f"Invalid day_of_week: {self.day_of_week}")
        if self.load_type not in LOAD_TYPES:
            raise ValueError(f"Invalid load_type: {self.load_type}")
        hour_sin, hour_cos = cyclical_hour_features(self.nsm)
        row = {
            "Lagging_Current_Reactive.Power_kVarh": self.lagging_reactive_power,
            "Leading_Current_Reactive_Power_kVarh": self.leading_reactive_power,
            "Lagging_Current_Power_Factor": self.lagging_power_factor,
            "Leading_Current_Power_Factor": self.leading_power_factor,
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
            "WeekStatus": self.week_status,
            "Day_of_week": self.day_of_week,
            "Load_Type": self.load_type,
        }
        return pd.DataFrame([row])[CONFIG.features.model_features]


class EnergyPredictor:
    def __init__(self, model_path: str | Path = CONFIG.paths.model_file) -> None:
        self.pipeline: Pipeline = joblib.load(model_path)

    def predict(self, reading: SensorReading) -> float:
        # Energy usage cannot be negative; clip the linear model's output.
        return max(0.0, float(self.pipeline.predict(reading.to_frame())[0]))
