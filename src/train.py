"""Training entry point.

Usage:
    python -m src.train
    python -m src.train --data data/raw/Steel_industry_data.csv --output models/steel_energy_pipeline_v1.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CONFIG, Config
from src.data_preprocessing import DataPreprocessor
from src.evaluate import regression_metrics
from src.model import build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    pipeline: Pipeline
    cv_rmse_mean: float
    cv_rmse_std: float
    test_metrics: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "cv_rmse_mean": self.cv_rmse_mean,
            "cv_rmse_std": self.cv_rmse_std,
            "test": self.test_metrics,
            "sklearn_version": sklearn.__version__,
        }


class Trainer:
    """Orchestrates preprocessing, cross-validation, fitting, evaluation and saving."""

    def __init__(self, config: Config = CONFIG) -> None:
        self.config = config
        self.preprocessor = DataPreprocessor(config.features)

    def train(self, data_path: str | Path) -> TrainingResult:
        X_train, X_test, y_train, y_test = self.preprocessor.run(data_path, self.config.train)
        pipeline = build_pipeline(self.config.features, self.config.train)

        cv_scores = -cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=self.config.train.cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        logger.info("CV RMSE: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

        pipeline.fit(X_train, y_train)
        metrics = regression_metrics(y_test, pipeline.predict(X_test))
        logger.info("Test metrics: %s", metrics)

        return TrainingResult(pipeline, float(cv_scores.mean()), float(cv_scores.std()), metrics)

    @staticmethod
    def save(result: TrainingResult, model_path: Path, metrics_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.pipeline, model_path)
        metrics_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Saved model -> %s, metrics -> %s", model_path, metrics_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the steel energy consumption model.")
    p.add_argument("--data", type=Path, default=CONFIG.paths.raw_data)
    p.add_argument("--output", type=Path, default=CONFIG.paths.model_file)
    p.add_argument("--metrics", type=Path, default=CONFIG.paths.metrics_file)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    trainer = Trainer()
    result = trainer.train(args.data)
    trainer.save(result, args.output, args.metrics)


if __name__ == "__main__":
    main()
