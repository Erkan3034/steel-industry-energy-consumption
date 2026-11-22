import json

import joblib

from src.train import Trainer


def test_trainer_end_to_end(raw_csv, tmp_path):
    trainer = Trainer()
    result = trainer.train(raw_csv)

    assert result.test_metrics["r2"] > 0.9
    assert result.cv_rmse_mean > 0

    model_path, metrics_path = tmp_path / "m.pkl", tmp_path / "metrics.json"
    trainer.save(result, model_path, metrics_path)

    reloaded = joblib.load(model_path)
    assert hasattr(reloaded, "predict")
    saved = json.loads(metrics_path.read_text())
    assert set(saved["test"]) == {"r2", "rmse", "mae"}
