import pytest

from src.config import CONFIG
from src.predict import EnergyPredictor, SensorReading


def make_reading(**overrides) -> SensorReading:
    base = dict(
        lagging_reactive_power=23.0,
        leading_reactive_power=0.0,
        lagging_power_factor=80.0,
        leading_power_factor=100.0,
        nsm=10 * 3600,
        day_of_week="Monday",
        load_type="Medium_Load",
    )
    base.update(overrides)
    return SensorReading(**base)


def test_week_status_derived_from_day():
    assert make_reading(day_of_week="Saturday").week_status == "Weekend"
    assert make_reading(day_of_week="Wednesday").week_status == "Weekday"


def test_to_frame_schema():
    df = make_reading().to_frame()
    assert list(df.columns) == CONFIG.features.model_features
    assert len(df) == 1


def test_invalid_category_rejected():
    with pytest.raises(ValueError):
        make_reading(load_type="Huge_Load").to_frame()


@pytest.mark.skipif(not CONFIG.paths.model_file.exists(), reason="model artifact not present")
def test_committed_model_predicts_reasonable_value():
    kwh = EnergyPredictor().predict(make_reading())
    assert 0 <= kwh < 200
