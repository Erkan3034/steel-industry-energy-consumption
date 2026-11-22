"""Smoke tests for the UIs (require the committed model artifact)."""

import pytest

from src.config import CONFIG

pytestmark = pytest.mark.skipif(not CONFIG.paths.model_file.exists(), reason="model artifact not present")


def test_gradio_predict_returns_kwh():
    from app.gradio_app import predict

    kwh, details = predict(23.0, 0.0, 80.0, 100.0, 10, 0, "Monday", "Medium_Load")
    assert kwh >= 0
    assert "Weekday" in details and "10:00 – 10:15" in details


def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(CONFIG.paths.model_file.parents[1] / "app" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert "kWh" in at.success[0].value
