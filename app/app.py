"""Streamlit UI for the Steel Industry Energy Consumption model."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.predict import DAYS, LOAD_TYPES, EnergyPredictor, SensorReading  # noqa: E402

st.set_page_config(page_title="Steel Energy Predictor", page_icon="🏭", layout="centered")


@st.cache_resource
def load_predictor() -> EnergyPredictor:
    return EnergyPredictor(CONFIG.paths.model_file)


predictor = load_predictor()

st.title("🏭 Steel Industry Energy Predictor")
st.write("Adjust the sensor readings and time to predict energy consumption (kWh) for a 15-minute interval.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚡ Power Metrics")
    lagging_power = st.number_input(
        "Lagging Reactive Power (kVarh)", min_value=0.0, max_value=100.0, value=10.0, step=1.0
    )
    leading_power = st.number_input(
        "Leading Reactive Power (kVarh)", min_value=0.0, max_value=30.0, value=0.0, step=1.0
    )
    lagging_pf = st.number_input(
        "Lagging Power Factor (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1
    )
    leading_pf = st.number_input(
        "Leading Power Factor (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1
    )

with col2:
    st.subheader("⏱️ Time & Load")
    hour = st.slider("Hour of the Day", 0, 23, 12)
    minute = st.select_slider("Minute", options=[0, 15, 30, 45], value=0)
    day_of_week = st.selectbox("Day of Week", DAYS)
    load_type = st.selectbox("Load Type", LOAD_TYPES)

reading = SensorReading(
    lagging_reactive_power=lagging_power,
    leading_reactive_power=leading_power,
    lagging_power_factor=lagging_pf,
    leading_power_factor=leading_pf,
    nsm=hour * 3600 + minute * 60,
    day_of_week=day_of_week,
    load_type=load_type,
)
st.caption(f"Week status (derived): **{reading.week_status}**")

if st.button("Predict Energy Consumption", type="primary"):
    prediction = predictor.predict(reading)
    st.success(f"### Predicted Energy Consumption: **{prediction:.2f} kWh**")

with st.expander("ℹ️ About the model"):
    st.markdown(
        "Polynomial (degree 2) + Ridge regression pipeline. Test set: **R² 0.9799 · RMSE 4.79 · MAE 3.07**. "
        "`CO2(tCO2)` is excluded to avoid target leakage; time of day is cyclically encoded (sin/cos). "
        "Trained on 2018 data from a single plant — see the Model Card for limitations."
    )
