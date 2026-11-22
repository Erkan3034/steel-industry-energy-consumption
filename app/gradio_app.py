"""Gradio UI for the Steel Industry Energy Consumption model (Hugging Face Spaces, free tier)."""

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.predict import DAYS, LOAD_TYPES, EnergyPredictor, SensorReading  # noqa: E402

predictor = EnergyPredictor(CONFIG.paths.model_file)


def predict(lagging_power, leading_power, lagging_pf, leading_pf, hour, minute, day_of_week, load_type):
    reading = SensorReading(
        lagging_reactive_power=float(lagging_power),
        leading_reactive_power=float(leading_power),
        lagging_power_factor=float(lagging_pf),
        leading_power_factor=float(leading_pf),
        nsm=int(hour) * 3600 + int(minute) * 60,
        day_of_week=day_of_week,
        load_type=load_type,
    )
    kwh = predictor.predict(reading)
    return f"## ⚡ {kwh:.2f} kWh\nPredicted consumption for this 15-minute interval ({reading.week_status})."


with gr.Blocks(title="Steel Energy Predictor") as demo:
    gr.Markdown(
        "# 🏭 Steel Industry Energy Predictor\n"
        "Adjust the sensor readings and time to predict energy consumption (kWh) for a 15-minute interval."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚡ Power Metrics")
            lagging_power = gr.Number(label="Lagging Reactive Power (kVarh)", value=10.0, minimum=0)
            leading_power = gr.Number(label="Leading Reactive Power (kVarh)", value=0.0, minimum=0)
            lagging_pf = gr.Number(label="Lagging Power Factor (%)", value=85.0, minimum=0, maximum=100)
            leading_pf = gr.Number(label="Leading Power Factor (%)", value=100.0, minimum=0, maximum=100)
        with gr.Column():
            gr.Markdown("### ⏱️ Time & Load")
            hour = gr.Slider(0, 23, value=12, step=1, label="Hour of the Day")
            minute = gr.Radio([0, 15, 30, 45], value=0, label="Minute")
            day_of_week = gr.Dropdown(list(DAYS), value="Monday", label="Day of Week")
            load_type = gr.Dropdown(list(LOAD_TYPES), value="Light_Load", label="Load Type")

    button = gr.Button("Predict Energy Consumption", variant="primary")
    output = gr.Markdown()
    inputs = [lagging_power, leading_power, lagging_pf, leading_pf, hour, minute, day_of_week, load_type]
    button.click(predict, inputs=inputs, outputs=output)

    gr.Examples(
        examples=[
            [3.2, 0.0, 73.2, 100.0, 1, 0, "Sunday", "Light_Load"],
            [23.0, 0.0, 80.0, 100.0, 10, 0, "Monday", "Medium_Load"],
            [35.0, 0.0, 85.0, 100.0, 11, 30, "Wednesday", "Maximum_Load"],
        ],
        inputs=inputs,
    )

    with gr.Accordion("ℹ️ About the model", open=False):
        gr.Markdown(
            "Polynomial (degree 2) + Ridge regression pipeline. "
            "Test set: **R² 0.9799 · RMSE 4.79 · MAE 3.07**. `CO2(tCO2)` is excluded to avoid target "
            "leakage; time of day is cyclically encoded (sin/cos). Trained on 2018 data from a single "
            "plant — see the "
            "[Model Card](https://github.com/Erkan3034/steel-industry-energy-consumption/blob/main/MODEL_CARD.md)."
        )

if __name__ == "__main__":
    demo.launch()
