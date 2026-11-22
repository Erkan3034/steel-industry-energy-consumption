"""Gradio UI for the Steel Industry Energy Consumption model (Hugging Face Space)."""

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG  # noqa: E402
from src.predict import DAYS, LOAD_TYPES, EnergyPredictor, SensorReading  # noqa: E402

try:  # ZeroGPU Spaces require at least one @spaces.GPU function; no-op elsewhere.
    import spaces

    gpu = spaces.GPU(duration=10)
except ImportError:

    def gpu(fn):
        return fn


GITHUB_URL = "https://github.com/Erkan3034/steel-industry-energy-consumption"
KAGGLE_URL = "https://www.kaggle.com/code/erkanturgut/steel-industry-energy-consumption/"
MODEL_CARD_URL = f"{GITHUB_URL}/blob/main/MODEL_CARD.md"
DATASET_URL = "https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption"

predictor = EnergyPredictor(CONFIG.paths.model_file)


@gpu
def _zerogpu_startup_check():
    """Never called. ZeroGPU refuses to start without one @spaces.GPU function, but this
    model runs on CPU, so predictions stay outside it and use no visitor GPU quota."""


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
    start = f"{int(hour):02d}:{int(minute):02d}"
    end_minutes = int(hour) * 60 + int(minute) + 15
    end = f"{(end_minutes // 60) % 24:02d}:{end_minutes % 60:02d}"
    details = (
        "| | |\n|:--|:--|\n"
        f"| Interval | {day_of_week} {start} – {end} ({reading.week_status}) |\n"
        f"| Load type | {load_type.replace('_', ' ')} |\n"
        f"| Equivalent average load | {kwh * 4:.1f} kW |\n"
    )
    return round(kwh, 2), details


CSS = """
.header {border-bottom: 1px solid var(--border-color-primary); padding-bottom: 12px; margin-bottom: 4px;}
.header h1 {margin-bottom: 4px;}
.links a {margin-right: 18px; font-weight: 500;}
.section-title {padding: 10px 12px 0 12px;}
.result-number input {font-size: 2rem !important; font-weight: 600;}
"""

HEADER = f"""
<div class="header">
<h1>Steel Industry Energy Consumption Predictor</h1>
<p>Estimates the electrical energy consumed by a steel plant in a 15-minute interval
from reactive power, power factor, time of day and load profile.</p>
<p class="links">
<a href="{KAGGLE_URL}" target="_blank">Kaggle notebook</a>
<a href="{GITHUB_URL}" target="_blank">Source code</a>
<a href="{MODEL_CARD_URL}" target="_blank">Model card</a>
<a href="{DATASET_URL}" target="_blank">Dataset (UCI)</a>
</p>
</div>
"""

MODEL_INFO = f"""
### Performance

Evaluated on a 20% hold-out test set (7,008 intervals).

| Metric | Value |
|:--|:--|
| R² | 0.9799 |
| RMSE | 4.79 kWh |
| MAE | 3.07 kWh |
| 5-fold CV RMSE (training set) | 4.94 ± 0.55 kWh |

Cross-validation and test RMSE are close, indicating no meaningful overfitting.

### Pipeline

| Step | Details |
|:--|:--|
| Numeric features | `StandardScaler` → `PolynomialFeatures(degree=2)` |
| Categorical features | `OneHotEncoder(drop="first")` |
| Regressor | `Ridge(alpha=10)`, alpha selected with `GridSearchCV` |
| Framework | scikit-learn 1.6.1, single `Pipeline` serialized with joblib |

### Feature engineering

- **Leakage prevention:** `CO2(tCO2)` is excluded. It is derived from consumption
  (correlation ≈ 0.99 with the target) and would not be known before the measurement.
- **Cyclical time encoding:** seconds from midnight (`NSM`) are mapped to `sin` / `cos`
  of the 24-hour cycle, so 23:45 and 00:15 are treated as neighbours.
- **Week status** is derived from the selected day.

### Data

DAEWOO Steel Co. (Gwangyang, South Korea), year 2018, 35,040 records at 15-minute resolution.

### Limitations

- Inputs are measured in the same interval as the target: this is an estimation model,
  not an ahead-of-time forecaster.
- Random train/test split on time-series data gives an optimistic estimate; a chronological
  split would be stricter.
- Single plant, single year. Weather, holidays and maintenance schedules are not included.

Full details: [Model card]({MODEL_CARD_URL}) · [Kaggle notebook]({KAGGLE_URL}) · [GitHub]({GITHUB_URL})
"""

with gr.Blocks(title="Steel Energy Consumption Predictor") as demo:
    gr.HTML(HEADER)

    with gr.Tab("Prediction"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("#### Electrical measurements", elem_classes="section-title")
                    with gr.Row():
                        lagging_power = gr.Number(
                            label="Lagging reactive power (kVarh)", value=23.0, minimum=0,
                            info="Inductive reactive energy in the interval",
                        )
                        leading_power = gr.Number(
                            label="Leading reactive power (kVarh)", value=0.0, minimum=0,
                            info="Capacitive reactive energy in the interval",
                        )
                    with gr.Row():
                        lagging_pf = gr.Number(
                            label="Lagging power factor (%)", value=80.0, minimum=0, maximum=100,
                        )
                        leading_pf = gr.Number(
                            label="Leading power factor (%)", value=100.0, minimum=0, maximum=100,
                        )
                with gr.Group():
                    gr.Markdown("#### Time and operating profile", elem_classes="section-title")
                    with gr.Row():
                        hour = gr.Slider(0, 23, value=10, step=1, label="Hour")
                        minute = gr.Radio([0, 15, 30, 45], value=0, label="Minute")
                    with gr.Row():
                        day_of_week = gr.Dropdown(list(DAYS), value="Monday", label="Day of week")
                        load_type = gr.Dropdown(list(LOAD_TYPES), value="Medium_Load", label="Load type")
                button = gr.Button("Predict", variant="primary")

            with gr.Column(scale=2):
                result = gr.Number(
                    label="Predicted consumption (kWh per 15 min)", value=None, interactive=False,
                    elem_classes="result-number",
                )
                details = gr.Markdown()
                gr.Markdown(
                    "Typical test-set error is about ±3 kWh (MAE). Predictions are clipped at 0 kWh.",
                )

        inputs = [lagging_power, leading_power, lagging_pf, leading_pf, hour, minute, day_of_week, load_type]
        button.click(predict, inputs=inputs, outputs=[result, details])

        gr.Examples(
            label="Example operating points",
            examples=[
                [3.2, 0.0, 73.2, 100.0, 1, 0, "Sunday", "Light_Load"],
                [23.0, 0.0, 80.0, 100.0, 10, 0, "Monday", "Medium_Load"],
                [35.0, 0.0, 85.0, 100.0, 11, 30, "Wednesday", "Maximum_Load"],
            ],
            inputs=inputs,
        )

    with gr.Tab("Model"):
        gr.Markdown(MODEL_INFO)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Base(primary_hue="slate", neutral_hue="slate"))
