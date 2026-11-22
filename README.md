# 🏭 Steel Industry Energy Consumption Prediction

[![CI](https://github.com/Erkan3034/steel-industry-energy-consumption/actions/workflows/ci.yml/badge.svg)](https://github.com/Erkan3034/steel-industry-energy-consumption/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow.svg)](https://huggingface.co/spaces/erkanturgut/steel-energy-predictor)

Predicts the energy consumption (`Usage_kWh`, per 15-minute interval) of a steel plant from electrical sensor readings and time features. Focus: **leakage prevention**, **overfitting control** and **cyclical time encoding** in a reproducible scikit-learn pipeline.

## 🏆 Results (hold-out test set, 20%)

| Metric | Score |
| :--- | :--- |
| **R²** | **0.9799** |
| **RMSE** | **4.7856 kWh** |
| **MAE** | **3.0710 kWh** |
| 5-Fold CV RMSE (train) | 4.9379 ± 0.5532 |

CV RMSE ≈ test RMSE → no sign of overfitting.

## 🔬 Methodology

1. **Leakage prevention** — `CO2(tCO2)` is dropped: it is derived from consumption (ρ ≈ 0.99 with the target) and would not be available before the fact.
2. **Cyclical time encoding** — `NSM` (seconds from midnight) → `hour_sin`, `hour_cos`, so 23:45 and 00:15 are adjacent.
3. **Model** — `StandardScaler → PolynomialFeatures(degree=2)` on numerics, `OneHotEncoder(drop="first")` on categoricals, `Ridge(alpha=10)` (alpha from `GridSearchCV`). Everything lives inside one `Pipeline`, so preprocessing is fit only on training folds.

## 📂 Repository Structure

```
.
├── app/
│   └── app.py                  # Streamlit UI
├── data/
│   ├── raw/                    # Steel_industry_data.csv (not committed)
│   └── README.md               # download instructions
├── docs/
│   └── huggingface_deployment.md
├── hf_space/
│   └── README.md               # Hugging Face Space config (front matter)
├── models/
│   └── steel_energy_pipeline_v1.pkl
├── notebooks/
│   └── steel-industry-energy-consumption-prediction.ipynb   # EDA + experiments
├── src/
│   ├── config.py               # paths, feature schema, hyperparameters
│   ├── data_preprocessing.py   # DataPreprocessor: load / validate / features / split
│   ├── model.py                # build_pipeline()
│   ├── evaluate.py             # regression metrics
│   ├── train.py                # Trainer + CLI
│   └── predict.py              # SensorReading + EnergyPredictor (inference)
├── tests/                      # pytest suite
├── .github/workflows/          # CI + Hugging Face sync
├── Dockerfile                  # HF Spaces (Docker SDK) / local container
├── requirements.txt
├── requirements-dev.txt
├── MODEL_CARD.md
├── CONTRIBUTING.md
└── LICENSE
```

## 🚀 Quickstart

```bash
git clone https://github.com/Erkan3034/steel-industry-energy-consumption.git
cd steel-industry-energy-consumption
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

**Run the app** (uses the committed model):

```bash
streamlit run app/app.py
```

**Retrain** (put the CSV in `data/raw/` first — see [data/README.md](data/README.md)):

```bash
python -m src.train
# -> models/steel_energy_pipeline_v1.pkl, models/metrics.json
```

**Test & lint:**

```bash
pytest -v
ruff check .
```

**Docker:**

```bash
docker build -t steel-energy .
docker run -p 8501:8501 steel-energy
```

**Programmatic inference:**

```python
from src.predict import EnergyPredictor, SensorReading

predictor = EnergyPredictor()
kwh = predictor.predict(SensorReading(
    lagging_reactive_power=23.0, leading_reactive_power=0.0,
    lagging_power_factor=80.0, leading_power_factor=100.0,
    nsm=10 * 3600, day_of_week="Monday", load_type="Medium_Load",
))
```

## 🌐 Live Demo

**[huggingface.co/spaces/erkanturgut/steel-energy-predictor](https://huggingface.co/spaces/erkanturgut/steel-energy-predictor)** — deployment steps in [docs/huggingface_deployment.md](docs/huggingface_deployment.md).

## 📊 Dataset

[Steel Industry Energy Consumption](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption) (DAEWOO Steel Co., South Korea, 2018; also on [UCI](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)). 35,040 rows, 15-minute resolution.

## ⚠️ Limitations

Single plant, single year, random (not time-based) split. See [MODEL_CARD.md](MODEL_CARD.md).

## 👨‍💻 Author

**Erkan Turgut** — [GitHub](https://github.com/Erkan3034) · [Kaggle](https://www.kaggle.com/erkanturgut) · [LinkedIn](https://www.linkedin.com/in/erkanturgut1205/)

## 📄 License

[MIT](LICENSE)
