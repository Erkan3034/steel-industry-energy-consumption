---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - sklearn
  - tabular-regression
  - ridge-regression
  - energy
  - industrial
model-index:
  - name: steel-energy-consumption-ridge
    results:
      - task:
          type: tabular-regression
          name: Tabular Regression
        dataset:
          type: uci/steel-industry-energy-consumption
          name: Steel Industry Energy Consumption (UCI 851)
        metrics:
          - type: r2
            value: 0.9799
            name: R² (test)
          - type: rmse
            value: 4.7856
            name: RMSE kWh (test)
          - type: mae
            value: 3.0710
            name: MAE kWh (test)
---

# Steel Industry Energy Consumption — Polynomial Ridge Regression

scikit-learn pipeline that estimates the energy consumption (`Usage_kWh`) of a steel plant for a 15-minute interval from reactive power, power factor, time of day and load profile.

- Live demo: [huggingface.co/spaces/erkanturgut/steel-energy-predictor](https://huggingface.co/spaces/erkanturgut/steel-energy-predictor)
- Kaggle notebook: [kaggle.com/code/erkanturgut/steel-industry-energy-consumption](https://www.kaggle.com/code/erkanturgut/steel-industry-energy-consumption/)
- Source code and training pipeline: [github.com/Erkan3034/steel-industry-energy-consumption](https://github.com/Erkan3034/steel-industry-energy-consumption)

## Performance

20% hold-out test set (7,008 intervals), random split with `random_state=42`.

| Metric | Value |
| :--- | :--- |
| R² | 0.9799 |
| RMSE | 4.7856 kWh |
| MAE | 3.0710 kWh |
| 5-fold CV RMSE (training set) | 4.9379 ± 0.5532 kWh |

## Model

| Step | Details |
| :--- | :--- |
| Numeric features | `StandardScaler` → `PolynomialFeatures(degree=2)` |
| Categorical features | `OneHotEncoder(drop="first", handle_unknown="ignore")` |
| Regressor | `Ridge(alpha=10)`, alpha selected with `GridSearchCV` |
| Artifact | `steel_energy_pipeline_v1.pkl` (joblib) |
| Framework | scikit-learn **1.6.1** — load with the same version |

### Input schema

| Column | Type | Notes |
| :--- | :--- | :--- |
| `Lagging_Current_Reactive.Power_kVarh` | float | |
| `Leading_Current_Reactive_Power_kVarh` | float | |
| `Lagging_Current_Power_Factor` | float | percent |
| `Leading_Current_Power_Factor` | float | percent |
| `hour_sin` | float | `sin(2π · NSM / 86400)` |
| `hour_cos` | float | `cos(2π · NSM / 86400)` |
| `WeekStatus` | str | `Weekday` / `Weekend` |
| `Day_of_week` | str | `Monday` … `Sunday` |
| `Load_Type` | str | `Light_Load` / `Medium_Load` / `Maximum_Load` |

`NSM` is seconds from midnight. `CO2(tCO2)` is intentionally not an input: it is derived from consumption and would leak the target.

## Usage

```bash
pip install scikit-learn==1.6.1 pandas numpy joblib huggingface_hub
```

```python
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("erkanturgut/steel-energy-consumption-ridge", "steel_energy_pipeline_v1.pkl")
model = joblib.load(path)

nsm = 10 * 3600  # 10:00
X = pd.DataFrame([{
    "Lagging_Current_Reactive.Power_kVarh": 23.0,
    "Leading_Current_Reactive_Power_kVarh": 0.0,
    "Lagging_Current_Power_Factor": 80.0,
    "Leading_Current_Power_Factor": 100.0,
    "hour_sin": np.sin(2 * np.pi * nsm / 86400),
    "hour_cos": np.cos(2 * np.pi * nsm / 86400),
    "WeekStatus": "Weekday",
    "Day_of_week": "Monday",
    "Load_Type": "Medium_Load",
}])
print(model.predict(X))  # kWh for the 15-minute interval
```

## Training data

Steel Industry Energy Consumption dataset (DAEWOO Steel Co., Gwangyang, South Korea): 2018, 15-minute resolution, 35,040 rows, no missing values. Available on [UCI](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption) and [Kaggle](https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption).

## Limitations

- Inputs are measured in the same interval as the target: this is an estimation model, not an ahead-of-time forecaster.
- A random split on time-series data gives an optimistic estimate; a chronological split would be stricter.
- Single plant, single year. Weather, holidays and maintenance schedules are not included.
- Polynomial terms can extrapolate poorly outside the training range.

## Security note

The artifact is a pickle (joblib). Only load pickles from sources you trust.

## Author

Erkan Turgut — [GitHub](https://github.com/Erkan3034) · [Kaggle](https://www.kaggle.com/erkanturgut) · [LinkedIn](https://www.linkedin.com/in/erkanturgut1205/)
