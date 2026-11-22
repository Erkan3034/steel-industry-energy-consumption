# Model Card: Steel Industry Energy Predictor

## 1. Model Details

| | |
| :--- | :--- |
| **Model type** | Ridge Regression (α = 10) on StandardScaler + PolynomialFeatures(degree 2) numerics and one-hot categoricals |
| **Framework** | scikit-learn 1.6.1 (single `Pipeline`, serialized with joblib) |
| **Artifact** | `models/steel_energy_pipeline_v1.pkl` |
| **Version** | 1.0.0 |
| **Training date** | 2026-11-23 |
| **Author** | Erkan Turgut |
| **License** | MIT |
| **Notebook** | [Kaggle: steel-industry-energy-consumption](https://www.kaggle.com/code/erkanturgut/steel-industry-energy-consumption/) |

## 2. Intended Use

- **Task:** regression of `Usage_kWh` for a 15-minute interval.
- **Inputs:** lagging/leading reactive power (kVarh), lagging/leading power factor (%), time of day (NSM → sin/cos), day of week, week status, load type.
- **Use cases:** energy monitoring, anomaly flagging (actual vs. expected consumption), what-if analysis, decision support for load planning.
- **Out of scope:** other plants without retraining; safety-critical control; billing or financial settlement.

## 3. Training Data

- **Source:** Steel Industry Energy Consumption dataset (DAEWOO Steel Co., Gwangyang, South Korea), via Kaggle / UCI.
- **Period:** 2018, 15-minute resolution, 35,040 rows, no missing values.
- **Split:** random 80/20 train/test (`random_state=42`), 5-fold CV on train.
- **Excluded features:** `CO2(tCO2)` (derived from consumption → target leakage), `date` (replaced by cyclical NSM features).

## 4. Evaluation

| Metric | Test set |
| :--- | :--- |
| R² | 0.9799 |
| RMSE | 4.7856 kWh |
| MAE | 3.0710 kWh |
| 5-Fold CV RMSE (train) | 4.9379 ± 0.5532 |

Target distribution is right-skewed (skewness ≈ 1.20); errors are larger in the high-consumption regime.

## 5. Limitations & Factors

- **Random split on time series:** adjacent 15-minute intervals are strongly autocorrelated, so a random split is optimistic. A chronological split (e.g. train Jan–Oct, test Nov–Dec) is the more realistic estimate of forecasting performance.
- **Concurrent inputs:** reactive power and power factor are measured in the *same* interval as the target. The model estimates consumption from concurrent electrical readings; it is **not** an ahead-of-time forecaster.
- **Temporal scope:** 2018 only. Changes in machinery, capacity, shifts or production mix cause drift.
- **Missing external variables:** weather, holidays, maintenance schedules and production orders are not included.
- **Extrapolation:** polynomial terms can behave poorly outside the training range; predictions are clipped at 0 kWh in the app.
- **Single site:** not validated on any other plant.

## 6. Ethical Considerations

No personal data is used — only industrial sensor data. Predictions are decision support and should not be the sole basis for safety-critical or financial decisions.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python -m src.train --data data/raw/Steel_industry_data.csv
```

The pickle must be loaded with the same scikit-learn version it was saved with (pinned in `requirements.txt`).
