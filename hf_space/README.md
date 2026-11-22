---
title: Steel Energy Predictor
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app/gradio_app.py
pinned: false
license: mit
short_description: Predict steel plant energy usage (kWh) from sensor data
tags:
  - regression
  - scikit-learn
  - energy
  - tabular-regression
---

# Steel Industry Energy Consumption Predictor

Polynomial (degree 2) + Ridge regression pipeline predicting `Usage_kWh` per 15-minute interval.

**Test set:** R² 0.9799 · RMSE 4.79 kWh · MAE 3.07 kWh

- Kaggle notebook: [kaggle.com/code/erkanturgut/steel-industry-energy-consumption](https://www.kaggle.com/code/erkanturgut/steel-industry-energy-consumption/)
- Source code, training pipeline and model card: [github.com/Erkan3034/steel-industry-energy-consumption](https://github.com/Erkan3034/steel-industry-energy-consumption)
