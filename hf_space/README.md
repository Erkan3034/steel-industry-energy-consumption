---
title: Steel Energy Predictor
emoji: 🏭
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Predict steel plant energy usage (kWh) from sensor data
tags:
  - regression
  - scikit-learn
  - energy
  - streamlit
---

# 🏭 Steel Industry Energy Predictor

Polynomial (degree 2) + Ridge regression pipeline predicting `Usage_kWh` per 15-minute interval.

**Test set:** R² 0.9799 · RMSE 4.79 kWh · MAE 3.07 kWh

Source code, training pipeline and model card: [github.com/Erkan3034/steel-industry-energy-consumption](https://github.com/Erkan3034/steel-industry-energy-consumption)
