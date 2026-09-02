# Results

This directory will contain the key outputs used to summarise model performance and interpretation.

## Planned Figures

### Classification

- `classification_confusion_matrix.png`
- `classification_precision_recall_curve.png`
- `classification_shap_summary.png`

The selected cost-sensitive Gradient Boosting classifier achieved:

- Balanced Accuracy: **0.830**
- Recall: **0.704**
- F1-score: **0.655**
- Correctly detected **19 of 27** high-pollution days

---

### Regression

- `regression_predictions.png`
- `regression_model_comparison.png`
- `regression_shap_summary.png`

The selected tuned Gradient Boosting regressor achieved:

- R²: **0.664**
- RMSE: **2.598 µg/m³**
- MAE: **1.923 µg/m³**

---

### ARIMAX Forecasting

- `arimax_walk_forward_forecast.png`

The ARIMAX(1,0,0) one-day-ahead forecasting model achieved:

- R²: **0.518**
- RMSE: **3.107 µg/m³**
- MAE: **2.234 µg/m³**
- 95% prediction interval coverage: **98.01%**

---

## Purpose

Only the most informative plots will be included here.

The aim is to make the main findings easy to understand without requiring readers to run every notebook.
