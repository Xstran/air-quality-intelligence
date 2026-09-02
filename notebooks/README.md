# Notebooks

The original research workflow is being reorganised into smaller, focused notebooks so that each stage of the analysis can be understood and reproduced independently.

## Planned Notebooks

### `01_data_preparation_eda.ipynb`

Data preparation and exploratory analysis:

- Load and validate hourly PM2.5 data
- Apply daily completeness checks
- Aggregate hourly observations to daily PM2.5
- Examine missing-data patterns
- Explore PM2.5 distributions and seasonal patterns
- Retrieve and inspect meteorological data
- Analyse relationships between weather and PM2.5
- Create wind-direction sine and cosine features
- Construct PM2.5 lag features
- Enforce calendar continuity

---

### `02_classification.ipynb`

High-pollution day classification:

- Define high-pollution days using the 15 µg/m³ threshold
- Compare meteorology-only and meteorology + lag feature sets
- Random Forest and Gradient Boosting baselines
- Expanding-window `TimeSeriesSplit`
- Hyperparameter optimisation
- Class-weighted and cost-sensitive learning
- SMOTE comparison
- Final 2023 evaluation
- Confusion matrix and precision-recall analysis
- SHAP interpretation

---

### `03_regression.ipynb`

Daily PM2.5 regression:

- Random Forest and Gradient Boosting baselines
- Meteorology-only vs meteorology + PM2.5 lag features
- Expanding-window `TimeSeriesSplit`
- Hyperparameter optimisation
- Final 2023 holdout evaluation
- RMSE, MAE and R² comparison
- Observed vs predicted analysis
- SHAP interpretation

---

### `04_arimax_forecasting.ipynb`

One-day-ahead PM2.5 forecasting:

- Construct a complete daily time-series
- Create lagged meteorological predictors
- Add temporal features
- Evaluate candidate ARIMAX specifications
- Select ARIMA order
- Walk-forward forecasting
- Evaluate RMSE, MAE and R²
- Analyse 95% prediction intervals

---

## Reproducibility

Reusable data-processing and modelling functions will gradually be moved into the `src/` directory.

The notebooks will focus on the analysis, results and interpretation rather than duplicating large blocks of implementation code.
