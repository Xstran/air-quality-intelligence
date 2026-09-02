# Air Quality Intelligence

Machine learning and time-series analysis for predicting daily **PM2.5 concentrations** and identifying **high-pollution days** in Manchester using meteorological and historical air-quality data.

The project combines data quality control, feature engineering, regression, imbalanced classification, time-series forecasting and model explainability.

---

## Problem

PM2.5 concentrations are influenced by both recent pollution levels and meteorological conditions such as wind, precipitation, temperature and atmospheric pressure.

This project investigates two related tasks:

1. **Regression** — predict daily PM2.5 concentration
2. **Classification** — identify days exceeding the WHO 24-hour PM2.5 guideline of **15 µg/m³**

A separate ARIMAX experiment evaluates genuine **one-day-ahead forecasting**.

---

## Key Results

### Regression

**Tuned Gradient Boosting Regressor**

- R²: **0.664**
- RMSE: **2.598 µg/m³**
- MAE: **1.923 µg/m³**

### High-Pollution Classification

**Cost-Sensitive Gradient Boosting**

- Balanced Accuracy: **0.830**
- Recall: **0.704**
- F1-score: **0.655**
- Detected **19 of 27** high-pollution days in the 2023 evaluation period

### One-Day-Ahead Forecasting

**ARIMAX (1,0,0)**

- R²: **0.518**
- RMSE: **3.107 µg/m³**
- MAE: **2.234 µg/m³**

---

## Data

The analysis covers **1 January 2021 to 5 November 2023**.

### Air Quality

Hourly PM2.5 observations were obtained from the **Manchester Piccadilly UK-AIR monitoring station (MAN3)**.

Daily PM2.5 means were retained only when at least **18 of 24 hourly measurements** were available, corresponding to a 75% daily completeness requirement.

### Meteorology

Meteorological variables were obtained from the **Open-Meteo Historical Weather API using ERA5 data**.

Final meteorological predictors included:

- Mean sea-level pressure
- Precipitation
- Mean wind speed
- Maximum wind gust
- Mean temperature
- Mean dew-point temperature
- Dominant wind-direction sine component
- Dominant wind-direction cosine component

Historical PM2.5 features were also created:

- `pm25_lag1`
- `pm25_lag2`

---

## Methodology

The final machine-learning dataset contained **1,010 observations**.

```text
2021–2022
Development data
      │
      ├── Expanding-window TimeSeriesSplit
      │
      └── Hyperparameter optimisation
      │
      ▼
Model selection
      │
      ▼
2023
Unseen final evaluation
