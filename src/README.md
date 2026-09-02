# Source Code

Reusable project code will be organised in this directory.

The aim is to separate data processing, feature engineering and modelling logic from the exploratory notebooks so that the workflow is easier to maintain, test and reproduce.

## Planned Structure

```text
src/
├── data_processing.py
├── feature_engineering.py
├── classification.py
├── regression.py
├── arimax_forecasting.py
└── evaluation.py
