
# 00 - PROJECT SETUP VALIDATION
# Quickly checks repository structure, dependencies and data.

from pathlib import Path
import importlib
import sys

import pandas as pd



# PROJECT PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# CHECK REQUIRED FILES


required_files = [
    "README.md",
    "requirements.txt",
    "data/raw/MAN3_PM25_2021.csv",
    "data/raw/MAN3_PM25_2022.csv",
    "data/raw/MAN3_PM25_2023.csv",
    "data/processed/merged_daily_data.csv",
    "src/__init__.py",
    "src/data_processing.py",
    "src/feature_engineering.py",
    "src/classification.py",
    "src/regression.py",
    "src/arimax_forecasting.py",
    "scripts/01_data_preparation_eda.py",
    "scripts/02_classification.py",
    "scripts/03_regression.py",
    "scripts/04_arimax_forecasting.py",
]

missing_files = [
    file
    for file in required_files
    if not (PROJECT_ROOT / file).exists()
]

if missing_files:
    print("Missing required files:")
    for file in missing_files:
        print(f"  - {file}")
    raise FileNotFoundError("Project structure validation failed.")

print("✓ Required project files found")


# CHECK PYTHON DEPENDENCIES


required_packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "requests",
    "sklearn",
    "scipy",
    "imblearn",
    "statsmodels",
    "pmdarima",
    "shap",
]

missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("\nMissing Python packages:")
    for package in missing_packages:
        print(f"  - {package}")

    raise ImportError(
        "Install the project dependencies with: "
        "pip install -r requirements.txt"
    )

print("✓ Required Python packages available")



# CHECK SOURCE MODULE IMPORTS


from src.feature_engineering import create_modelling_features
from src.classification import evaluate_classifier, tune_classifier
from src.regression import evaluate_regressor, tune_regressor
from src.arimax_forecasting import (
    create_arimax_predictors,
    fit_arimax,
    walk_forward_forecast,
    evaluate_arimax_forecast,
)

print("✓ Source modules import successfully")



# CHECK PROCESSED DATASET


data_path = PROJECT_ROOT / "data" / "processed" / "merged_daily_data.csv"

data = pd.read_csv(
    data_path,
    parse_dates=["Date"],
)

required_columns = [
    "Date",
    "pm25_daily_mean",
    "pm25_daily_mean_interpolated",
    "was_interpolated",
    "year",
    "month",
    "season",
    "temperature_2m_mean",
    "dew_point_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "wind_direction_sin",
    "wind_direction_cos",
    "pressure_msl_mean",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise KeyError(
        f"Processed dataset is missing columns: {missing_columns}"
    )

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found in processed dataset.")

print("✓ Processed dataset loaded successfully")
print("  Rows:", len(data))
print("  Columns:", len(data.columns))
print(
    "  Date range:",
    data["Date"].min().date(),
    "to",
    data["Date"].max().date(),
)


# CHECK MODELLING FEATURE CREATION


feature_data = create_modelling_features(data)

if len(feature_data) != 1010:
    raise ValueError(
        f"Expected 1,010 modelling rows, found {len(feature_data)}."
    )

print("✓ Modelling features created successfully")
print("  Final modelling observations:", len(feature_data))



# CHECK KEY RESULT FIGURES


result_files = [
    "correlation_heatmap.png",
    "classification_confusion_matrix.png",
    "classification_precision_recall_curve.png",
    "classification_roc_curve.png",
    "classification_shap_summary.png",
    "regression_observed_vs_predicted.png",
    "regression_shap_summary.png",
    "arimax_observed_vs_predicted.png",
]

missing_results = [
    file
    for file in result_files
    if not (PROJECT_ROOT / "results" / file).exists()
]

if missing_results:
    print("\nWarning: some result figures are missing:")
    for file in missing_results:
        print(f"  - {file}")
else:
    print("✓ Key result figures found")


# FINAL STATUS


print("\n========================================")
print("PROJECT VALIDATION PASSED")
print("========================================")
print("The repository is ready for modelling.")
