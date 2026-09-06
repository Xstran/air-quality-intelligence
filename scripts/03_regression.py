
# 03 - DAILY PM2.5 REGRESSION
# Predict daily PM2.5 concentration
# Development: 2021-2022 | Final holdout: 2023


from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# CELL 1 - PROJECT PATHS AND SOURCE MODULES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    create_modelling_features,
    FEATURES_METEOROLOGY_ONLY,
    FEATURES_METEOROLOGY_LAG,
    PM25_TARGET,
)

from src.regression import evaluate_regressor, tune_regressor

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_daily_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)



# CELL 2 - LOAD PREPARED DATA


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}\n"
        "Run scripts/01_data_preparation_eda.py first."
    )

df_merged = pd.read_csv(DATA_PATH, parse_dates=["Date"])

print("Loaded prepared dataset")
print("Shape:", df_merged.shape)
print(
    "Date range:",
    df_merged["Date"].min().date(),
    "to",
    df_merged["Date"].max().date(),
)



# CELL 3 - CREATE MODELLING FEATURES


feature_data = create_modelling_features(df_merged)

print("\nFinal modelling rows:", len(feature_data))
print("Regression target:", PM25_TARGET)



# CELL 4 - CHRONOLOGICAL TRAIN / TEST SPLIT


# 2021-2022 development data
train_data = feature_data[feature_data["year"].isin([2021, 2022])].copy()

# 2023 untouched final holdout
test_data = feature_data[feature_data["year"] == 2023].copy()

# Feature set 1: meteorology only
X_train_met = train_data[FEATURES_METEOROLOGY_ONLY]
X_test_met = test_data[FEATURES_METEOROLOGY_ONLY]

# Feature set 2: meteorology + PM2.5 lag features
X_train_met_lag = train_data[FEATURES_METEOROLOGY_LAG]
X_test_met_lag = test_data[FEATURES_METEOROLOGY_LAG]

# Continuous PM2.5 regression target
y_train_reg = train_data[PM25_TARGET]
y_test_reg = test_data[PM25_TARGET]

print("\nRegression target summary:")
print("Train samples:", len(y_train_reg))
print("Test samples:", len(y_test_reg))
print(f"Train mean PM2.5: {y_train_reg.mean():.2f} µg/m³")
print(f"Test mean PM2.5: {y_test_reg.mean():.2f} µg/m³")
print(f"Train max PM2.5: {y_train_reg.max():.2f} µg/m³")
print(f"Test max PM2.5: {y_test_reg.max():.2f} µg/m³")

print(
    "\nTraining date range:",
    train_data["Date"].min().date(),
    "to",
    train_data["Date"].max().date(),
)
print(
    "Testing date range:",
    test_data["Date"].min().date(),
    "to",
    test_data["Date"].max().date(),
)

# Guard against temporal leakage
assert train_data["Date"].max() < test_data["Date"].min()
print("\nNo temporal train/test overlap confirmed.")



# CELL 5 - FEATURE SETS


feature_sets = {
    "Meteorology only": {
        "X_train": X_train_met,
        "X_test": X_test_met,
    },
    "Meteorology + lag": {
        "X_train": X_train_met_lag,
        "X_test": X_test_met_lag,
    },
}



# CELL 6 - EXPANDING-WINDOW TIMESERIESSPLIT


tscv = TimeSeriesSplit(n_splits=3)

print("\nTimeSeriesSplit with 3 chronological folds:")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_met), start=1):
    train_dates = train_data.iloc[train_idx]["Date"]
    val_dates = train_data.iloc[val_idx]["Date"]

    print(f"\nFold {fold}")
    print(
        "  Train:",
        train_dates.min().date(),
        "to",
        train_dates.max().date(),
        f"({len(train_idx)} observations)",
    )
    print(
        "  Validation:",
        val_dates.min().date(),
        "to",
        val_dates.max().date(),
        f"({len(val_idx)} observations)",
    )


# CELL 7 - BASELINE REGRESSION MODELS


regression_baseline_models = {
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        random_state=42
    ),
}

print("\nRegression baseline models defined.")



# CELL 8 - HYPERPARAMETER SEARCH SPACES


rf_reg_param_dist = {
    "n_estimators": randint(100, 701),
    "max_depth": [15, 20, 25, 30, None],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 6),
    "max_features": [0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    "bootstrap": [True],
}

gb_reg_param_dist = {
    "n_estimators": randint(100, 701),
    "learning_rate": loguniform(0.008, 0.05),
    "max_depth": [2, 3, 4],
    "min_samples_split": randint(10, 26),
    "min_samples_leaf": randint(1, 8),
    "subsample": uniform(0.70, 0.25),
    "max_features": ["sqrt", "log2", 0.5, 0.7, 0.9, None],
}



# CELL 9 - TUNE REGRESSION MODELS


tuning_regressors = {
    "Tuned Random Forest Regressor": {
        "model": RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        "param_dist": rf_reg_param_dist,
    },
    "Tuned Gradient Boosting Regressor": {
        "model": GradientBoostingRegressor(
            random_state=42
        ),
        "param_dist": gb_reg_param_dist,
    },
}

tuned_reg_searches = {}

for model_name, config in tuning_regressors.items():
    for feature_set_name, data in feature_sets.items():

        print(f"\nTuning {model_name} - {feature_set_name}")

        search = tune_regressor(
            model=config["model"],
            param_dist=config["param_dist"],
            X_train=data["X_train"],
            y_train=y_train_reg,
            cv=tscv,
            n_iter=100,
        )

        tuned_reg_searches[(model_name, feature_set_name)] = search

        print("Best parameters:", search.best_params_)
        print(f"Best CV RMSE: {-search.best_score_:.3f}")



# CELL 10 - DEVELOPMENT CV COMPARISON

cv_rows = []

for (model_name, feature_set_name), search in tuned_reg_searches.items():
    best_index = search.best_index_

    cv_rows.append({
        "Model": model_name,
        "Feature set": feature_set_name,
        "Training CV RMSE": -search.cv_results_["mean_train_score"][best_index],
        "Validation CV RMSE": -search.best_score_,
        "CV RMSE standard deviation": search.cv_results_["std_test_score"][best_index],
    })

reg_cv_results = (
    pd.DataFrame(cv_rows)
    .sort_values("Validation CV RMSE")
    .reset_index(drop=True)
)

print("\nRegression model selection using 2021-2022 only:")
print(reg_cv_results.round(3))

reg_cv_results.to_csv(
    RESULTS_DIR / "regression_cv_results.csv",
    index=False,
)



# CELL 11 - SELECT FINAL REGRESSOR BEFORE USING 2023


# best_score_ contains negative RMSE, so the largest value is best
selected_reg_key = max(
    tuned_reg_searches,
    key=lambda key: tuned_reg_searches[key].best_score_,
)

selected_reg_model_name, selected_reg_feature_set = selected_reg_key

print("\nREGRESSOR SELECTED BEFORE 2023 EVALUATION")
print("Model:", selected_reg_model_name)
print("Feature set:", selected_reg_feature_set)
print(
    "Development CV RMSE:",
    round(-tuned_reg_searches[selected_reg_key].best_score_, 3),
)


# CELL 12 - FINAL 2023 EVALUATION


reg_results = []
reg_outputs = {}

# Evaluate baseline regressors
for model_name, model in regression_baseline_models.items():
    for feature_set_name, data in feature_sets.items():

        result, output = evaluate_regressor(
            model=model,
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=y_train_reg,
            y_test=y_test_reg,
            model_name=model_name,
            feature_set_name=feature_set_name,
        )

        reg_results.append(result)
        reg_outputs[(model_name, feature_set_name)] = output


# Evaluate tuned regressors
for (model_name, feature_set_name), search in tuned_reg_searches.items():
    data = feature_sets[feature_set_name]

    result, output = evaluate_regressor(
        model=search.best_estimator_,
        X_train=data["X_train"],
        X_test=data["X_test"],
        y_train=y_train_reg,
        y_test=y_test_reg,
        model_name=model_name,
        feature_set_name=feature_set_name,
    )

    reg_results.append(result)
    reg_outputs[(model_name, feature_set_name)] = output


reg_results_df = (
    pd.DataFrame(reg_results)
    .drop_duplicates(
        subset=["Model", "Feature set"],
        keep="last",
    )
    .reset_index(drop=True)
)

reg_results_df["Selected by development CV"] = (
    (reg_results_df["Model"] == selected_reg_model_name)
    & (reg_results_df["Feature set"] == selected_reg_feature_set)
)

reg_results_df = (
    reg_results_df
    .sort_values(
        by=["Selected by development CV", "Model", "Feature set"],
        ascending=[False, True, True],
    )
    .reset_index(drop=True)
)

print("\nFinal 2023 regression results:")
print(reg_results_df.round(3))

reg_results_df.to_csv(
    RESULTS_DIR / "regression_model_comparison.csv",
    index=False,
)


# CELL 13 - RETRIEVE SELECTED REGRESSION MODEL


best_reg_key = (
    selected_reg_model_name,
    selected_reg_feature_set,
)

best_reg_output = reg_outputs[best_reg_key]
best_reg_model = best_reg_output["model"]
best_reg_pred = best_reg_output["pred"]
X_train_reg_best = best_reg_output["X_train"]
X_test_reg_best = best_reg_output["X_test"]

selected_reg_metrics = reg_results_df[
    reg_results_df["Selected by development CV"]
].iloc[0]

print("\nBest regression model:", selected_reg_model_name)
print("Best regression feature set:", selected_reg_feature_set)



# CELL 14 - TRAIN VS TEST PERFORMANCE


train_pred = best_reg_model.predict(X_train_reg_best)

train_metrics = {
    "Dataset": "Training",
    "RMSE": np.sqrt(mean_squared_error(y_train_reg, train_pred)),
    "MAE": mean_absolute_error(y_train_reg, train_pred),
    "R2": r2_score(y_train_reg, train_pred),
}

test_metrics = {
    "Dataset": "Testing",
    "RMSE": np.sqrt(mean_squared_error(y_test_reg, best_reg_pred)),
    "MAE": mean_absolute_error(y_test_reg, best_reg_pred),
    "R2": r2_score(y_test_reg, best_reg_pred),
}

train_test_df = pd.DataFrame([train_metrics, test_metrics])

print("\nSelected-model train/test comparison:")
print(train_test_df.round(3))

train_test_df.to_csv(
    RESULTS_DIR / "regression_train_test_comparison.csv",
    index=False,
)



# CELL 15 - OBSERVED VS PREDICTED PM2.5


plot_dates = test_data["Date"]

plt.figure(figsize=(14, 5))

plt.plot(
    plot_dates,
    y_test_reg.values,
    linewidth=1,
    marker="o",
    markersize=2,
    label="Observed PM2.5",
)

plt.plot(
    plot_dates,
    best_reg_pred,
    linewidth=1,
    marker="o",
    markersize=2,
    label="Predicted PM2.5",
)

plt.axhline(
    y=15,
    linestyle="--",
    linewidth=0.8,
    label="WHO 15 µg/m³",
)

plt.title(
    f"Observed vs Predicted Daily PM2.5 - {selected_reg_model_name}\n"
    f"RMSE={selected_reg_metrics['RMSE']:.2f}, "
    f"MAE={selected_reg_metrics['MAE']:.2f}, "
    f"R²={selected_reg_metrics['R2']:.3f}"
)

plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "regression_observed_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# CELL 16 - FEATURE IMPORTANCE


reg_importance_df = (
    pd.DataFrame({
        "Feature": X_train_reg_best.columns,
        "Importance": best_reg_model.feature_importances_,
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

print("\nRegression feature importance:")
print(reg_importance_df.round(4))

reg_importance_df.to_csv(
    RESULTS_DIR / "regression_feature_importance.csv",
    index=False,
)

plt.figure(figsize=(8, 5))
plt.barh(
    reg_importance_df["Feature"],
    reg_importance_df["Importance"],
)
plt.gca().invert_yaxis()
plt.title("Feature Importance - Selected Regression Model")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "regression_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# CELL 17 - SHAP EXPLAINABILITY


print("\nCalculating regression SHAP values...")

reg_explainer = shap.TreeExplainer(best_reg_model)
reg_shap_values = reg_explainer.shap_values(X_test_reg_best)

reg_shap_importance_df = (
    pd.DataFrame({
        "Feature": X_test_reg_best.columns,
        "Mean absolute SHAP": np.abs(reg_shap_values).mean(axis=0),
    })
    .sort_values("Mean absolute SHAP", ascending=False)
    .reset_index(drop=True)
)

print("\nRegression SHAP importance:")
print(reg_shap_importance_df.round(4))

reg_shap_importance_df.to_csv(
    RESULTS_DIR / "regression_shap_importance.csv",
    index=False,
)

# Global SHAP bar plot
shap.summary_plot(
    reg_shap_values,
    X_test_reg_best,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "regression_shap_bar.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# SHAP beeswarm plot
shap.summary_plot(
    reg_shap_values,
    X_test_reg_best,
    show=False,
)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "regression_shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# CELL 18 - FINAL SUMMARY


print("FINAL REGRESSION SUMMARY")


print("Model:", selected_reg_model_name)
print("Feature set:", selected_reg_feature_set)
print(f"RMSE: {selected_reg_metrics['RMSE']:.3f} µg/m³")
print(f"MAE: {selected_reg_metrics['MAE']:.3f} µg/m³")
print(f"R²: {selected_reg_metrics['R2']:.3f}")
print("\nResults saved to:", RESULTS_DIR)
