
# 04 - ARIMAX ONE-DAY-AHEAD PM2.5 FORECASTING
#
# Model selection:
#   2021 - training
#   2022 - validation
#
# Final model:
#   2021-2022 - training
#   2023      - walk-forward holdout evaluation


from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pmdarima import auto_arima

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    InterpolationWarning,
    ValueWarning,
)


# CELL 1 - PROJECT PATHS AND SOURCE MODULES


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import MET_FEATURES
from src.arimax_forecasting import (
    create_arimax_predictors,
    fit_arimax,
    walk_forward_forecast,
    evaluate_arimax_forecast,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "merged_daily_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)



# CELL 2 - LOAD OBSERVED DAILY PM2.5


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_PATH}\n"
        "Run scripts/01_data_preparation_eda.py first."
    )

processed_data = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# Use observed PM2.5 only.
# The interpolated EDA series is deliberately excluded.
pm25_data = processed_data[
    ["Date", "pm25_daily_mean"]
].copy()

print("Observed PM2.5 data loaded")
print("Rows available:", len(pm25_data))



# CELL 3 - LOAD COMPLETE METEOROLOGICAL SERIES


# Open-Meteo historical ERA5 endpoint
url = "https://archive-api.open-meteo.com/v1/era5"

# Manchester Piccadilly monitoring-site coordinates
LAT = 53.4815
LON = -2.2379

weather_variables = [
    "pressure_msl_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_gusts_10m_max",
    "temperature_2m_mean",
    "dew_point_2m_mean",
    "wind_direction_10m_dominant",
]

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2021-01-01",
    "end_date": "2023-11-05",
    "daily": ",".join(weather_variables),
    "timezone": "Europe/London",
}

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    weather = pd.DataFrame(response.json()["daily"])

except requests.exceptions.RequestException as error:
    raise RuntimeError(
        f"Could not download Open-Meteo weather data: {error}"
    ) from error

weather = weather.rename(columns={"time": "Date"})
weather["Date"] = pd.to_datetime(weather["Date"])

# Circular wind-direction encoding
wind_radians = np.deg2rad(weather["wind_direction_10m_dominant"])
weather["wind_direction_sin"] = np.sin(wind_radians)
weather["wind_direction_cos"] = np.cos(wind_radians)

missing_features = [
    feature
    for feature in MET_FEATURES
    if feature not in weather.columns
]

if missing_features:
    raise KeyError(
        f"Missing meteorological features: {missing_features}"
    )

print("\nMeteorological data loaded")
print("Weather rows:", len(weather))
print(
    "Weather date range:",
    weather["Date"].min().date(),
    "to",
    weather["Date"].max().date(),
)



# CELL 4 - CREATE COMPLETE DAILY TIME-SERIES DATASET


daily_calendar = pd.DataFrame({
    "Date": pd.date_range(
        start="2021-01-01",
        end="2023-11-05",
        freq="D",
    )
})

ts_data = (
    daily_calendar
    .merge(pm25_data, on="Date", how="left")
    .merge(
        weather[["Date"] + MET_FEATURES],
        on="Date",
        how="left",
    )
    .set_index("Date")
    .sort_index()
)

ts_data = ts_data.rename(
    columns={"pm25_daily_mean": "pm25"}
)

# Confirm complete calendar
expected_index = pd.date_range(
    start="2021-01-01",
    end="2023-11-05",
    freq="D",
)

assert ts_data.index.equals(expected_index)
assert ts_data[MET_FEATURES].isna().sum().sum() == 0

train_ts = ts_data.loc["2021-01-01":"2022-12-31"].copy()
test_ts = ts_data.loc["2023-01-01":"2023-11-05"].copy()

print("\nComplete time-series period:")
print(ts_data.index.min().date(), "to", ts_data.index.max().date())
print("Total calendar days:", len(ts_data))

print("\nTraining period:")
print(train_ts.index.min().date(), "to", train_ts.index.max().date())
print("Training calendar days:", len(train_ts))
print("Observed training targets:", train_ts["pm25"].notna().sum())
print("Missing training targets:", train_ts["pm25"].isna().sum())

print("\nTesting period:")
print(test_ts.index.min().date(), "to", test_ts.index.max().date())
print("Testing calendar days:", len(test_ts))
print("Observed testing targets:", test_ts["pm25"].notna().sum())
print("Missing testing targets:", test_ts["pm25"].isna().sum())

assert train_ts.index.max() < test_ts.index.min()
print("\nDaily time-series dataset prepared successfully.")



# CELL 5 - MODEL-SELECTION SPLIT


# 2021: fit candidate configurations
selection_train_ts = ts_data.loc[
    "2021-01-01":"2021-12-31"
].copy()

# 2022: validate candidate configurations
selection_valid_ts = ts_data.loc[
    "2022-01-01":"2022-12-31"
].copy()

# 2021-2022: final model training
final_train_ts = ts_data.loc[
    "2021-01-01":"2022-12-31"
].copy()

# 2023: untouched final test period
final_test_ts = ts_data.loc[
    "2023-01-01":"2023-11-05"
].copy()

y_selection_train = selection_train_ts["pm25"]
y_selection_valid = selection_valid_ts["pm25"]
y_final_train = final_train_ts["pm25"]
y_final_test = final_test_ts["pm25"]

print("\nModel-selection training period:")
print(
    selection_train_ts.index.min().date(),
    "to",
    selection_train_ts.index.max().date(),
)
print("Observed targets:", y_selection_train.notna().sum())

print("\nModel-selection validation period:")
print(
    selection_valid_ts.index.min().date(),
    "to",
    selection_valid_ts.index.max().date(),
)
print("Observed targets:", y_selection_valid.notna().sum())

print("\nFinal unseen test period:")
print(
    final_test_ts.index.min().date(),
    "to",
    final_test_ts.index.max().date(),
)
print("Observed targets:", y_final_test.notna().sum())

assert selection_train_ts.index.max() < selection_valid_ts.index.min()
assert final_train_ts.index.max() < final_test_ts.index.min()

print("\nChronological model-development split confirmed.")



# CELL 6 - CREATE ONE-DAY-AHEAD ARIMAX PREDICTORS


ts_model_data, weather_lag1_features = create_arimax_predictors(
    ts_data,
    MET_FEATURES,
)

annual_features = [
    "year_sin_1",
    "year_cos_1",
    "year_sin_2",
    "year_cos_2",
]

weekly_features = [
    "week_sin",
    "week_cos",
]

weather_persistence_features = [
    "pressure_3d_mean",
    "wind_speed_3d_mean",
    "wind_gust_3d_mean",
    "temperature_3d_mean",
    "precipitation_3d_sum",
]

print("\nARIMAX predictors created")
print("Previous-day weather features:", len(weather_lag1_features))



# CELL 7 - STATIONARITY TESTING


# Short interpolation is used only to preserve spacing for
# stationarity testing, not as a forecasting target.
stationarity_series = y_selection_train.interpolate(
    method="time",
    limit=2,
    limit_area="inside",
)

assert stationarity_series.notna().all()

# Augmented Dickey-Fuller
adf_result = adfuller(
    stationarity_series,
    regression="c",
    autolag="AIC",
)

adf_statistic = adf_result[0]
adf_p_value = adf_result[1]

# KPSS
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InterpolationWarning)

    kpss_result = kpss(
        stationarity_series,
        regression="c",
        nlags="auto",
    )

kpss_statistic = kpss_result[0]
kpss_p_value = kpss_result[1]

if adf_p_value < 0.05 and kpss_p_value > 0.05:
    stationarity_conclusion = (
        "Both tests support a stationary PM2.5 series"
    )
    selected_d = 0

elif adf_p_value >= 0.05 and kpss_p_value <= 0.05:
    stationarity_conclusion = (
        "Both tests indicate that differencing may be required"
    )
    selected_d = 1

else:
    stationarity_conclusion = (
        "Tests provide mixed evidence; d=0 retained"
    )
    selected_d = 0

print("\nADF stationarity test")
print(f"Test statistic: {adf_statistic:.4f}")
print(f"p-value: {adf_p_value:.4f}")

print("\nKPSS stationarity test")
print(f"Test statistic: {kpss_statistic:.4f}")
print(f"p-value: {kpss_p_value:.4f}")

print("\nConclusion:", stationarity_conclusion)
print("Differencing order:", selected_d)

stationarity_results = pd.DataFrame([
    {
        "Test": "ADF",
        "Statistic": adf_statistic,
        "p-value": adf_p_value,
    },
    {
        "Test": "KPSS",
        "Statistic": kpss_statistic,
        "p-value": kpss_p_value,
    },
])

stationarity_results.to_csv(
    RESULTS_DIR / "arimax_stationarity_tests.csv",
    index=False,
)


# CELL 8 - DEFINE CANDIDATE FEATURE SETS


# Reduced set removes wind speed and dew point because of
# strong relationships with wind gusts and temperature.
reduced_weather_features = [
    feature
    for feature in MET_FEATURES
    if feature not in [
        "wind_speed_10m_mean",
        "dew_point_2m_mean",
    ]
]

reduced_weather_lag1_features = [
    f"{feature}_lag1"
    for feature in reduced_weather_features
]

arimax_feature_sets = {
    "Reduced lagged weather + annual":
        reduced_weather_lag1_features + annual_features,

    "All lagged weather + annual":
        weather_lag1_features + annual_features,

    "Reduced lagged weather + seasonals":
        reduced_weather_lag1_features
        + annual_features
        + weekly_features,

    "Full lagged temporal weather":
        weather_lag1_features
        + weather_persistence_features
        + annual_features
        + weekly_features,
}

# Remove accidental duplicate predictors
arimax_feature_sets = {
    name: list(dict.fromkeys(features))
    for name, features in arimax_feature_sets.items()
}

print("\nCandidate ARIMAX feature sets:")

for name, features in arimax_feature_sets.items():
    print(f"- {name}: {len(features)} predictors")



# CELL 9 - AUTO_ARIMA SEARCH SETTINGS


auto_arima_settings = {
    "d": selected_d,
    "start_p": 0,
    "max_p": 10,
    "start_q": 0,
    "max_q": 5,
    "max_order": 12,
    "seasonal": False,
    "information_criterion": "aicc",
    "stepwise": False,
    "n_jobs": -1,
    "with_intercept": True,
    "suppress_warnings": True,
    "error_action": "ignore",
    "trace": False,
}



# CELL 10 - ARIMAX CONFIGURATION SELECTION


selection_results = []

for feature_set_name, feature_columns in arimax_feature_sets.items():

    # 2021 predictors and target
    X_selection_train_raw = ts_model_data.loc[
        "2021-01-01":"2021-12-31",
        feature_columns,
    ].copy()

    y_selection_train_model = ts_model_data.loc[
        "2021-01-01":"2021-12-31",
        "pm25",
    ].copy()

    # Lagged predictors are unavailable at the start of the series
    first_complete_date = X_selection_train_raw.dropna().index.min()

    if pd.isna(first_complete_date):
        print(
            f"Skipped {feature_set_name}: "
            "no complete training predictors."
        )
        continue

    X_selection_train_raw = (
        X_selection_train_raw
        .loc[first_complete_date:]
        .asfreq("D")
    )

    y_selection_train_model = (
        y_selection_train_model
        .loc[first_complete_date:]
        .asfreq("D")
    )

    if X_selection_train_raw.isna().any().any():
        print(
            f"Skipped {feature_set_name}: "
            "internal training predictor gaps."
        )
        continue

    # 2022 validation predictors and target
    X_selection_valid_raw = (
        ts_model_data.loc[
            "2022-01-01":"2022-12-31",
            feature_columns,
        ]
        .copy()
        .asfreq("D")
    )

    y_selection_valid_model = (
        ts_model_data.loc[
            "2022-01-01":"2022-12-31",
            "pm25",
        ]
        .copy()
        .asfreq("D")
    )

    if X_selection_valid_raw.isna().any().any():
        print(
            f"Skipped {feature_set_name}: "
            "validation predictors contain missing values."
        )
        continue

    # Scale using 2021 only
    selection_scaler = StandardScaler()

    X_selection_train_scaled = pd.DataFrame(
        selection_scaler.fit_transform(X_selection_train_raw),
        columns=feature_columns,
        index=X_selection_train_raw.index,
    ).asfreq("D")

    X_selection_valid_scaled = pd.DataFrame(
        selection_scaler.transform(X_selection_valid_raw),
        columns=feature_columns,
        index=X_selection_valid_raw.index,
    ).asfreq("D")

    # Short interpolation is used only for auto_arima order search
    y_auto_order = y_selection_train_model.interpolate(
        method="time",
        limit=2,
        limit_area="inside",
    )

    assert y_auto_order.notna().all()
    assert X_selection_train_scaled.isna().sum().sum() == 0

    try:
        automatic_model = auto_arima(
            y=y_auto_order,
            X=X_selection_train_scaled,
            **auto_arima_settings,
        )

        selected_order = automatic_model.order

        print(f"\n{feature_set_name}")
        print("Automatically selected order:", selected_order)

        # Refit selected order using the original observed PM2.5 series
        candidate_fitted = fit_arimax(
            y_train=y_selection_train_model,
            X_train=X_selection_train_scaled,
            order=selected_order,
            maxiter=300,
        )

        converged = bool(
            candidate_fitted.mle_retvals.get(
                "converged",
                True,
            )
        )

        # Append 2022 observations without re-estimating parameters
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ValueWarning)

            candidate_extended = candidate_fitted.append(
                endog=y_selection_valid_model,
                exog=X_selection_valid_scaled,
                refit=False,
            )

        prediction_start = len(y_selection_train_model)
        prediction_end = (
            prediction_start
            + len(y_selection_valid_model)
            - 1
        )

        # One-step-ahead predictions across 2022
        validation_forecast = candidate_extended.get_prediction(
            start=prediction_start,
            end=prediction_end,
            dynamic=False,
        )

        validation_predictions = pd.Series(
            np.asarray(validation_forecast.predicted_mean),
            index=y_selection_valid_model.index,
            name="ARIMAX validation prediction",
            dtype="float64",
        )

        validation_mask = (
            y_selection_valid_model.notna()
            & validation_predictions.notna()
        )

        validation_actual = y_selection_valid_model.loc[
            validation_mask
        ]

        validation_predicted = validation_predictions.loc[
            validation_mask
        ]

        if len(validation_actual) == 0:
            continue

        validation_rmse = np.sqrt(
            mean_squared_error(
                validation_actual,
                validation_predicted,
            )
        )

        validation_mae = mean_absolute_error(
            validation_actual,
            validation_predicted,
        )

        validation_r2 = r2_score(
            validation_actual,
            validation_predicted,
        )

        selection_results.append({
            "Feature set": feature_set_name,
            "Order": selected_order,
            "Predictors": len(feature_columns),
            "Evaluated days": int(validation_mask.sum()),
            "RMSE": validation_rmse,
            "MAE": validation_mae,
            "R2": validation_r2,
            "AIC": candidate_fitted.aic,
            "Converged": converged,
        })

    except Exception as error:
        print(f"Skipped {feature_set_name}: {error}")


# CELL 11 - SELECT BEST ARIMAX CONFIGURATION

arimax_selection_results = pd.DataFrame(selection_results)

if arimax_selection_results.empty:
    raise RuntimeError(
        "No ARIMAX configuration fitted successfully."
    )

# Prefer converged models
converged_results = arimax_selection_results[
    arimax_selection_results["Converged"]
].copy()

if not converged_results.empty:
    arimax_selection_results = converged_results

# Primary criterion is validation RMSE
arimax_selection_results = (
    arimax_selection_results
    .sort_values(
        by=["RMSE", "MAE", "Predictors", "AIC"],
        ascending=True,
    )
    .reset_index(drop=True)
)

best_arimax_configuration = arimax_selection_results.iloc[0]

selected_feature_set_name = best_arimax_configuration["Feature set"]
selected_arimax_features = arimax_feature_sets[
    selected_feature_set_name
]
selected_arimax_order = tuple(
    best_arimax_configuration["Order"]
)

print("\nBEST ARIMAX CONFIGURATION")
print("Feature set:", selected_feature_set_name)
print("Order:", selected_arimax_order)
print("Number of predictors:", len(selected_arimax_features))
print(
    "Validation RMSE:",
    round(best_arimax_configuration["RMSE"], 3),
)
print(
    "Validation MAE:",
    round(best_arimax_configuration["MAE"], 3),
)
print(
    "Validation R²:",
    round(best_arimax_configuration["R2"], 3),
)

print("\nConfiguration comparison:")
print(arimax_selection_results.round(3))

arimax_selection_results.to_csv(
    RESULTS_DIR / "arimax_validation_results.csv",
    index=False,
)


# CELL 12 - PREPARE FINAL 2021-2022 / 2023 DATA


X_final_train_raw = (
    ts_model_data.loc[
        "2021-01-01":"2022-12-31",
        selected_arimax_features,
    ]
    .copy()
    .asfreq("D")
)

X_final_test_raw = (
    ts_model_data.loc[
        "2023-01-01":"2023-11-05",
        selected_arimax_features,
    ]
    .copy()
    .asfreq("D")
)

y_final_train = (
    ts_model_data.loc[
        "2021-01-01":"2022-12-31",
        "pm25",
    ]
    .copy()
    .asfreq("D")
)

y_final_test = (
    ts_model_data.loc[
        "2023-01-01":"2023-11-05",
        "pm25",
    ]
    .copy()
    .asfreq("D")
)

# First date where lagged predictors are available
first_complete = X_final_train_raw.dropna().index.min()

if pd.isna(first_complete):
    raise ValueError(
        "No complete rows found for final ARIMAX training."
    )

X_final_train_raw = X_final_train_raw.loc[first_complete:]
y_final_train = y_final_train.loc[first_complete:]

if X_final_train_raw.isna().any().any():
    raise ValueError(
        "Final training predictors contain missing values."
    )

if X_final_test_raw.isna().any().any():
    raise ValueError(
        "Final test predictors contain missing values."
    )



# CELL 13 - SCALE FINAL PREDICTORS

# Fit scaler only on 2021-2022
final_scaler = StandardScaler()

X_final_train_scaled = pd.DataFrame(
    final_scaler.fit_transform(X_final_train_raw),
    columns=selected_arimax_features,
    index=X_final_train_raw.index,
).asfreq("D")

# Apply same transformation to unseen 2023 predictors
X_final_test_scaled = pd.DataFrame(
    final_scaler.transform(X_final_test_raw),
    columns=selected_arimax_features,
    index=X_final_test_raw.index,
).asfreq("D")



# CELL 14 - FIT FINAL ARIMAX MODEL

final_fitted = fit_arimax(
    y_train=y_final_train,
    X_train=X_final_train_scaled,
    order=selected_arimax_order,
    maxiter=500,
)

converged = bool(
    final_fitted.mle_retvals.get(
        "converged",
        True,
    )
)

print("\nFinal selected feature set:", selected_feature_set_name)
print("Final ARIMAX order:", selected_arimax_order)

print(
    "Training period:",
    y_final_train.index.min().date(),
    "to",
    y_final_train.index.max().date(),
)

print("Training calendar days:", len(y_final_train))
print("Observed training targets:", y_final_train.notna().sum())
print("Model converged:", converged)
print("Training AIC:", round(final_fitted.aic, 3))



# CELL 15 - WALK-FORWARD FORECASTING ON 2023


predictions, lower_limits, upper_limits = walk_forward_forecast(
    fitted_model=final_fitted,
    y_test=y_final_test,
    X_test=X_final_test_scaled,
    alpha=0.05,
)

print(
    "\nWalk-forward predictions created:",
    len(predictions),
)


# CELL 16 - FINAL 2023 EVALUATION


final_metrics = evaluate_arimax_forecast(
    y_test=y_final_test,
    predictions=predictions,
    lower_limits=lower_limits,
    upper_limits=upper_limits,
)

final_result = pd.DataFrame([{
    "Model": f"Walk-forward ARIMAX{selected_arimax_order}",
    "Feature set": selected_feature_set_name,
    **final_metrics,
}])

print("\nFINAL ARIMAX EVALUATION")
print(final_result.round(3))

final_result.to_csv(
    RESULTS_DIR / "arimax_final_results.csv",
    index=False,
)



# CELL 17 - SAVE DAILY FORECASTS


forecast_results = pd.DataFrame({
    "Actual PM2.5": y_final_test,
    "Predicted PM2.5": predictions,
    "Lower 95%": lower_limits,
    "Upper 95%": upper_limits,
})

forecast_results.index.name = "Date"

forecast_results.to_csv(
    RESULTS_DIR / "arimax_2023_forecasts.csv"
)



# CELL 18 - OBSERVED VS PREDICTED FORECAST PLOT

mask = y_final_test.notna() & predictions.notna()

actual = y_final_test.loc[mask]
predicted = predictions.loc[mask]

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

mae = mean_absolute_error(
    actual,
    predicted,
)

r2 = r2_score(
    actual,
    predicted,
)

plt.figure(figsize=(14, 5))

plt.plot(
    actual.index,
    actual,
    label="Observed PM2.5",
    linewidth=1.3,
)

plt.plot(
    predicted.index,
    predicted,
    label="Predicted PM2.5",
    linewidth=1.3,
)

plt.fill_between(
    predicted.index,
    lower_limits.loc[mask],
    upper_limits.loc[mask],
    alpha=0.2,
    label="95% Prediction Interval",
)

plt.title(
    f"Observed vs Predicted Daily PM2.5 in 2023\n"
    f"Walk-forward ARIMAX{selected_arimax_order} | "
    f"RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}"
)

plt.xlabel("Date")
plt.ylabel("Daily PM2.5 (µg/m³)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "arimax_observed_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# CELL 19 - RESIDUAL DIAGNOSTICS


arimax_residuals = pd.Series(
    final_fitted.filter_results.standardized_forecasts_error[0],
    index=y_final_train.index,
).dropna()

ljung_box = acorr_ljungbox(
    arimax_residuals,
    lags=[7, 14],
    model_df=1,
    return_df=True,
)

print("\nLjung-Box residual diagnostics:")
print(ljung_box.round(4))

ljung_box.to_csv(
    RESULTS_DIR / "arimax_ljung_box.csv"
)



# CELL 20 - FINAL SUMMARY



print("FINAL ARIMAX SUMMARY")


print("Model:", f"ARIMAX{selected_arimax_order}")
print("Feature set:", selected_feature_set_name)
print("Evaluated days:", final_metrics["Evaluated days"])
print(f"RMSE: {final_metrics['RMSE']:.3f} µg/m³")
print(f"MAE: {final_metrics['MAE']:.3f} µg/m³")
print(f"R²: {final_metrics['R2']:.3f}")
print(
    "95% interval coverage:",
    f"{final_metrics['95% interval coverage (%)']:.3f}%",
)
print(
    "Mean interval width:",
    f"{final_metrics['Mean interval width']:.3f} µg/m³",
)

print("\nResults saved to:", RESULTS_DIR)
