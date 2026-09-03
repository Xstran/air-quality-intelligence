import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    ValueWarning
)


ANNUAL_FEATURES = [
    "year_sin_1",
    "year_cos_1",
    "year_sin_2",
    "year_cos_2"
]

WEEKLY_FEATURES = [
    "week_sin",
    "week_cos"
]

WEATHER_PERSISTENCE_FEATURES = [
    "pressure_3d_mean",
    "wind_speed_3d_mean",
    "wind_gust_3d_mean",
    "temperature_3d_mean",
    "precipitation_3d_sum"
]


def create_arimax_predictors(df, met_features):
    """
    Create lagged meteorological and calendar predictors for
    one-day-ahead ARIMAX forecasting.
    """
    ts_model_data = df.copy()

    # Confirm a complete daily calendar before creating lags
    expected_daily_index = pd.date_range(
        start=ts_model_data.index.min(),
        end=ts_model_data.index.max(),
        freq="D"
    )

    if not ts_model_data.index.equals(expected_daily_index):
        raise ValueError(
            "ARIMAX data must contain a complete daily calendar."
        )

    # Annual and weekly cyclical features
    day_of_year = ts_model_data.index.dayofyear
    day_of_week = ts_model_data.index.dayofweek

    ts_model_data["year_sin_1"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    ts_model_data["year_cos_1"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    ts_model_data["year_sin_2"] = np.sin(
        4 * np.pi * day_of_year / 365.25
    )
    ts_model_data["year_cos_2"] = np.cos(
        4 * np.pi * day_of_year / 365.25
    )

    ts_model_data["week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    ts_model_data["week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # Previous-day meteorological predictors
    weather_lag1_features = []

    for feature in met_features:
        lag_feature = f"{feature}_lag1"
        ts_model_data[lag_feature] = (
            ts_model_data[feature].shift(1)
        )
        weather_lag1_features.append(lag_feature)

    # Three-day weather persistence using information
    # available only up to the previous day
    ts_model_data["pressure_3d_mean"] = (
        ts_model_data["pressure_msl_mean"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    ts_model_data["wind_speed_3d_mean"] = (
        ts_model_data["wind_speed_10m_mean"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    ts_model_data["wind_gust_3d_mean"] = (
        ts_model_data["wind_gusts_10m_max"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    ts_model_data["temperature_3d_mean"] = (
        ts_model_data["temperature_2m_mean"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    ts_model_data["precipitation_3d_sum"] = (
        ts_model_data["precipitation_sum"]
        .shift(1)
        .rolling(3)
        .sum()
    )

    return ts_model_data, weather_lag1_features


def fit_arimax(
    y_train,
    X_train,
    order,
    maxiter=500
):
    """
    Fit an ARIMAX model using the selected order and
    exogenous predictors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ValueWarning
        )
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning
        )

        model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(
            disp=False,
            maxiter=maxiter
        )

    return fitted_model


def walk_forward_forecast(
    fitted_model,
    y_test,
    X_test,
    alpha=0.05
):
    """
    Generate one-day-ahead walk-forward ARIMAX forecasts.

    After each forecast, the model state is updated with the
    newly observed PM2.5 value without re-estimating parameters.
    """
    current_model = fitted_model

    predictions = []
    lower_limits = []
    upper_limits = []

    for date, actual in y_test.items():

        x_today = X_test.loc[[date]]

        # Generate one-day-ahead forecast
        forecast = current_model.get_forecast(
            steps=1,
            exog=x_today
        )

        prediction = float(
            forecast.predicted_mean.iloc[0]
        )

        # Prediction interval
        interval = forecast.conf_int(
            alpha=alpha
        )

        lower = float(interval.iloc[0, 0])
        upper = float(interval.iloc[0, 1])

        predictions.append(prediction)
        lower_limits.append(lower)
        upper_limits.append(upper)

        # Update model state using the newly observed value
        # without re-estimating model parameters
        new_observation = pd.Series(
            [actual],
            index=pd.DatetimeIndex([date]),
            name=y_test.name
        )

        current_model = current_model.append(
            new_observation,
            exog=x_today,
            refit=False
        )

    predictions = pd.Series(
        predictions,
        index=y_test.index,
        name="Predicted PM2.5"
    )

    lower_limits = pd.Series(
        lower_limits,
        index=y_test.index,
        name="Lower 95%"
    )

    upper_limits = pd.Series(
        upper_limits,
        index=y_test.index,
        name="Upper 95%"
    )

    return predictions, lower_limits, upper_limits



def evaluate_arimax_forecast(
    y_test,
    predictions,
    lower_limits,
    upper_limits
):
    """
    Evaluate walk-forward ARIMAX predictions on dates
    where observed PM2.5 values are available.
    """
    mask = (
        y_test.notna()
        & predictions.notna()
    )

    actual = y_test.loc[mask]
    predicted = predictions.loc[mask]

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    lower = lower_limits.loc[mask]
    upper = upper_limits.loc[mask]

    coverage = (
        (
            (actual >= lower)
            & (actual <= upper)
        ).mean()
        * 100
    )

    mean_interval_width = (
        upper - lower
    ).mean()

    return {
        "Evaluated days": int(mask.sum()),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "95% interval coverage (%)": coverage,
        "Mean interval width": mean_interval_width
    }





