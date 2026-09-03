import numpy as np


# Final meteorological predictors
MET_FEATURES = [
    "pressure_msl_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_gusts_10m_max",
    "temperature_2m_mean",
    "dew_point_2m_mean",
    "wind_direction_sin",
    "wind_direction_cos"
]

# Historical PM2.5 predictors
LAG_FEATURES = [
    "pm25_lag1",
    "pm25_lag2"
]

# Two feature sets used throughout modelling
FEATURES_METEOROLOGY_ONLY = MET_FEATURES.copy()
FEATURES_METEOROLOGY_LAG = MET_FEATURES + LAG_FEATURES

# Modelling targets
PM25_TARGET = "pm25_daily_mean"
CLASSIFICATION_TARGET = "high_pm25_day"

# WHO daily PM2.5 threshold used in the project
HIGH_PM25_THRESHOLD = 15


def add_wind_direction_features(df):
    """
    Convert wind direction in degrees into sine and cosine features.

    Wind direction is circular, so sine and cosine encoding avoids treating
    directions such as 359° and 1° as being far apart.
    """
    df = df.copy()

    # Convert wind direction from degrees to radians
    wind_direction_rad = np.deg2rad(
        df["wind_direction_10m_dominant"]
    )

    # Create circular wind-direction features
    df["wind_direction_sin"] = np.sin(wind_direction_rad)
    df["wind_direction_cos"] = np.cos(wind_direction_rad)

    return df


def create_modelling_features(df):
    """
    Create PM2.5 lag features and the high-pollution classification target.

    Only observed PM2.5 values are used for modelling. Rows are retained only
    when lag-1 and lag-2 correspond to the previous one and two calendar days.
    """
    # Create a copy so the original merged dataset remains unchanged
    feature_data = df.copy()

    # Ensure chronological ordering before creating lag features
    feature_data = (
        feature_data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Create lag features from observed PM2.5 values
    feature_data["pm25_lag1"] = feature_data[PM25_TARGET].shift(1)
    feature_data["pm25_lag2"] = feature_data[PM25_TARGET].shift(2)

    # Check that lagged observations represent consecutive calendar days
    feature_data["date_gap_lag1"] = (
        feature_data["Date"].diff().dt.days
    )

    feature_data["date_gap_lag2"] = (
        feature_data["Date"].diff(2).dt.days
    )

    # Keep rows with an observed target, valid lags and calendar continuity
    valid_model_rows = (
        feature_data[PM25_TARGET].notna()
        & feature_data["pm25_lag1"].notna()
        & feature_data["pm25_lag2"].notna()
        & (feature_data["date_gap_lag1"] == 1)
        & (feature_data["date_gap_lag2"] == 2)
    )

    feature_data = (
        feature_data.loc[valid_model_rows]
        .copy()
        .reset_index(drop=True)
    )

    # Binary target:
    # 1 = daily PM2.5 above 15 µg/m³
    # 0 = daily PM2.5 at or below 15 µg/m³
    feature_data[CLASSIFICATION_TARGET] = (
        feature_data[PM25_TARGET] > HIGH_PM25_THRESHOLD
    ).astype(int)

    # Final integrity checks
    assert feature_data[PM25_TARGET].notna().all()
    assert feature_data["pm25_lag1"].notna().all()
    assert feature_data["pm25_lag2"].notna().all()
    assert feature_data["was_interpolated"].sum() == 0

    return feature_data
