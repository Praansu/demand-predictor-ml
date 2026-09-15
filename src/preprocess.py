"""Feature engineering for parking demand prediction.

Transforms raw timestamped occupancy data into ML-ready features.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def add_time_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """Add time-based features from timestamp column."""
    df = df.copy()
    dt = pd.to_datetime(df[timestamp_col])

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek  # 0=Mon
    df["day_of_month"] = dt.dt.day
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encoding for hour (preserves 23->0 continuity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cyclical encoding for day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str = "occupancy", lags: list = None
) -> pd.DataFrame:
    """Add lag features for time-series prediction."""
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 24h, 48h, 1w

    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame, target_col: str = "occupancy", windows: list = None
) -> pd.DataFrame:
    """Add rolling window statistics."""
    if windows is None:
        windows = [3, 6, 12, 24]

    df = df.copy()
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
        df[f"{target_col}_rolling_min_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).min()
        )
        df[f"{target_col}_rolling_max_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).max()
        )

    return df


def add_holiday_features(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """Add holiday indicators (US holidays)."""
    try:
        import holidays

        us_holidays = holidays.US()
        df = df.copy()
        df["is_holiday"] = (
            pd.to_datetime(df[timestamp_col])
            .dt.date.apply(lambda d: d in holidays.US())
            .astype(int)
        )
    except ImportError:
        df["is_holiday"] = 0
    return df


def prepare_features(
    df: pd.DataFrame,
    target_col: str = "occupancy",
    timestamp_col: str = "timestamp",
    add_lags: bool = True,
    add_rolling: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Full feature engineering pipeline.

    Returns:
        X: Feature DataFrame
        y: Target Series
    """
    df = df.copy()

    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Time features
    df = add_time_features(df, timestamp_col)

    # Holiday features
    df = add_holiday_features(df, timestamp_col)

    # Lag features
    if add_lags:
        df = add_lag_features(df, target_col)

    # Rolling features
    if add_rolling:
        df = add_rolling_features(df, target_col)

    # Drop rows with NaN from lag/rolling features
    df = df.dropna().reset_index(drop=True)

    # Separate features and target
    # Exclude non-feature columns
    exclude_cols = [target_col, "timestamp", "capacity"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    return X, y


def train_test_split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    gap: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-series aware train/test split (no random shuffle)."""
    n = len(X)
    test_start = int(n * (1 - test_size))

    # Add gap to prevent leakage
    train_end = test_start - gap

    X_train = X.iloc[:train_end]
    X_test = X.iloc[test_start:]
    y_train = y.iloc[:train_end]
    y_test = y.iloc[test_start:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick test
    import sys

    sys.path.append(".")
    from generate_data import generate_parking_data

    df = generate_parking_data(days=30)
    X, y = prepare_features(df)
    print(f"Features: {X.shape[1]}, Samples: {len(X)}")
    print(f"Feature columns: {list(X.columns)}")
