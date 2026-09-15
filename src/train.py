"""Model training for parking demand prediction.

Trains and saves multiple models for comparison.
"""

import argparse
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib

from preprocess import prepare_features, train_test_split_time_series


MODELS = {
    "linear": LinearRegression(),
    "rf": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "xgb": xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> object:
    """Train a single model."""
    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}"
        )

    model = MODELS[model_name]
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    return model


def evaluate_model_on_split(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    return evaluate_model(y_test.values, y_pred)


def cross_validate_model(
    model_name: str, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> dict:
    """Cross-validate model using time-series split."""
    from sklearn.model_selection import TimeSeriesSplit

    model = MODELS[model_name]
    tscv = TimeSeriesSplit(n_splits=5)

    cv_scores = cross_val_score(
        model, X, y, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    mae_scores = -cv_scores

    return {
        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(),
        "scores": mae_scores.tolist(),
    }


def save_model(model, path: str):
    """Save model to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(path: str):
    """Load model from disk."""
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(
        description="Train parking demand prediction models"
    )
    parser.add_argument(
        "--data", default="data/parking.csv", help="Path to training data CSV"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to train: linear, rf, xgb, or all",
    )
    parser.add_argument(
        "--output-dir", default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument("--cv", action="store_true", help="Run cross-validation")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(df)
    print(f"Features: {X.shape[1]}, Samples: {len(X)}")

    # Time-series split
    X_train, X_test, y_train, y_test = train_test_split_time_series(
        X,
        y,
        test_size=args.test_size,
        gap=24,  # 1-day gap to prevent leakage
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Determine which models to train
    if "all" in args.models:
        model_names = list(MODELS.keys())
    else:
        model_names = args.models

    # Train and evaluate each model
    results = {}
    trained_models = {}

    for name in model_names:
        print(f"\n{'=' * 50}")
        print(f"Training {name.upper()}")
        print(f"{'=' * 50}")

        model = train_model(name, X_train, y_train)
        trained_models[name] = model

        # Test set evaluation
        metrics = evaluate_model_on_split(model, X_test, y_test)
        results[name] = metrics
        print(f"Test MAE: {metrics['mae']:.2f}")
        print(f"Test RMSE: {metrics['rmse']:.2f}")
        print(f"Test R²: {metrics['r2']:.4f}")

        # Cross-validation
        if args.cv:
            print("Running cross-validation...")
            cv_results = cross_validate_model(name, X, y)
            print(f"CV MAE: {cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f}")
            results[name]["cv"] = cv_results

        # Save model
        save_model(model, f"{args.output_dir}/{name}_model.pkl")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 60)
    for name, metrics in results.items():
        print(
            f"{name:<12} {metrics['mae']:>8.2f} {metrics['rmse']:>8.2f} {metrics['r2']:>8.4f}"
        )

    # Save best model separately
    best_model_name = min(results.keys(), key=lambda k: results[k]["mae"])
    best_model = trained_models[best_model_name]
    save_model(best_model, f"{args.output_dir}/best_model.pkl")
    print(
        f"\nBest model: {best_model_name} (MAE: {results[best_model_name]['mae']:.2f})"
    )


if __name__ == "__main__":
    main()
