"""Evaluation and visualization for trained models."""

import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import prepare_features, train_test_split_time_series


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.Series = None,
    model_name: str = "Model",
    output_path: str = None,
):
    """Plot predictions vs actuals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Predictions vs Actuals scatter
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction"
    )
    ax.set_xlabel("Actual Occupancy")
    ax.set_ylabel("Predicted Occupancy")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Time series (if timestamps provided)
    ax = axes[0, 1]
    if timestamps is not None:
        n_points = min(len(y_true), 168)  # Last week
        ax.plot(
            timestamps[-n_points:],
            y_true[-n_points:],
            label="Actual",
            alpha=0.7,
            linewidth=1,
        )
        ax.plot(
            timestamps[-n_points:],
            y_pred[-n_points:],
            label="Predicted",
            alpha=0.7,
            linewidth=1,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Occupancy")
        ax.set_title(f"{model_name}: Time Series (Last Week)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax.text(
            0.5,
            0.5,
            "No timestamps provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Time Series (no timestamps)")

    # 3. Residuals
    ax = axes[1, 0]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{model_name}: Residuals vs Predicted")
    ax.grid(True, alpha=0.3)

    # 4. Residual distribution
    ax = axes[1, 1]
    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{model_name}: Residual Distribution")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list,
    model_name: str,
    output_path: str = None,
    top_n: int = 20,
):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} doesn't have feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Feature Importance")
    plt.title(f"{model_name}: Top {top_n} Feature Importances")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance plot to {output_path}")
    plt.close()


def plot_model_comparison(results: dict, output_path: str = None):
    """Plot comparison of multiple models."""
    models = list(results.keys())
    mae = [results[m]["mae"] for m in models]
    rmse = [results[m]["rmse"] for m in models]
    r2 = [results[m]["r2"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, metric, values, title, ylabel in [
        (axes[0], "MAE", mae, "Mean Absolute Error", "MAE (lower is better)"),
        (axes[1], "RMSE", rmse, "Root Mean Squared Error", "RMSE (lower is better)"),
        (axes[2], "R²", r2, "R² Score", "R² (higher is better)"),
    ]:
        bars = ax.bar(
            models,
            values,
            color=colors[: len(models)],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", required=True, help="Path to model pickle")
    parser.add_argument(
        "--data", default="data/parking.csv", help="Path to test data CSV"
    )
    parser.add_argument(
        "--output-dir", default="eval_output", help="Directory for output plots"
    )
    parser.add_argument(
        "--compare", nargs="+", help="Paths to multiple models for comparison"
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Prepare features
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split_time_series(
        X, y, test_size=0.2, gap=24
    )

    if args.compare:
        # Compare multiple models
        print("Comparing models...")
        results = {}
        for model_path in args.compare:
            model_name = Path(model_path).stem.replace("_model", "")
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test.values, model.predict(X_test))
            results[Path(model_path).stem] = metrics

            # Plot individual
            plot_predictions(
                y_test.values,
                y_pred,
                X_test.index if hasattr(X_test, "index") else None,
                model_name=model_name,
                output_path=f"{args.output_dir}/{Path(model_path).stem}_eval.png",
            )

            # Feature importance
            plot_feature_importance(
                model,
                X.columns.tolist(),
                Path(model_path).stem,
                output_path=f"{args.output_dir}/{Path(model_path).stem}_importance.png",
            )

        # Comparison plot
        plot_model_comparison(
            results, output_path=f"{args.output_dir}/model_comparison.png"
        )

        # Print summary
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        print("-" * 60)
        for name, metrics in results.items():
            print(
                f"{name:<20} {metrics['mae']:>8.2f} {metrics['rmse']:>8.2f} {metrics['r2']:>8.4f}"
            )

    else:
        # Single model evaluation
        print(f"Loading model from {args.model}...")
        model = joblib.load(args.model)
        model_name = Path(args.model).stem

        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test.values, y_pred)

        print(f"\n{model_name} Evaluation:")
        print(f"  MAE:  {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")

        # Plots
        plot_predictions(
            y_test.values,
            y_pred,
            timestamps=df.loc[X_test.index, "timestamp"]
            if "timestamp" in df.columns
            else None,
            model_name=model_name,
            output_path=f"{args.output_dir}/{model_name}_eval.png",
        )

        plot_feature_importance(
            model,
            X.columns.tolist(),
            model_name,
            output_path=f"{args.output_dir}/{model_name}_importance.png",
        )


if __name__ == "__main__":
    main()
