
"""
STMaCS SYSTEM 455
BASELINE MACHINE-LEARNING MODEL DEVELOPMENT

Purpose
-------
Train and evaluate baseline predictive models for future Buffer Top
temperature at 5, 15, 30, and 60-minute horizons.

The machine-learning model is a predictive component of the STMaCS
predictive-heating system. The immediate ML task is to predict future
storage-tank temperature. These predictions will later be supplied to
a separate predictive-control layer that determines whether future
hot-water demand will be satisfied and whether backup heating is
necessary.

Models
------
1. Persistence baseline
2. Ridge Regression
3. Random Forest
4. HistGradientBoosting

Evaluation
----------
- MAE
- RMSE
- R²
- Improvement over persistence
- Mean error
- Maximum absolute error
- Prediction standard deviation

Methodology
-----------
- Uses the leakage-controlled chronological datasets created by
  build_ml_datasets.py.
- No random shuffling.
- No additional train/test splitting.
- No future target leakage.
- No additional imputation.
- Ambient temperature remains excluded.
- No synthetic weather data is introduced.
- Persistence is retained as the physical benchmark.

Future role
-----------
The best-performing predictive model will later be integrated into
the predictive-control layer. The controller will use predicted
future temperature, expected demand, required temperature thresholds,
and eventually weather/solar forecasts to determine whether backup
heating is required and how long it should operate.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

ML_DIR = BASE_DIR / "data" / "processed" / "ml"

RESULTS_DIR = ML_DIR / "baseline_results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [5, 15, 30, 60]

RANDOM_STATE = 42


# =============================================================================
# DISPLAY
# =============================================================================

print("=" * 80)
print("STMaCS SYSTEM 455 BASELINE MACHINE-LEARNING MODELS")
print("=" * 80)

print()
print("Purpose:")
print("  Predict future Buffer Top temperature as the predictive")
print("  intelligence component of the STMaCS predictive-heating system.")

print()
print("Prediction horizons:")
print("  - 5 minutes")
print("  - 15 minutes")
print("  - 30 minutes")
print("  - 60 minutes")

print()
print("Models:")
print("  - Persistence baseline")
print("  - Ridge Regression")
print("  - Random Forest")
print("  - HistGradientBoosting")

print()
print("Evaluation:")
print("  - MAE")
print("  - RMSE")
print("  - R²")
print("  - Improvement over persistence")
print("  - Mean error")
print("  - Maximum absolute error")
print("  - Prediction standard deviation")

print()
print("Methodology:")
print("  - Chronological train/test datasets")
print("  - No random shuffling")
print("  - No additional data splitting")
print("  - No future target leakage")
print("  - Ambient temperature excluded")
print("  - No synthetic weather data")
print("=" * 80)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def create_models():
    """
    Create baseline regression models.

    Ridge uses StandardScaler because its coefficients are sensitive
    to differences in feature scale.

    Tree-based models do not require feature scaling.
    """

    models = {

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(
                alpha=1.0
            ))
        ]),

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    }

    return models


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression performance metrics.
    """

    errors = y_true.to_numpy() - np.asarray(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    mean_error = np.mean(errors)

    max_absolute_error = np.max(
        np.abs(errors)
    )

    prediction_std = np.std(
        np.asarray(y_pred)
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_error": mean_error,
        "max_absolute_error": max_absolute_error,
        "prediction_std": prediction_std
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

all_results = []


for horizon in HORIZONS:

    print()
    print("=" * 80)
    print(f"PROCESSING {horizon}-MINUTE PREDICTION HORIZON")
    print("=" * 80)

    train_file = (
        ML_DIR /
        f"system455_{horizon}min_train.csv"
    )

    test_file = (
        ML_DIR /
        f"system455_{horizon}min_test.csv"
    )

    print()
    print("Loading:")
    print(f"  Train: {train_file}")
    print(f"  Test : {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print()
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows : {len(test_df):,}")

    # =========================================================================
    # TIMESTAMP VERIFICATION
    # =========================================================================

    train_df["timestamp"] = pd.to_datetime(
        train_df["timestamp"]
    )

    test_df["timestamp"] = pd.to_datetime(
        test_df["timestamp"]
    )

    if not train_df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Training timestamps are not chronological."
        )

    if not test_df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Testing timestamps are not chronological."
        )

    if train_df["timestamp"].max() >= test_df["timestamp"].min():
        raise ValueError(
            "Training and testing periods overlap."
        )

    print("[OK] Training timestamps are chronological.")
    print("[OK] Testing timestamps are chronological.")
    print("[OK] Training and testing periods do not overlap.")

    # =========================================================================
    # TARGET
    # =========================================================================

    target_column = (
        f"target_buffer_top_{horizon}min"
    )

    if target_column not in train_df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    if target_column not in test_df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    # =========================================================================
    # PREDICTORS
    # =========================================================================

    predictor_columns = [
        column
        for column in train_df.columns
        if column not in [
            "timestamp",
            target_column
        ]
        and not column.startswith("target_")
    ]

    print()
    print(f"Predictor features: {len(predictor_columns)}")

    missing_in_test = [
        column
        for column in predictor_columns
        if column not in test_df.columns
    ]

    if missing_in_test:
        raise ValueError(
            "Predictors missing from test dataset:\n"
            + "\n".join(missing_in_test)
        )

    # =========================================================================
    # EXTRACT DATA
    # =========================================================================

    X_train = train_df[predictor_columns].copy()
    X_test = test_df[predictor_columns].copy()

    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    # =========================================================================
    # FINAL DATA QUALITY CHECK
    # =========================================================================

    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()

    if train_missing > 0:
        raise ValueError(
            f"Training predictors contain {train_missing} missing values."
        )

    if test_missing > 0:
        raise ValueError(
            f"Testing predictors contain {test_missing} missing values."
        )

    if y_train.isna().sum() > 0:
        raise ValueError(
            "Training target contains missing values."
        )

    if y_test.isna().sum() > 0:
        raise ValueError(
            "Testing target contains missing values."
        )

    print("[OK] Predictor completeness verified.")
    print("[OK] Target completeness verified.")

    # =========================================================================
    # PERSISTENCE BASELINE
    # =========================================================================

    print()
    print("-" * 80)
    print("PERSISTENCE BASELINE")
    print("-" * 80)

    if "buffer_top" not in test_df.columns:
        raise ValueError(
            "buffer_top is required for persistence baseline."
        )

    persistence_prediction = (
        test_df["buffer_top"].to_numpy()
    )

    persistence_metrics = calculate_metrics(
        y_test,
        persistence_prediction
    )

    print()
    print(f"MAE :  {persistence_metrics['mae']:.6f} °C")
    print(f"RMSE:  {persistence_metrics['rmse']:.6f} °C")
    print(f"R²  :  {persistence_metrics['r2']:.6f}")

    all_results.append({

        "horizon_min": horizon,
        "model": "Persistence",

        "mae": persistence_metrics["mae"],
        "rmse": persistence_metrics["rmse"],
        "r2": persistence_metrics["r2"],

        "mean_error": persistence_metrics["mean_error"],
        "max_absolute_error":
            persistence_metrics["max_absolute_error"],
        "prediction_std":
            persistence_metrics["prediction_std"],

        "mae_improvement_vs_persistence_pct": 0.0,
        "rmse_improvement_vs_persistence_pct": 0.0,

        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(predictor_columns)
    })

    persistence_output = pd.DataFrame({

        "timestamp": test_df["timestamp"],

        "actual": y_test.to_numpy(),

        "predicted":
            persistence_prediction,

        "error":
            y_test.to_numpy() - persistence_prediction,

        "absolute_error":
            np.abs(
                y_test.to_numpy() -
                persistence_prediction
            )
    })

    persistence_file = (
        PREDICTIONS_DIR /
        f"system455_{horizon}min_persistence_predictions.csv"
    )

    persistence_output.to_csv(
        persistence_file,
        index=False
    )

    print()
    print(f"[SAVED] {persistence_file}")

    # =========================================================================
    # MACHINE-LEARNING MODELS
    # =========================================================================

    models = create_models()

    for model_name, model in models.items():

        print()
        print("-" * 80)
        print(f"TRAINING: {model_name}")
        print("-" * 80)

        print("Training model...")

        model.fit(
            X_train,
            y_train
        )

        print("[OK] Training complete.")

        # ---------------------------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # ---------------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            predictions
        )

        mae = metrics["mae"]
        rmse = metrics["rmse"]
        r2 = metrics["r2"]

        # ---------------------------------------------------------------------
        # IMPROVEMENT OVER PERSISTENCE
        # ---------------------------------------------------------------------

        mae_improvement = (
            (
                persistence_metrics["mae"] -
                mae
            )
            /
            persistence_metrics["mae"]
            * 100
            if persistence_metrics["mae"] != 0
            else np.nan
        )

        rmse_improvement = (
            (
                persistence_metrics["rmse"] -
                rmse
            )
            /
            persistence_metrics["rmse"]
            * 100
            if persistence_metrics["rmse"] != 0
            else np.nan
        )

        # ---------------------------------------------------------------------
        # DISPLAY
        # ---------------------------------------------------------------------

        print()
        print("Performance:")
        print(f"  MAE :  {mae:.6f} °C")
        print(f"  RMSE:  {rmse:.6f} °C")
        print(f"  R²  :  {r2:.6f}")

        print()
        print("Prediction error:")
        print(
            f"  Mean error            : "
            f"{metrics['mean_error']:+.6f} °C"
        )

        print(
            f"  Maximum absolute error: "
            f"{metrics['max_absolute_error']:.6f} °C"
        )

        print(
            f"  Prediction std        : "
            f"{metrics['prediction_std']:.6f} °C"
        )

        print()
        print("Improvement over persistence:")
        print(
            f"  MAE improvement : "
            f"{mae_improvement:+.2f}%"
        )

        print(
            f"  RMSE improvement: "
            f"{rmse_improvement:+.2f}%"
        )

        # ---------------------------------------------------------------------
        # STORE RESULTS
        # ---------------------------------------------------------------------

        all_results.append({

            "horizon_min": horizon,
            "model": model_name,

            "mae": mae,
            "rmse": rmse,
            "r2": r2,

            "mean_error":
                metrics["mean_error"],

            "max_absolute_error":
                metrics["max_absolute_error"],

            "prediction_std":
                metrics["prediction_std"],

            "mae_improvement_vs_persistence_pct":
                mae_improvement,

            "rmse_improvement_vs_persistence_pct":
                rmse_improvement,

            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(predictor_columns)
        })

        # ---------------------------------------------------------------------
        # SAVE PREDICTIONS
        # ---------------------------------------------------------------------

        prediction_output = pd.DataFrame({

            "timestamp":
                test_df["timestamp"],

            "actual":
                y_test.to_numpy(),

            "predicted":
                predictions,

            "error":
                y_test.to_numpy() -
                predictions,

            "absolute_error":
                np.abs(
                    y_test.to_numpy() -
                    predictions
                )
        })

        prediction_file = (
            PREDICTIONS_DIR /
            f"system455_{horizon}min_"
            f"{model_name.lower()}_predictions.csv"
        )

        prediction_output.to_csv(
            prediction_file,
            index=False
        )

        print()
        print(f"[SAVED] {prediction_file}")


# =============================================================================
# SAVE MASTER RESULTS
# =============================================================================

print()
print("=" * 80)
print("SAVING MODEL PERFORMANCE RESULTS")
print("=" * 80)

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    by=[
        "horizon_min",
        "mae"
    ]
)

results_file = (
    RESULTS_DIR /
    "baseline_model_performance.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print()
print(f"[SAVED] {results_file}")


# =============================================================================
# BEST MODEL PER HORIZON
# =============================================================================

print()
print("=" * 80)
print("BEST MODEL BY PREDICTION HORIZON")
print("=" * 80)

best_models = []

for horizon in HORIZONS:

    horizon_results = results_df[
        results_df["horizon_min"] == horizon
    ].copy()

    best = horizon_results.loc[
        horizon_results["mae"].idxmin()
    ]

    best_models.append(best)

    print()
    print(f"{horizon}-minute horizon:")
    print(f"  Best model: {best['model']}")
    print(f"  MAE       : {best['mae']:.6f} °C")
    print(f"  RMSE      : {best['rmse']:.6f} °C")
    print(f"  R²        : {best['r2']:.6f}")

    if pd.notna(
        best.get(
            "mae_improvement_vs_persistence_pct",
            np.nan
        )
    ):

        print(
            f"  Improvement over persistence: "
            f"{best['mae_improvement_vs_persistence_pct']:+.2f}%"
        )


best_models_df = pd.DataFrame(
    best_models
)

best_models_file = (
    RESULTS_DIR /
    "best_model_by_horizon.csv"
)

best_models_df.to_csv(
    best_models_file,
    index=False
)

print()
print(f"[SAVED] {best_models_file}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("BASELINE MODEL DEVELOPMENT COMPLETE")
print("=" * 80)

print()
print("Results directory:")
print(f"  {RESULTS_DIR}")

print()
print("Files created:")
print("  - baseline_model_performance.csv")
print("  - best_model_by_horizon.csv")

print()
print("Prediction files:")
print("  - Persistence predictions")
print("  - Ridge predictions")
print("  - Random Forest predictions")
print("  - HistGradientBoosting predictions")

print()
print("Important:")
print("  - Real System 455 data was used.")
print("  - Chronological train/test separation was preserved.")
print("  - No random shuffling was performed.")
print("  - No additional train/test split was performed.")
print("  - Persistence was used as the physical benchmark.")
print("  - Ambient temperature remained excluded.")
print("  - No synthetic weather data was introduced.")
print("  - No future target columns were used as predictors.")
print("  - Predictions were evaluated on unseen chronological data.")

print()
print("NEXT STAGE:")
print("  Use the baseline results to determine which predictive model")
print("  and prediction horizon are most suitable for the STMaCS")
print("  predictive-heating control layer.")

print()
print("=" * 80)
print("READY FOR MODEL COMPARISON AND INTERPRETATION")
print("=" * 80)

