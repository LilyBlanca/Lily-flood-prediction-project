import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Project settings
# =========================

DATA_PATH = "data/raw/train.csv"
MODEL_PATH = "models/flood_model.pkl"
MODEL_COMPARISON_PATH = "reports/model_comparison.csv"
ERROR_ANALYSIS_PATH = "reports/error_analysis.csv"

TARGET = "FloodProbability"
RANDOM_STATE = 42


# =========================
# Helper functions
# =========================

def evaluate_model(y_true, y_pred):
    """
    Calculate regression metrics.
    MAE: average absolute error
    RMSE: square root of average squared error
    R2: how much variance the model explains
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def load_data(data_path):
    """
    Load training data from CSV.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Cannot find data file: {data_path}\n"
            "Please make sure train.csv is inside data/raw/"
        )

    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in the dataset.\n"
            f"Available columns are: {list(df.columns)}"
        )

    return df


def make_models():
    """
    Define models for comparison.
    """
    models = {
        "Dummy Baseline": DummyRegressor(strategy="mean"),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models


def explain_bias_variance(train_rmse, val_rmse):
    """
    Simple beginner-friendly bias/variance diagnosis.
    This is not perfect science, but it helps you understand model behavior.
    """
    gap = val_rmse - train_rmse

    if train_rmse > 0.05 and val_rmse > 0.05:
        return "Possible underfitting / high bias"
    elif gap > 0.02:
        return "Possible overfitting / high variance"
    else:
        return "Reasonable fit"


# =========================
# Main training process
# =========================

def main():
    print("Loading data...")

    df = load_data(DATA_PATH)

    print(f"Data shape: {df.shape}")
    print(f"Target column: {TARGET}")

    # Drop id column if it exists.
    # Many Kaggle datasets have an id column.
    # Usually, id is not useful for prediction.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Split features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    print("\nTraining models...")

    models = make_models()

    results = []
    best_model = None
    best_model_name = None
    best_val_rmse = float("inf")

    # Train each model
    for name, model in models.items():
        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Evaluation
        train_metrics = evaluate_model(y_train, train_pred)
        val_metrics = evaluate_model(y_val, val_pred)

        diagnosis = explain_bias_variance(
            train_metrics["RMSE"],
            val_metrics["RMSE"]
        )

        results.append({
            "Model": name,

            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],

            "Val_MAE": val_metrics["MAE"],
            "Val_RMSE": val_metrics["RMSE"],
            "Val_R2": val_metrics["R2"],

            "Diagnosis": diagnosis
        })

        print(f"Train RMSE: {train_metrics['RMSE']:.6f}")
        print(f"Val RMSE:   {val_metrics['RMSE']:.6f}")
        print(f"Val R2:     {val_metrics['R2']:.6f}")
        print(f"Diagnosis:  {diagnosis}")

        # Choose best model using validation RMSE
        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_model = model
            best_model_name = name

    # Create reports folder
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save model comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Val_RMSE", ascending=True)

    print("\n==============================")
    print("Model Comparison")
    print("==============================")
    print(results_df)

    results_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    # Save best model
    joblib.dump(best_model, MODEL_PATH)

    print("\n==============================")
    print("Best Model")
    print("==============================")
    print(f"Best model: {best_model_name}")
    print(f"Best validation RMSE: {best_val_rmse:.6f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model comparison saved to: {MODEL_COMPARISON_PATH}")

    # Error analysis for best model
    print("\nCreating error analysis table...")

    best_val_pred = best_model.predict(X_val)

    error_df = X_val.copy()
    error_df["Actual_FloodProbability"] = y_val.values
    error_df["Predicted_FloodProbability"] = best_val_pred
    error_df["Error"] = (
        error_df["Predicted_FloodProbability"]
        - error_df["Actual_FloodProbability"]
    )
    error_df["Absolute_Error"] = error_df["Error"].abs()

    # Sort by biggest errors first
    error_df = error_df.sort_values("Absolute_Error", ascending=False)

    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)

    print(f"Error analysis saved to: {ERROR_ANALYSIS_PATH}")

    print("\nTraining finished successfully!")


if __name__ == "__main__":
    main()