import os
import joblib
import pandas as pd


MODEL_PATH = "models/flood_model.pkl"
TEST_PATH = "data/raw/test.csv"
OUTPUT_PATH = "reports/predictions.csv"


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find model file: {MODEL_PATH}\n"
            "Please run python src/train.py first."
        )

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Cannot find test file: {TEST_PATH}\n"
            "Please make sure test.csv is inside data/raw/"
        )

    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Keep id column for output, but do not use it as a feature
    if "id" in test_df.columns:
        ids = test_df["id"]
        X_test = test_df.drop(columns=["id"])
    else:
        ids = None
        X_test = test_df

    predictions = model.predict(X_test)

    if ids is not None:
        output = pd.DataFrame({
            "id": ids,
            "FloodProbability": predictions
        })
    else:
        output = pd.DataFrame({
            "FloodProbability": predictions
        })

    os.makedirs("reports", exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()