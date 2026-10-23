import os
import sys
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "sol_features.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "ai",
    "sol_model.joblib"
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "backtest",
    "backtest_results.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 70)
    print("LOADING AI MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    print(f"Model loaded: {MODEL_FILE}")
    print(f"Model type: {type(model).__name__}")

    # Get features stored inside the model
    model_features = getattr(model, "feature_names_in_", None)

    if model_features is None:

        print("\nWARNING:")
        print("Model does not contain feature_names_in_")

    else:

        model_features = list(model_features)

        print(
            f"\nUsing features stored inside model:"
        )

        print(
            f"Number of features: {len(model_features)}"
        )

        print("Features:")

        for feature in model_features:
            print(f"  - {feature}")

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset loaded: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    original_columns = list(df.columns)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("\nColumn normalization:")

    for old, new in zip(original_columns, df.columns):

        if old != new:
            print(f"  {old} -> {new}")

    # --------------------------------------------------------
    # Datetime
    # --------------------------------------------------------

    if "datetime" in df.columns:

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

    print("\nAvailable columns:")

    print(list(df.columns))

    return df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(model, df):

    print("\n" + "=" * 70)
    print("CHECKING MODEL FEATURES")
    print("=" * 70)

    model_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    # --------------------------------------------------------
    # If model contains feature names
    # --------------------------------------------------------

    if model_features is not None:

        model_features = [
            str(x).strip().lower()
            for x in model_features
        ]

    else:

        raise ValueError(
            "Model does not contain feature_names_in_. "
            "Cannot safely determine required features."
        )

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:

        print("\nERROR: Dataset is missing model features:")

        for feature in missing_features:
            print(f"  - {feature}")

        print("\nAvailable dataset columns:")

        for column in df.columns:
            print(f"  - {column}")

        raise ValueError(
            "Dataset is missing model features."
        )

    # --------------------------------------------------------
    # Feature order
    # --------------------------------------------------------

    print("\nAll model features found successfully!")

    print(
        f"Total model features: {len(model_features)}"
    )

    return model_features


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, model_features):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    X = df[model_features].copy()

    # --------------------------------------------------------
    # Convert values to numeric
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Forward fill
    # --------------------------------------------------------

    X = X.ffill()

    # --------------------------------------------------------
    # Backward fill
    # --------------------------------------------------------

    X = X.bfill()

    # --------------------------------------------------------
    # Final NaN protection
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Feature matrix still contains NaN or infinity."
        )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print(
        f"Features used: {len(X.columns)}"
    )

    return X


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

def generate_predictions(model, X):

    print("\n" + "=" * 70)
    print("GENERATING AI PREDICTIONS")
    print("=" * 70)

    predictions = model.predict(X)

    print(
        f"Predictions generated: {len(predictions)}"
    )

    unique, counts = np.unique(
        predictions,
        return_counts=True
    )

    print("\nPrediction distribution:")

    for value, count in zip(unique, counts):

        percentage = (
            count / len(predictions)
        ) * 100

        print(
            f"  {value:>3}: "
            f"{count:>6} "
            f"({percentage:.2f}%)"
        )

    return predictions


# ============================================================
# CREATE BACKTEST RESULTS
# ============================================================

def create_results(df, predictions):

    print("\n" + "=" * 70)
    print("CREATING BACKTEST RESULTS")
    print("=" * 70)

    results = df.copy()

    results["prediction"] = predictions

    # --------------------------------------------------------
    # Actual target if available
    # --------------------------------------------------------

    if "target" in results.columns:

        results["correct"] = (
            results["prediction"]
            == results["target"]
        ).astype(int)

    # --------------------------------------------------------
    # Future return if available
    # --------------------------------------------------------

    if "future_return_6h" in results.columns:

        results["strategy_return"] = (
            results["prediction"]
            * results["future_return_6h"]
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    print(
        f"Results saved to:\n{RESULT_FILE}"
    )

    return results


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(results):

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    # --------------------------------------------------------
    # Prediction counts
    # --------------------------------------------------------

    if "prediction" in results.columns:

        print("\nPREDICTIONS:")

        print(
            results["prediction"]
            .value_counts()
            .sort_index()
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    if "correct" in results.columns:

        accuracy = (
            results["correct"].mean()
            * 100
        )

        print(
            f"\nPrediction Accuracy: "
            f"{accuracy:.2f}%"
        )

    # --------------------------------------------------------
    # Strategy return
    # --------------------------------------------------------

    if "strategy_return" in results.columns:

        strategy_returns = (
            results["strategy_return"]
            .dropna()
        )

        if len(strategy_returns) > 0:

            total_return = (
                strategy_returns.sum()
            )

            average_return = (
                strategy_returns.mean()
            )

            winning = (
                strategy_returns > 0
            ).sum()

            losing = (
                strategy_returns < 0
            ).sum()

            print("\nSTRATEGY:")

            print(
                f"Total Return     : "
                f"{total_return:.6f}"
            )

            print(
                f"Average Return   : "
                f"{average_return:.6f}"
            )

            print(
                f"Winning Trades   : "
                f"{winning}"
            )

            print(
                f"Losing Trades    : "
                f"{losing}"
            )

            if losing > 0:

                win_rate = (
                    winning
                    / (winning + losing)
                    * 100
                )

                print(
                    f"Win Rate         : "
                    f"{win_rate:.2f}%"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SOLANA AI BACKTEST ENGINE")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = load_model()

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_data()

        # ----------------------------------------------------
        # Get model features
        # ----------------------------------------------------

        model_features = get_model_features(
            model,
            df
        )

        # ----------------------------------------------------
        # Prepare feature matrix
        # ----------------------------------------------------

        X = prepare_features(
            df,
            model_features
        )

        # ----------------------------------------------------
        # Generate predictions
        # ----------------------------------------------------

        predictions = generate_predictions(
            model,
            X
        )

        # ----------------------------------------------------
        # Create results
        # ----------------------------------------------------

        results = create_results(
            df,
            predictions
        )

        # ----------------------------------------------------
        # Analyze
        # ----------------------------------------------------

        analyze_results(results)

        print("\n" + "=" * 70)
        print("BACKTEST ENGINE COMPLETE 🚀")
        print("=" * 70)

    except Exception as e:

        print("\n" + "=" * 70)
        print("BACKTEST ERROR")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        print("\nPlease fix the error above.")

        sys.exit(1)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()