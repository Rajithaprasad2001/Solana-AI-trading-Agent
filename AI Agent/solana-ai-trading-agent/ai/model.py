# ============================================================
# SOLANA AI TRADING AGENT
# MACHINE LEARNING MODEL TRAINING
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/sol_features.csv"
MODEL_FILE = "ai/sol_model.joblib"

TARGET_COLUMN = "target"

# IMPORTANT:
# future_return_6h is used to create the target.
# It MUST NOT be given to the model as an input feature.
LEAKAGE_COLUMNS = [
    "future_return_6h",
    "target",
]

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 60)
    print("LOADING SOLANA DATA")
    print("=" * 60)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}\n"
        )

    df = pd.read_csv(DATA_FILE)

    print("\nDataset loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)

    # Replace positive/negative infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Count missing values before cleaning
    nan_before = df.isna().sum().sum()

    print("\nNaN values before cleaning:", nan_before)

    # Remove rows with missing target
    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    # Remove rows containing NaN
    df = df.dropna()

    # Reset index
    df = df.reset_index(drop=True)

    # Final check
    nan_after = df.isna().sum().sum()

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    inf_after = np.isinf(
        numeric_df
    ).sum().sum()

    print("NaN values after cleaning:", nan_after)
    print("Infinity values after cleaning:", inf_after)

    print("\nClean dataset shape:")
    print(df.shape)

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    # Target
    y = df[TARGET_COLUMN].astype(int)

    # Remove target and future information
    X = df.drop(
        columns=[
            column
            for column in LEAKAGE_COLUMNS
            if column in df.columns
        ],
        errors="ignore"
    )

    # Remove datetime column
    if "Datetime" in X.columns:
        X = X.drop(
            columns=["Datetime"]
        )

    if "DateTime" in X.columns:
        X = X.drop(
            columns=["DateTime"]
        )

    if "timestamp" in X.columns:
        X = X.drop(
            columns=["timestamp"]
        )

    if "Timestamp" in X.columns:
        X = X.drop(
            columns=["Timestamp"]
        )

    # Keep only numeric columns
    X = X.select_dtypes(
        include=[np.number]
    )

    # Final infinity cleanup
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill remaining missing values
    X = X.fillna(0)

    # Convert to float32
    X = X.astype(np.float32)

    # Final safety check
    X = X.replace(
        [np.inf, -np.inf],
        0
    )

    X = X.fillna(0)

    print("\nFeatures selected:")
    print("-" * 60)

    for i, column in enumerate(X.columns, 1):
        print(f"{i:02d}. {column}")

    print("\nTotal features:", len(X.columns))

    print("\nTarget distribution:")

    print(
        y.value_counts()
        .sort_index()
    )

    print("\nTarget percentages:")

    print(
        (y.value_counts(normalize=True)
         .sort_index() * 100)
        .round(2)
    )

    return X, y


# ============================================================
# TIME BASED TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n" + "=" * 60)
    print("TIME-BASED TRAIN / TEST SPLIT")
    print("=" * 60)

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    print(
        "\nTraining period:"
    )

    print(
        f"Rows 0 -> {split_index - 1}"
    )

    print(
        "\nTesting period:"
    )

    print(
        f"Rows {split_index} -> {len(X) - 1}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# FINAL MODEL SAFETY CHECK
# ============================================================

def safety_check(
    X_train,
    X_test
):

    print("\n" + "=" * 60)
    print("MODEL INPUT SAFETY CHECK")
    print("=" * 60)

    # Replace infinity
    X_train = X_train.replace(
        [np.inf, -np.inf],
        0
    )

    X_test = X_test.replace(
        [np.inf, -np.inf],
        0
    )

    # Replace NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Convert to float32
    X_train = X_train.astype(
        np.float32
    )

    X_test = X_test.astype(
        np.float32
    )

    # Check
    train_nan = X_train.isna().sum().sum()
    test_nan = X_test.isna().sum().sum()

    train_inf = np.isinf(
        X_train.values
    ).sum()

    test_inf = np.isinf(
        X_test.values
    ).sum()

    print(
        "Training NaN:",
        train_nan
    )

    print(
        "Testing NaN:",
        test_nan
    )

    print(
        "Training Infinity:",
        train_inf
    )

    print(
        "Testing Infinity:",
        test_inf
    )

    if train_nan != 0:
        raise ValueError(
            "Training data still contains NaN."
        )

    if test_nan != 0:
        raise ValueError(
            "Testing data still contains NaN."
        )

    if train_inf != 0:
        raise ValueError(
            "Training data still contains Infinity."
        )

    if test_inf != 0:
        raise ValueError(
            "Testing data still contains Infinity."
        )

    print(
        "\n✓ Model input is clean."
    )

    return X_train, X_test


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining started...")

    model.fit(
        X_train,
        y_train
    )

    print("\n✓ Training completed.")

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return predictions


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    feature_names
):

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 60)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False
    )

    print(
        importance.head(20)
        .to_string(index=False)
    )

    return importance


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    feature_names
):

    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    # Store feature names inside model
    # This helps prediction.py later.
    model.feature_names_in_ = np.array(
        feature_names
    )

    os.makedirs(
        os.path.dirname(MODEL_FILE),
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("SOLANA AI TRADING AGENT")
    print("MACHINE LEARNING TRAINING")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Clean
    df = clean_data(df)

    # 3. Prepare features
    X, y = prepare_features(df)

    # 4. Split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # 5. Safety check
    (
        X_train,
        X_test
    ) = safety_check(
        X_train,
        X_test
    )

    # 6. Train
    model = train_model(
        X_train,
        y_train
    )

    # 7. Evaluate
    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    # 8. Feature importance
    importance = show_feature_importance(
        model,
        X.columns
    )

    # 9. Save
    save_model(
        model,
        X.columns
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nModel:"
    )

    print(
        MODEL_FILE
    )

    print(
        "\nFeatures:",
        len(X.columns)
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print(
        "\nNext step:"
    )

    print(
        "Connect the model to predictor.py"
    )

    print(
        "Then build strategy + risk management."
    )

    print(
        "\n" + "=" * 60
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()