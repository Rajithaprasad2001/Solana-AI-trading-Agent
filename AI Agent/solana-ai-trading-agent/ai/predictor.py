import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# SOLANA AI TRADING AGENT
# LIVE AI PREDICTOR
# ============================================================

MODEL_FILE = "ai/sol_model.joblib"
DATA_FILE = "data/live_sol_features.csv"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    print("Loading AI model...")

    model = joblib.load(MODEL_FILE)

    print("AI model loaded successfully.")

    return model


# ============================================================
# LOAD LIVE FEATURE DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Live feature dataset not found: {DATA_FILE}"
        )

    print("Loading live feature dataset...")

    df = pd.read_csv(DATA_FILE)

    if df.empty:

        raise ValueError(
            "Live feature dataset is empty."
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df, model):

    # --------------------------------------------------------
    # Get exact features used during training
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        feature_columns = list(
            model.feature_names_in_
        )

    else:

        excluded_columns = [
            "Datetime",
            "timestamp",
            "target",
            "future_return_6h"
        ]

        feature_columns = [
            column
            for column in df.columns
            if column not in excluded_columns
        ]

    print("\nFeatures used by AI model:")

    print(feature_columns)

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing live features: {missing_columns}"
        )

    # --------------------------------------------------------
    # Get latest market row
    # --------------------------------------------------------

    latest = df.iloc[-1:].copy()

    # --------------------------------------------------------
    # Create model input
    # --------------------------------------------------------

    X = latest[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove infinity
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    X = X.ffill()

    X = X.bfill()

    X = X.fillna(0)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if X.isnull().values.any():

        raise ValueError(
            "NaN values still exist in model input."
        )

    if np.isinf(X.values).any():

        raise ValueError(
            "Infinity values still exist in model input."
        )

    return X, latest


# ============================================================
# GET CURRENT PRICE
# ============================================================

def get_current_price(latest):

    possible_columns = [
        "Close",
        "close"
    ]

    for column in possible_columns:

        if column in latest.columns:

            return float(
                latest[column].iloc[0]
            )

    return None


# ============================================================
# GET TIMESTAMP
# ============================================================

def get_timestamp(latest):

    possible_columns = [
        "Datetime",
        "timestamp"
    ]

    for column in possible_columns:

        if column in latest.columns:

            return str(
                latest[column].iloc[0]
            )

    return "Unknown"


# ============================================================
# CONVERT CLASS TO SIGNAL
# ============================================================

def class_to_signal(prediction):

    if prediction == 1:

        return "BUY"

    elif prediction == 0:

        return "SELL"

    else:

        return "HOLD"


# ============================================================
# PREDICT
# ============================================================

def predict():

    print("\n")
    print("=" * 60)
    print("SOLANA AI LIVE PREDICTOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load LIVE features
    # --------------------------------------------------------

    df = load_data()

    print(
        f"Live feature rows : {len(df)}"
    )

    # --------------------------------------------------------
    # Prepare latest features
    # --------------------------------------------------------

    X, latest = prepare_features(
        df,
        model
    )

    # --------------------------------------------------------
    # AI prediction
    # --------------------------------------------------------

    prediction = model.predict(X)[0]

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

    # --------------------------------------------------------
    # Convert prediction to signal
    # --------------------------------------------------------

    signal = class_to_signal(
        prediction
    )

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    current_price = get_current_price(
        latest
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = get_timestamp(
        latest
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 60)
    print("AI LIVE PREDICTION")
    print("=" * 60)

    print(
        f"\nTime        : {timestamp}"
    )

    if current_price is not None:

        print(
            f"SOL Price   : ${current_price:.4f}"
        )

    print(
        f"Prediction  : {prediction}"
    )

    print(
        f"Signal      : {signal}"
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if probabilities is not None:

        print("\nPrediction probabilities:")

        classes = model.classes_

        for cls, probability in zip(
            classes,
            probabilities
        ):

            label = class_to_signal(
                cls
            )

            print(
                f"{label:<10}: "
                f"{probability * 100:.2f}%"
            )

        confidence = float(
            np.max(probabilities)
        )

        print(
            f"\nConfidence  : "
            f"{confidence * 100:.2f}%"
        )

    print("\n" + "=" * 60)

    return {

        "signal": signal,

        "prediction": int(
            prediction
        ),

        "price": current_price,

        "timestamp": timestamp,

        "probabilities": probabilities,

        "confidence": (
            float(np.max(probabilities))
            if probabilities is not None
            else None
        )
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        result = predict()

    except Exception as e:

        print("\nERROR:")
        print(str(e))

        raise