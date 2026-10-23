import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/sol_features.csv"
MODEL_FILE = "ai/sol_model.joblib"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",

    "return_1h",
    "return_3h",
    "return_6h",
    "return_24h",

    "ema_9",
    "ema_21",
    "ema_50",

    "rsi_14",

    "macd",
    "macd_signal",

    "bb_high",
    "bb_low",
    "bb_middle",
    "bb_width",

    "atr_14",

    "volume_ma_20",
    "volume_change",
    "volume_ratio",

    "volatility_24h",

    "price_vs_ema9",
    "price_vs_ema21",
    "price_vs_ema50",
]

TARGET = "target"


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("SOLANA AI MODEL TRAINING")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

try:
    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:
    print("\nERROR: Dataset not found:")
    print(DATA_FILE)
    raise SystemExit


print(f"Dataset shape: {df.shape}")


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

# Remove hidden spaces
# Convert column names to lowercase

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


print("\nCleaned columns:")
print(df.columns.tolist())


# ============================================================
# DATETIME
# ============================================================

if "datetime" in df.columns:

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)


# ============================================================
# CHECK TARGET
# ============================================================

if TARGET not in df.columns:

    print("\nERROR: Target column not found:")
    print(TARGET)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise SystemExit


# ============================================================
# REMOVE MISSING TARGET
# ============================================================

df = df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    raise SystemExit


# ============================================================
# CREATE X AND Y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

for column in FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# CLEAN INF / NAN
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


valid_rows = X.notna().all(
    axis=1
)


X = X.loc[
    valid_rows
].copy()


y = y.loc[
    valid_rows
].copy()


# ============================================================
# DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print(
    f"Training samples: {len(X)}"
)

print(
    f"Number of features: {len(FEATURES)}"
)


print("\nTarget distribution:")

print(
    y.value_counts()
    .sort_index()
)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)


split_index = int(
    len(X) * 0.80
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]


y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("CREATING RANDOM FOREST MODEL")
print("=" * 60)


model = RandomForestClassifier(

    n_estimators=400,

    max_depth=10,

    min_samples_leaf=5,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")
print("Please wait...")


model.fit(
    X_train,
    y_train
)


print("\nTraining completed!")


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")


predictions = model.predict(
    X_test
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)


print(
    f"Accuracy: {accuracy:.4f}"
)


print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")


print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": FEATURES,

    "importance": model.feature_importances_

})


importance = importance.sort_values(

    "importance",

    ascending=False
)


print("\n" + "=" * 60)
print("TOP 15 FEATURES")
print("=" * 60)


print(
    importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)


joblib.dump(
    model,
    MODEL_FILE
)


print(
    f"Model saved to: {MODEL_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE 🚀")
print("=" * 60)