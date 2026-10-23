import pandas as pd
import numpy as np


# ==========================================
# FILE PATHS
# ==========================================

input_file = "data/sol_usd_1h.csv"
output_file = "data/sol_features.csv"


# ==========================================
# LOAD DATA
# ==========================================

print("Loading SOL market data...")

df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape}")


# ==========================================
# CLEAN COLUMN NAMES
# ==========================================

df.columns = [column.strip() for column in df.columns]


# ==========================================
# DATETIME
# ==========================================

df["Datetime"] = pd.to_datetime(
    df["Datetime"],
    errors="coerce"
)

df = df.sort_values(
    "Datetime"
).reset_index(drop=True)


# ==========================================
# MAKE SURE NUMERIC COLUMNS ARE NUMERIC
# ==========================================

numeric_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================
# BASIC RETURNS
# ==========================================

print("Creating return features...")


df["return_1h"] = (
    df["Close"].pct_change(1)
)


df["return_3h"] = (
    df["Close"].pct_change(3)
)


df["return_6h"] = (
    df["Close"].pct_change(6)
)


df["return_24h"] = (
    df["Close"].pct_change(24)
)


# ==========================================
# EXPONENTIAL MOVING AVERAGES
# ==========================================

print("Creating EMA features...")


df["ema_9"] = (
    df["Close"]
    .ewm(
        span=9,
        adjust=False
    )
    .mean()
)


df["ema_21"] = (
    df["Close"]
    .ewm(
        span=21,
        adjust=False
    )
    .mean()
)


df["ema_50"] = (
    df["Close"]
    .ewm(
        span=50,
        adjust=False
    )
    .mean()
)


# ==========================================
# RSI 14
# ==========================================

print("Creating RSI...")


delta = df["Close"].diff()

gain = delta.clip(
    lower=0
)

loss = -delta.clip(
    upper=0
)


avg_gain = (
    gain
    .ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    )
    .mean()
)


avg_loss = (
    loss
    .ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    )
    .mean()
)


rs = avg_gain / avg_loss.replace(
    0,
    np.nan
)


df["rsi_14"] = (
    100 - (100 / (1 + rs))
)


# ==========================================
# MACD
# ==========================================

print("Creating MACD...")


ema_12 = (
    df["Close"]
    .ewm(
        span=12,
        adjust=False
    )
    .mean()
)


ema_26 = (
    df["Close"]
    .ewm(
        span=26,
        adjust=False
    )
    .mean()
)


df["macd"] = (
    ema_12 - ema_26
)


df["macd_signal"] = (
    df["macd"]
    .ewm(
        span=9,
        adjust=False
    )
    .mean()
)


df["macd_histogram"] = (
    df["macd"] -
    df["macd_signal"]
)


# ==========================================
# BOLLINGER BANDS
# ==========================================

print("Creating Bollinger Bands...")


bb_period = 20


df["bb_middle"] = (
    df["Close"]
    .rolling(
        window=bb_period
    )
    .mean()
)


bb_std = (
    df["Close"]
    .rolling(
        window=bb_period
    )
    .std()
)


df["bb_high"] = (
    df["bb_middle"] +
    (2 * bb_std)
)


df["bb_low"] = (
    df["bb_middle"] -
    (2 * bb_std)
)


df["bb_width"] = (
    (
        df["bb_high"] -
        df["bb_low"]
    )
    /
    df["bb_middle"]
)


# ==========================================
# ATR 14
# ==========================================

print("Creating ATR...")


previous_close = (
    df["Close"].shift(1)
)


true_range_1 = (
    df["High"] -
    df["Low"]
)


true_range_2 = (
    (df["High"] - previous_close)
    .abs()
)


true_range_3 = (
    (df["Low"] - previous_close)
    .abs()
)


df["true_range"] = pd.concat(
    [
        true_range_1,
        true_range_2,
        true_range_3
    ],
    axis=1
).max(
    axis=1
)


df["atr_14"] = (
    df["true_range"]
    .rolling(
        window=14
    )
    .mean()
)


# ==========================================
# VOLUME FEATURES
# ==========================================

print("Creating volume features...")


df["volume_change"] = (
    df["Volume"].pct_change()
)


df["volume_ma_20"] = (
    df["Volume"]
    .rolling(
        window=20
    )
    .mean()
)


df["volume_ratio"] = (
    df["Volume"] /
    df["volume_ma_20"]
)


# ==========================================
# VOLATILITY
# ==========================================

print("Creating volatility features...")


df["volatility_24h"] = (
    df["return_1h"]
    .rolling(
        window=24
    )
    .std()
)


# ==========================================
# PRICE VS EMA
# ==========================================

print("Creating price-vs-EMA features...")


df["price_vs_ema9"] = (
    df["Close"] /
    df["ema_9"] - 1
)


df["price_vs_ema21"] = (
    df["Close"] /
    df["ema_21"] - 1
)


df["price_vs_ema50"] = (
    df["Close"] /
    df["ema_50"] - 1
)


# ==========================================
# FUTURE RETURN
# ==========================================

print("Creating future return...")


df["future_return_6h"] = (
    df["Close"].shift(-6) /
    df["Close"] - 1
)


# ==========================================
# TARGET
# ==========================================
#
#  1  = BUY
#  0  = HOLD
# -1  = SELL
#
# BUY  : future return > +0.5%
# SELL : future return < -0.5%
# HOLD : between -0.5% and +0.5%
#
# ==========================================

print("Creating trading target...")


def create_target(future_return):

    if pd.isna(future_return):
        return np.nan

    if future_return > 0.005:
        return 1

    elif future_return < -0.005:
        return -1

    else:
        return 0


df["target"] = (
    df["future_return_6h"]
    .apply(create_target)
)


# ==========================================
# REMOVE TEMPORARY COLUMN
# ==========================================

df = df.drop(
    columns=["true_range"]
)


# ==========================================
# REMOVE NaN VALUES
# ==========================================

print("Cleaning dataset...")


df = df.dropna(
    subset=[
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
        "macd_histogram",
        "bb_high",
        "bb_low",
        "bb_middle",
        "bb_width",
        "atr_14",
        "volume_change",
        "volume_ma_20",
        "volume_ratio",
        "volatility_24h",
        "price_vs_ema9",
        "price_vs_ema21",
        "price_vs_ema50",
        "future_return_6h",
        "target"
    ]
)


# ==========================================
# TARGET AS INTEGER
# ==========================================

df["target"] = (
    df["target"]
    .astype(int)
)


# ==========================================
# FINAL COLUMN ORDER
# ==========================================

columns = [
    "Datetime",

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

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
    "macd_histogram",

    "bb_high",
    "bb_low",
    "bb_middle",
    "bb_width",

    "atr_14",

    "volume_change",
    "volume_ma_20",
    "volume_ratio",

    "volatility_24h",

    "price_vs_ema9",
    "price_vs_ema21",
    "price_vs_ema50",

    "future_return_6h",

    "target"
]


df = df[columns]


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n")
print("==========================================")
print("FEATURE ENGINEERING COMPLETE")
print("==========================================")


print("\nDataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nLatest rows:")
print(
    df.tail(3)
)


print("\nTarget distribution:")
print(
    df["target"].value_counts()
    .sort_index()
)


print("\nTarget percentages:")
print(
    (
        df["target"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)
)


# ==========================================
# SAVE DATASET
# ==========================================

df.to_csv(
    output_file,
    index=False
)


print("\nSaved to:")
print(output_file)


print("\n==========================================")
print("DONE")
print("==========================================")