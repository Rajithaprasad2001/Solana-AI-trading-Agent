import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ============================================================
# SOLANA AI TRADING AGENT
# LIVE MARKET DATA + FEATURE ENGINEERING
# ============================================================


# ============================================================
# CONFIG
# ============================================================

EXCHANGE = ccxt.binance({
    "enableRateLimit": True
})

SYMBOL = "SOL/USDT"
TIMEFRAME = "1h"

CANDLE_LIMIT = 200

RAW_OUTPUT_FILE = "data/live_sol_usdt.csv"
FEATURE_OUTPUT_FILE = "data/live_sol_features.csv"


# ============================================================
# CREATE DATA DIRECTORY
# ============================================================

os.makedirs("data", exist_ok=True)


# ============================================================
# GET LIVE MARKET DATA
# ============================================================

def get_market_data():

    print("\n" + "=" * 60)
    print("SOLANA LIVE MARKET DATA")
    print("=" * 60)

    print(f"Symbol     : {SYMBOL}")
    print(f"Timeframe  : {TIMEFRAME}")
    print(f"Candles    : {CANDLE_LIMIT}")

    try:

        candles = EXCHANGE.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            limit=CANDLE_LIMIT
        )

        if not candles:

            raise ValueError(
                "No market data received from Binance."
            )

        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        # ----------------------------------------------------
        # Convert timestamp
        # ----------------------------------------------------

        df["Datetime"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        # Remove unnecessary timestamp column
        df = df.drop(
            columns=["timestamp"]
        )

        # ----------------------------------------------------
        # Convert numeric columns
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Remove invalid rows
        # ----------------------------------------------------

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna(
            subset=numeric_columns
        )

        if len(df) < 60:

            raise ValueError(
                "Not enough candles for feature calculation."
            )

        # ----------------------------------------------------
        # Save raw live data
        # ----------------------------------------------------

        df.to_csv(
            RAW_OUTPUT_FILE,
            index=False
        )

        print("\nLIVE MARKET DATA")
        print("-" * 60)

        latest = df.iloc[-1]

        print(
            f"Time       : {latest['Datetime']}"
        )

        print(
            f"Open       : ${latest['Open']:.4f}"
        )

        print(
            f"High       : ${latest['High']:.4f}"
        )

        print(
            f"Low        : ${latest['Low']:.4f}"
        )

        print(
            f"Close      : ${latest['Close']:.4f}"
        )

        print(
            f"Volume     : {latest['Volume']:.2f}"
        )

        print("-" * 60)

        print(
            f"Raw data saved: {RAW_OUTPUT_FILE}"
        )

        return df

    except Exception as e:

        print("\nERROR GETTING MARKET DATA:")
        print(str(e))

        return None


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(df):

    print("\n" + "=" * 60)
    print("CREATING AI FEATURES")
    print("=" * 60)


    # ========================================================
    # BASIC RETURNS
    # ========================================================

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


    # ========================================================
    # EXPONENTIAL MOVING AVERAGES
    # ========================================================

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


    # ========================================================
    # RSI 14
    # ========================================================

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

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi_14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


    # ========================================================
    # MACD
    # ========================================================

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


    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

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


    # ========================================================
    # ATR 14
    # ========================================================

    print("Creating ATR...")

    previous_close = (
        df["Close"].shift(1)
    )

    true_range_1 = (
        df["High"] -
        df["Low"]
    )

    true_range_2 = (
        df["High"] -
        previous_close
    ).abs()

    true_range_3 = (
        df["Low"] -
        previous_close
    ).abs()

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


    # ========================================================
    # VOLUME FEATURES
    # ========================================================

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


    # ========================================================
    # VOLATILITY
    # ========================================================

    print("Creating volatility...")

    df["volatility_24h"] = (
        df["return_1h"]
        .rolling(
            window=24
        )
        .std()
    )


    # ========================================================
    # PRICE VS EMA
    # ========================================================

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


    # ========================================================
    # REMOVE TEMPORARY COLUMN
    # ========================================================

    df = df.drop(
        columns=["true_range"]
    )


    # ========================================================
    # CLEAN DATA
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    feature_columns = [
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
        "price_vs_ema50"
    ]

    df = df.dropna(
        subset=feature_columns
    )


    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

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
        "price_vs_ema50"
    ]

    df = df[columns]


    # ========================================================
    # SAVE FEATURE DATASET
    # ========================================================

    df.to_csv(
        FEATURE_OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # DISPLAY LATEST FEATURES
    # ========================================================

    latest = df.iloc[-1]

    print("\n" + "=" * 60)
    print("LATEST AI FEATURES")
    print("=" * 60)

    print(
        f"Time              : {latest['Datetime']}"
    )

    print(
        f"SOL Price         : ${latest['Close']:.4f}"
    )

    print(
        f"Return 1h         : {latest['return_1h']:.6f}"
    )

    print(
        f"Return 3h         : {latest['return_3h']:.6f}"
    )

    print(
        f"Return 6h         : {latest['return_6h']:.6f}"
    )

    print(
        f"Return 24h        : {latest['return_24h']:.6f}"
    )

    print(
        f"EMA 9             : {latest['ema_9']:.4f}"
    )

    print(
        f"EMA 21            : {latest['ema_21']:.4f}"
    )

    print(
        f"EMA 50            : {latest['ema_50']:.4f}"
    )

    print(
        f"RSI 14            : {latest['rsi_14']:.2f}"
    )

    print(
        f"MACD              : {latest['macd']:.6f}"
    )

    print(
        f"MACD Signal       : {latest['macd_signal']:.6f}"
    )

    print(
        f"MACD Histogram    : {latest['macd_histogram']:.6f}"
    )

    print(
        f"BB High           : {latest['bb_high']:.4f}"
    )

    print(
        f"BB Low            : {latest['bb_low']:.4f}"
    )

    print(
        f"BB Width          : {latest['bb_width']:.6f}"
    )

    print(
        f"ATR 14            : {latest['atr_14']:.6f}"
    )

    print(
        f"Volume Ratio      : {latest['volume_ratio']:.4f}"
    )

    print(
        f"Volatility 24h    : {latest['volatility_24h']:.6f}"
    )

    print(
        f"Price vs EMA 9    : {latest['price_vs_ema9']:.6f}"
    )

    print(
        f"Price vs EMA 21   : {latest['price_vs_ema21']:.6f}"
    )

    print(
        f"Price vs EMA 50   : {latest['price_vs_ema50']:.6f}"
    )

    print("-" * 60)

    print(
        f"Feature data saved: {FEATURE_OUTPUT_FILE}"
    )

    print("=" * 60)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("SOLANA AI TRADING AGENT")
    print("LIVE DATA + FEATURE ENGINEERING")
    print("=" * 60)

    print(
        f"Symbol     : {SYMBOL}"
    )

    print(
        f"Timeframe  : {TIMEFRAME}"
    )

    print(
        f"Candles    : {CANDLE_LIMIT}"
    )

    print()
    print("Connecting to Binance...")
    print()

    df = get_market_data()

    if df is not None:

        try:

            feature_df = create_features(
                df
            )

            print()
            print("=" * 60)
            print("LIVE FEATURE ENGINEERING COMPLETE")
            print("=" * 60)

            print(
                f"Rows available: {len(feature_df)}"
            )

            print(
                f"File: {FEATURE_OUTPUT_FILE}"
            )

            print("=" * 60)

        except Exception as e:

            print("\nERROR CREATING FEATURES:")
            print(str(e))

            raise

    else:

        print()
        print("=" * 60)
        print("LIVE FEED FAILED")
        print("=" * 60)