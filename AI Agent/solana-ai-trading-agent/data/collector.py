import yfinance as yf
import pandas as pd


def get_market_data(
    symbol="SOL-USD",
    period="1y",
    interval="1h"
):
    """
    Download historical market data.

    symbol:
        SOL-USD = Solana / USD

    period:
        How much historical data to download.

    interval:
        Candle timeframe.
    """

    print(f"Downloading {symbol} market data...")

    data = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError("No market data received.")

    # Handle possible multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Keep only the columns we need
    required_columns = [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    available_columns = [
        column for column in required_columns
        if column in data.columns
    ]

    data = data[available_columns]

    return data


if __name__ == "__main__":

    df = get_market_data()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nLatest candle:")
    print(df.tail(1))

    # Save dataset
    df.to_csv(
        "data/sol_usd_1h.csv",
        index=False
    )

    print("\nDataset saved to:")
    print("data/sol_usd_1h.csv")