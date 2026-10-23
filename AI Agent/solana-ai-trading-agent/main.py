import time
from datetime import datetime, timezone

# ============================================================
# SOLANA AI TRADING AGENT
# MAIN CONTROLLER
# ============================================================

from live_feed import get_market_data, create_features
from ai.predictor import predict
from strategy.strategy import generate_signal, display_signal
from strategy.risk_manager import create_trade_plan, display_trade


# ============================================================
# CONFIG
# ============================================================

LOOP_INTERVAL = 60 * 60   # 1 hour


# ============================================================
# DISPLAY HEADER
# ============================================================

def display_header():

    print()
    print("=" * 70)
    print("SOLANA AI TRADING AGENT")
    print("=" * 70)

    print("Mode       : PAPER TRADING")
    print("Market     : SOL/USDT")
    print("Timeframe  : 1H")
    print("Status     : RUNNING")

    print("=" * 70)


# ============================================================
# RUN ONE TRADING CYCLE
# ============================================================

def run_trading_cycle():

    print()
    print("=" * 70)
    print("NEW TRADING CYCLE")
    print("=" * 70)

    print(
        "Time       :",
        datetime.now(timezone.utc)
    )

    # --------------------------------------------------------
    # STEP 1 - GET MARKET DATA
    # --------------------------------------------------------

    print()
    print("[1/5] Getting live market data...")

    df = get_market_data()

    if df is None:

        print("ERROR: Market data unavailable.")
        return

    print(
        f"Market rows : {len(df)}"
    )


    # --------------------------------------------------------
    # STEP 2 - CREATE FEATURES
    # --------------------------------------------------------

    print()
    print("[2/5] Creating trading features...")

    try:

        feature_df = create_features(df)

    except Exception as e:

        print()
        print("ERROR CREATING FEATURES:")
        print(str(e))

        return

    if feature_df is None:

        print("ERROR: Feature generation failed.")
        return

    print(
        f"Feature rows: {len(feature_df)}"
    )


    # --------------------------------------------------------
    # STEP 3 - AI PREDICTION
    # --------------------------------------------------------

    print()
    print("[3/5] Running AI prediction...")

    try:

        prediction_result = predict()

    except Exception as e:

        print()
        print("ERROR RUNNING AI MODEL:")
        print(str(e))

        return

    if prediction_result is None:

        print("ERROR: AI prediction unavailable.")
        return


    # --------------------------------------------------------
    # SHOW AI RESULT
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("AI PREDICTION")
    print("-" * 70)

    if isinstance(prediction_result, dict):

        print(
            f"Prediction : "
            f"{prediction_result.get('prediction')}"
        )

        print(
            f"Price      : "
            f"${prediction_result.get('price', 0):.4f}"
        )

        probabilities = prediction_result.get(
            "probabilities"
        )

        if probabilities is not None:

            print(
                f"Probabilities: "
                f"{probabilities}"
            )

    print("-" * 70)


    # --------------------------------------------------------
    # STEP 4 - STRATEGY
    # --------------------------------------------------------

    print()
    print("[4/5] Generating trading strategy...")

    strategy_result = generate_signal(
        prediction_result
    )

    display_signal(
        strategy_result
    )


    # --------------------------------------------------------
    # CHECK HOLD
    # --------------------------------------------------------

    if strategy_result["signal"] == "HOLD":

        print()
        print("=" * 70)
        print("NO TRADE")
        print("=" * 70)

        print(
            f"Reason     : "
            f"{strategy_result['reason']}"
        )

        print(
            f"Confidence : "
            f"{strategy_result['confidence'] * 100:.2f}%"
        )

        print("=" * 70)

        return


    # --------------------------------------------------------
    # STEP 5 - RISK MANAGEMENT
    # --------------------------------------------------------

    print()
    print("[5/5] Checking risk management...")

    price = float(
        prediction_result.get(
            "price",
            0
        )
    )

    confidence = float(
        strategy_result["confidence"]
    )

    signal = strategy_result["signal"]


    if price <= 0:

        print("ERROR: Invalid entry price.")
        return


    try:

        trade_plan = create_trade_plan(
            signal=signal,
            confidence=confidence,
            balance=1000.0,
            entry_price=price
        )

    except Exception as e:

        print()
        print("ERROR CREATING TRADE PLAN:")
        print(str(e))

        return


    # --------------------------------------------------------
    # DISPLAY TRADE PLAN
    # --------------------------------------------------------

    display_trade(
        trade_plan
    )


    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRADING DECISION")
    print("=" * 70)

    print(
        f"Signal     : {signal}"
    )

    print(
        f"Confidence : "
        f"{confidence * 100:.2f}%"
    )

    print(
        "Risk Check : PASSED"
    )

    print(
        "Execution  : PAPER ONLY"
    )

    print("=" * 70)


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    display_header()

    print()
    print("Starting trading engine...")
    print()

    while True:

        try:

            run_trading_cycle()

            print()
            print("=" * 70)
            print("TRADING CYCLE COMPLETE")
            print("=" * 70)

            print()
            print("WAITING FOR NEXT MARKET UPDATE")

            print(
                "Next update in 60 minutes."
            )

            print(
                "Press CTRL+C to stop."
            )

            print("=" * 70)

            time.sleep(
                LOOP_INTERVAL
            )

        except KeyboardInterrupt:

            print()
            print()
            print("=" * 70)
            print("TRADING AGENT STOPPED")
            print("=" * 70)

            break

        except Exception as e:

            print()
            print("=" * 70)
            print("MAIN LOOP ERROR")
            print("=" * 70)

            print(
                str(e)
            )

            print("=" * 70)

            print()
            print("Retrying in 60 seconds...")

            time.sleep(60)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()