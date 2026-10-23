import sys
import os


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ============================================================
# STRATEGY CONFIG
# ============================================================

# Minimum AI confidence required to allow a trade
MIN_CONFIDENCE = 0.55


# ============================================================
# GENERATE TRADING SIGNAL
# ============================================================

def generate_signal(prediction_result):

    """
    Convert AI prediction into a final trading signal.

    Expected prediction_result:

    {
        "signal": "BUY",
        "prediction": 1,
        "price": 119.10,
        "timestamp": "...",
        "probabilities": [...]
    }

    Returns:

    {
        "signal": "BUY / SELL / HOLD",
        "confidence": 0.70,
        "reason": "AI_SIGNAL"
    }
    """

    # --------------------------------------------------------
    # No prediction
    # --------------------------------------------------------

    if prediction_result is None:

        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": "NO_PREDICTION"
        }


    # --------------------------------------------------------
    # Get AI signal
    # --------------------------------------------------------

    signal = prediction_result.get(
        "signal",
        "HOLD"
    )


    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = prediction_result.get(
        "probabilities",
        None
    )


    # --------------------------------------------------------
    # Calculate confidence
    # --------------------------------------------------------

    confidence = 0.0

    if probabilities is not None:

        try:

            confidence = float(
                max(probabilities)
            )

        except Exception:

            confidence = 0.0


    # --------------------------------------------------------
    # Normalize signal
    # --------------------------------------------------------

    signal = str(
        signal
    ).upper()


    if signal not in [
        "BUY",
        "SELL",
        "HOLD"
    ]:

        signal = "HOLD"


    # --------------------------------------------------------
    # Confidence filter
    # --------------------------------------------------------

    if confidence < MIN_CONFIDENCE:

        return {
            "signal": "HOLD",
            "confidence": confidence,
            "reason": "LOW_CONFIDENCE"
        }


    # --------------------------------------------------------
    # HOLD should remain HOLD
    # --------------------------------------------------------

    if signal == "HOLD":

        return {
            "signal": "HOLD",
            "confidence": confidence,
            "reason": "AI_HOLD"
        }


    # --------------------------------------------------------
    # Valid BUY / SELL signal
    # --------------------------------------------------------

    return {
        "signal": signal,
        "confidence": confidence,
        "reason": "AI_SIGNAL"
    }


# ============================================================
# DISPLAY SIGNAL
# ============================================================

def display_signal(result):

    print()
    print("=" * 60)
    print("SOLANA AI TRADING STRATEGY")
    print("=" * 60)

    print(
        f"Signal      : {result['signal']}"
    )

    print(
        f"Confidence  : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Reason      : {result['reason']}"
    )

    print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SOLANA AI STRATEGY TEST")
    print("=" * 60)


    # --------------------------------------------------------
    # TEST 1 - HIGH CONFIDENCE BUY
    # --------------------------------------------------------

    print()
    print("TEST 1 : HIGH CONFIDENCE BUY")

    test_buy = {

        "signal": "BUY",

        "prediction": 1,

        "price": 119.10,

        "timestamp": "TEST",

        "probabilities": [
            0.20,
            0.10,
            0.70
        ]
    }


    result = generate_signal(
        test_buy
    )


    display_signal(
        result
    )


    # --------------------------------------------------------
    # TEST 2 - LOW CONFIDENCE
    # --------------------------------------------------------

    print()
    print("TEST 2 : LOW CONFIDENCE")

    test_low_confidence = {

        "signal": "BUY",

        "prediction": 1,

        "price": 119.10,

        "timestamp": "TEST",

        "probabilities": [
            0.34,
            0.32,
            0.34
        ]
    }


    result = generate_signal(
        test_low_confidence
    )


    display_signal(
        result
    )


    # --------------------------------------------------------
    # TEST 3 - SELL
    # --------------------------------------------------------

    print()
    print("TEST 3 : HIGH CONFIDENCE SELL")

    test_sell = {

        "signal": "SELL",

        "prediction": 0,

        "price": 119.10,

        "timestamp": "TEST",

        "probabilities": [
            0.75,
            0.15,
            0.10
        ]
    }


    result = generate_signal(
        test_sell
    )


    display_signal(
        result
    )


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STRATEGY TEST COMPLETE")
    print("=" * 60)