import math


# ============================================================
# SOLANA AI TRADING AGENT
# RISK MANAGER
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RISK_PER_TRADE = 0.01       # 1% account risk
DEFAULT_STOP_LOSS_PCT = 0.02        # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT = 0.04      # 4% take profit
MIN_CONFIDENCE = 0.60               # 60% minimum confidence


# ============================================================
# CALCULATE POSITION SIZE
# ============================================================

def calculate_position_size(
    balance,
    entry_price,
    stop_loss_price,
    risk_per_trade=DEFAULT_RISK_PER_TRADE
):

    if balance <= 0:
        raise ValueError("Balance must be greater than zero.")

    if entry_price <= 0:
        raise ValueError("Entry price must be greater than zero.")

    if stop_loss_price <= 0:
        raise ValueError("Stop loss price must be greater than zero.")

    risk_amount = balance * risk_per_trade

    price_risk = abs(
        entry_price - stop_loss_price
    )

    if price_risk <= 0:
        raise ValueError(
            "Entry price and stop loss price cannot be equal."
        )

    quantity = risk_amount / price_risk

    return quantity


# ============================================================
# CALCULATE LONG TRADE
# ============================================================

def calculate_long_trade(
    balance,
    entry_price,
    stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
    take_profit_pct=DEFAULT_TAKE_PROFIT_PCT
):

    stop_loss_price = (
        entry_price *
        (1 - stop_loss_pct)
    )

    take_profit_price = (
        entry_price *
        (1 + take_profit_pct)
    )

    quantity = calculate_position_size(
        balance,
        entry_price,
        stop_loss_price
    )

    position_value = (
        quantity *
        entry_price
    )

    risk_amount = (
        quantity *
        abs(
            entry_price -
            stop_loss_price
        )
    )

    potential_profit = (
        quantity *
        abs(
            take_profit_price -
            entry_price
        )
    )

    return {
        "side": "LONG",
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "quantity": quantity,
        "position_value": position_value,
        "risk_amount": risk_amount,
        "potential_profit": potential_profit
    }


# ============================================================
# CALCULATE SHORT TRADE
# ============================================================

def calculate_short_trade(
    balance,
    entry_price,
    stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
    take_profit_pct=DEFAULT_TAKE_PROFIT_PCT
):

    stop_loss_price = (
        entry_price *
        (1 + stop_loss_pct)
    )

    take_profit_price = (
        entry_price *
        (1 - take_profit_pct)
    )

    quantity = calculate_position_size(
        balance,
        entry_price,
        stop_loss_price
    )

    position_value = (
        quantity *
        entry_price
    )

    risk_amount = (
        quantity *
        abs(
            entry_price -
            stop_loss_price
        )
    )

    potential_profit = (
        quantity *
        abs(
            take_profit_price -
            entry_price
        )
    )

    return {
        "side": "SHORT",
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "quantity": quantity,
        "position_value": position_value,
        "risk_amount": risk_amount,
        "potential_profit": potential_profit
    }


# ============================================================
# CONFIDENCE CHECK
# ============================================================

def confidence_is_valid(
    confidence,
    minimum_confidence=MIN_CONFIDENCE
):

    if confidence is None:
        return False

    return confidence >= minimum_confidence


# ============================================================
# CREATE TRADE PLAN
# ============================================================

def create_trade_plan(
    signal,
    confidence,
    balance,
    entry_price
):

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if balance <= 0:
        raise ValueError(
            "Balance must be greater than zero."
        )

    if entry_price <= 0:
        raise ValueError(
            "Entry price must be greater than zero."
        )

    signal = signal.upper()

    # --------------------------------------------------------
    # Confidence filter
    # --------------------------------------------------------

    if not confidence_is_valid(
        confidence
    ):

        return {
            "approved": False,
            "reason": "LOW_CONFIDENCE",
            "signal": signal,
            "confidence": confidence
        }

    # --------------------------------------------------------
    # HOLD
    # --------------------------------------------------------

    if signal == "HOLD":

        return {
            "approved": False,
            "reason": "HOLD_SIGNAL",
            "signal": signal,
            "confidence": confidence
        }

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if signal == "BUY":

        trade = calculate_long_trade(
            balance,
            entry_price
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    elif signal == "SELL":

        trade = calculate_short_trade(
            balance,
            entry_price
        )

    else:

        return {
            "approved": False,
            "reason": "INVALID_SIGNAL",
            "signal": signal,
            "confidence": confidence
        }

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    trade["approved"] = True
    trade["reason"] = "TRADE_APPROVED"
    trade["confidence"] = confidence

    return trade


# ============================================================
# DISPLAY TRADE
# ============================================================

def display_trade(trade):

    print("\n")
    print("=" * 60)
    print("RISK MANAGER")
    print("=" * 60)

    if not trade.get("approved", False):

        print(
            f"Trade Approved : NO"
        )

        print(
            f"Reason         : "
            f"{trade.get('reason')}"
        )

        print(
            f"Signal         : "
            f"{trade.get('signal')}"
        )

        confidence = trade.get(
            "confidence"
        )

        if confidence is not None:

            print(
                f"Confidence     : "
                f"{confidence * 100:.2f}%"
            )

        print("=" * 60)

        return

    print(
        f"Trade Approved : YES"
    )

    print(
        f"Side           : "
        f"{trade['side']}"
    )

    print(
        f"Confidence     : "
        f"{trade['confidence'] * 100:.2f}%"
    )

    print(
        f"Entry Price    : "
        f"${trade['entry_price']:.4f}"
    )

    print(
        f"Stop Loss      : "
        f"${trade['stop_loss']:.4f}"
    )

    print(
        f"Take Profit    : "
        f"${trade['take_profit']:.4f}"
    )

    print(
        f"Quantity       : "
        f"{trade['quantity']:.6f} SOL"
    )

    print(
        f"Position Value : "
        f"${trade['position_value']:.2f}"
    )

    print(
        f"Risk Amount    : "
        f"${trade['risk_amount']:.2f}"
    )

    print(
        f"Potential Profit: "
        f"${trade['potential_profit']:.2f}"
    )

    print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SOLANA AI RISK MANAGER TEST")
    print("=" * 60)

    balance = 1000.0
    entry_price = 119.11
    confidence = 0.70

    trade = create_trade_plan(
        signal="BUY",
        confidence=confidence,
        balance=balance,
        entry_price=entry_price
    )

    display_trade(trade)