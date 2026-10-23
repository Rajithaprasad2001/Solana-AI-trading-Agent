import os
import csv
from datetime import datetime, timezone


# ============================================================
# SOLANA AI TRADING AGENT
# PAPER TRADER
# ============================================================

INITIAL_BALANCE = 1000.0

TRADE_FILE = "paper/paper_trades.csv"
EQUITY_FILE = "paper/equity_curve.csv"


# ============================================================
# PAPER TRADER
# ============================================================

class PaperTrader:

    def __init__(self, initial_balance=INITIAL_BALANCE):

        self.initial_balance = float(initial_balance)

        self.balance = float(initial_balance)

        self.position = None

        self.trade_history = []

        self.total_pnl = 0.0

        self.equity_history = []

        self._create_files()

    # ========================================================
    # CREATE CSV FILES
    # ========================================================

    def _create_files(self):

        os.makedirs("paper", exist_ok=True)

        if not os.path.exists(TRADE_FILE):

            with open(
                TRADE_FILE,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "side",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "pnl",
                    "reason"
                ])

        if not os.path.exists(EQUITY_FILE):

            with open(
                EQUITY_FILE,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "equity"
                ])

    # ========================================================
    # GET EQUITY
    # ========================================================

    def get_equity(self):

        if self.position is None:

            return self.balance

        return self.balance

    # ========================================================
    # OPEN POSITION
    # ========================================================

    def open_position(
        self,
        side,
        entry_price,
        quantity
    ):

        if self.position is not None:

            print("Position already open.")

            return False

        side = str(side).upper()

        if side not in ["LONG", "SHORT"]:

            print("Invalid position side.")

            return False

        entry_price = float(entry_price)

        quantity = float(quantity)

        if entry_price <= 0:

            print("Invalid entry price.")

            return False

        if quantity <= 0:

            print("Invalid quantity.")

            return False

        self.position = {

            "side": side,

            "entry_price": entry_price,

            "quantity": quantity,

            "entry_time": datetime.now(
                timezone.utc
            ).isoformat()
        }

        print()
        print("=" * 60)
        print("PAPER POSITION OPENED")
        print("=" * 60)

        print(f"Side        : {side}")
        print(
            f"Entry Price : ${entry_price:.4f}"
        )

        print(
            f"Quantity    : {quantity:.6f} SOL"
        )

        print(
            f"Position    : ${entry_price * quantity:.2f}"
        )

        print("=" * 60)

        return True

    # ========================================================
    # CLOSE POSITION
    # ========================================================

    def close_position(
        self,
        exit_price,
        reason="MANUAL"
    ):

        if self.position is None:

            print("No open position.")

            return None

        exit_price = float(exit_price)

        side = self.position["side"]

        entry_price = self.position[
            "entry_price"
        ]

        quantity = self.position[
            "quantity"
        ]

        # ----------------------------------------------------
        # Calculate PNL
        # ----------------------------------------------------

        if side == "LONG":

            pnl = (
                exit_price - entry_price
            ) * quantity

        else:

            pnl = (
                entry_price - exit_price
            ) * quantity

        self.balance += pnl

        self.total_pnl += pnl

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        trade = {

            "timestamp": timestamp,

            "side": side,

            "entry_price": entry_price,

            "exit_price": exit_price,

            "quantity": quantity,

            "pnl": pnl,

            "reason": reason
        }

        self.trade_history.append(
            trade
        )

        # ----------------------------------------------------
        # Save trade
        # ----------------------------------------------------

        with open(
            TRADE_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                side,
                entry_price,
                exit_price,
                quantity,
                pnl,
                reason
            ])

        # ----------------------------------------------------
        # Save equity
        # ----------------------------------------------------

        with open(
            EQUITY_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                self.balance
            ])

        print()
        print("=" * 60)
        print("PAPER POSITION CLOSED")
        print("=" * 60)

        print(f"Side        : {side}")

        print(
            f"Entry Price : ${entry_price:.4f}"
        )

        print(
            f"Exit Price  : ${exit_price:.4f}"
        )

        print(
            f"Quantity    : {quantity:.6f} SOL"
        )

        print(
            f"PNL         : ${pnl:.2f}"
        )

        print(
            f"Balance     : ${self.balance:.2f}"
        )

        print(
            f"Reason      : {reason}"
        )

        print("=" * 60)

        self.position = None

        return trade

    # ========================================================
    # PROCESS SIGNAL
    # ========================================================

    def process_signal(
        self,
        signal,
        price,
        quantity
    ):

        signal = str(signal).upper()

        price = float(price)

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if signal == "BUY":

            if self.position is None:

                return self.open_position(
                    "LONG",
                    price,
                    quantity
                )

            if self.position["side"] == "SHORT":

                self.close_position(
                    price,
                    "SIGNAL_REVERSAL"
                )

                return self.open_position(
                    "LONG",
                    price,
                    quantity
                )

            return False

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        if signal == "SELL":

            if self.position is None:

                return self.open_position(
                    "SHORT",
                    price,
                    quantity
                )

            if self.position["side"] == "LONG":

                self.close_position(
                    price,
                    "SIGNAL_REVERSAL"
                )

                return self.open_position(
                    "SHORT",
                    price,
                    quantity
                )

            return False

        # ----------------------------------------------------
        # HOLD
        # ----------------------------------------------------

        if signal == "HOLD":

            print(
                "HOLD signal - "
                "no new position opened."
            )

            return False

        return False

    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        print()
        print("=" * 60)
        print("PAPER TRADER STATUS")
        print("=" * 60)

        print(
            f"Initial Balance : "
            f"${self.initial_balance:.2f}"
        )

        print(
            f"Current Balance : "
            f"${self.balance:.2f}"
        )

        print(
            f"Total PNL       : "
            f"${self.total_pnl:.2f}"
        )

        if self.position:

            print(
                f"Position        : "
                f"{self.position['side']}"
            )

            print(
                f"Entry Price     : "
                f"${self.position['entry_price']:.4f}"
            )

            print(
                f"Quantity        : "
                f"{self.position['quantity']:.6f}"
            )

        else:

            print(
                "Position        : NONE"
            )

        print(
            f"Total Trades    : "
            f"{len(self.trade_history)}"
        )

        print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SOLANA AI PAPER TRADER TEST")
    print("=" * 60)

    trader = PaperTrader(
        initial_balance=1000
    )

    # --------------------------------------------------------
    # TEST LONG
    # --------------------------------------------------------

    trader.open_position(
        side="LONG",
        entry_price=119.10,
        quantity=4.0
    )

    # Simulate price going up

    trader.close_position(
        exit_price=123.00,
        reason="TAKE_PROFIT"
    )

    # --------------------------------------------------------
    # TEST SHORT
    # --------------------------------------------------------

    trader.open_position(
        side="SHORT",
        entry_price=123.00,
        quantity=4.0
    )

    # Simulate price going down

    trader.close_position(
        exit_price=120.00,
        reason="TAKE_PROFIT"
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    trader.status()

    print()
    print("PAPER TRADER TEST COMPLETE")
    print("=" * 60)