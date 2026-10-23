import pandas as pd

FILE = "backtest/backtest_results.csv"

df = pd.read_csv(FILE)

df["entry_time"] = pd.to_datetime(df["entry_time"])
df["exit_time"] = pd.to_datetime(df["exit_time"])

# ============================================================
# BASIC STATISTICS
# ============================================================

total_trades = len(df)

wins = df[df["net_pnl"] > 0]
losses = df[df["net_pnl"] < 0]

win_count = len(wins)
loss_count = len(losses)

win_rate = (
    win_count / total_trades * 100
    if total_trades > 0
    else 0
)

# ============================================================
# PROFITABILITY
# ============================================================

net_pnl = df["net_pnl"].sum()

gross_profit = wins["net_pnl"].sum()
gross_loss = abs(losses["net_pnl"].sum())

profit_factor = (
    gross_profit / gross_loss
    if gross_loss > 0
    else float("inf")
)

avg_win = (
    wins["net_pnl"].mean()
    if win_count > 0
    else 0
)

avg_loss = (
    losses["net_pnl"].mean()
    if loss_count > 0
    else 0
)

best_trade = df["net_pnl"].max()
worst_trade = df["net_pnl"].min()

# ============================================================
# EQUITY / DRAWDOWN
# ============================================================

df["equity"] = df["net_pnl"].cumsum()

df["peak"] = df["equity"].cummax()

df["drawdown"] = df["equity"] - df["peak"]

max_drawdown = df["drawdown"].min()

# ============================================================
# LONG / SHORT
# ============================================================

df["side"] = df["side"].astype(str).str.upper()

long_df = df[df["side"] == "LONG"]
short_df = df[df["side"] == "SHORT"]

long_trades = len(long_df)
short_trades = len(short_df)

long_pnl = long_df["net_pnl"].sum()
short_pnl = short_df["net_pnl"].sum()

# ============================================================
# FEES
# ============================================================

total_fees = df["fees"].sum()

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("SOLANA AI TRADING AGENT - BACKTEST ANALYSIS")
print("=" * 70)

print()
print("TRADE STATISTICS")
print("-" * 70)

print(f"Total Trades       : {total_trades}")
print(f"Wins               : {win_count}")
print(f"Losses             : {loss_count}")
print(f"Win Rate           : {win_rate:.2f}%")

print()
print("PROFITABILITY")
print("-" * 70)

print(f"Net PNL            : {net_pnl:.2f}")
print(f"Gross Profit       : {gross_profit:.2f}")
print(f"Gross Loss         : {gross_loss:.2f}")
print(f"Profit Factor      : {profit_factor:.2f}")
print(f"Average Win        : {avg_win:.2f}")
print(f"Average Loss       : {avg_loss:.2f}")
print(f"Best Trade         : {best_trade:.2f}")
print(f"Worst Trade        : {worst_trade:.2f}")

print()
print("RISK")
print("-" * 70)

print(f"Maximum Drawdown   : {max_drawdown:.2f}")

print()
print("LONG / SHORT")
print("-" * 70)

print(f"Long Trades        : {long_trades}")
print(f"Long PNL           : {long_pnl:.2f}")
print(f"Short Trades       : {short_trades}")
print(f"Short PNL          : {short_pnl:.2f}")

print()
print("COSTS")
print("-" * 70)

print(f"Total Fees         : {total_fees:.2f}")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)