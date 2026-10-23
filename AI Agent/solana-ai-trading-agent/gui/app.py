from flask import Flask, render_template_string, jsonify
import os
import json
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(BASE_DIR, "ai", "sol_model.joblib")
FEATURE_FILE = os.path.join(BASE_DIR, "data", "sol_features.csv")
TRADES_FILE = os.path.join(BASE_DIR, "paper", "paper_trades.csv")
EQUITY_FILE = os.path.join(BASE_DIR, "paper", "equity_curve.csv")


# ============================================================
# LOAD MODEL
# ============================================================

model = None

try:
    model = joblib.load(MODEL_FILE)
    print("AI model loaded successfully.")
except Exception as e:
    print("Model loading error:", e)


# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        print("CSV error:", e)

    return pd.DataFrame()


def get_latest_data():

    df = safe_read_csv(FEATURE_FILE)

    if df.empty:
        return None

    # Try datetime sorting
    time_columns = [
        "timestamp",
        "time",
        "datetime",
        "date"
    ]

    for col in time_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
                break
            except:
                pass

    return df.iloc[-1]


def get_model_features():

    if model is None:
        return []

    try:

        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)

        # Pipeline
        if hasattr(model, "named_steps"):

            for _, step in model.named_steps.items():

                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)

    except:
        pass

    return []


def predict_latest():

    latest = get_latest_data()

    if latest is None or model is None:
        return {
            "prediction": None,
            "signal": "N/A",
            "confidence": 0,
            "probabilities": {}
        }

    feature_names = get_model_features()

    # --------------------------------------------------------
    # Build feature dataframe
    # --------------------------------------------------------

    if feature_names:

        values = {}

        for feature in feature_names:

            if feature in latest.index:
                values[feature] = latest[feature]
            else:
                values[feature] = 0.0

        X = pd.DataFrame([values])

    else:

        exclude = {
            "target",
            "timestamp",
            "time",
            "datetime",
            "date"
        }

        numeric_cols = [
            c for c in latest.index
            if c not in exclude
        ]

        X = pd.DataFrame(
            [[latest[c] for c in numeric_cols]],
            columns=numeric_cols
        )

    # Convert everything to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace bad values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(X)[0]

        probabilities = {}

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(X)[0]

            classes = getattr(
                model,
                "classes_",
                range(len(probs))
            )

            for cls, prob in zip(classes, probs):
                probabilities[str(cls)] = float(prob)

        # ----------------------------------------------------
        # Convert prediction to signal
        # ----------------------------------------------------

        try:
            pred_int = int(prediction)
        except:
            pred_int = 0

        if pred_int == 1:
            signal = "BUY"

        elif pred_int == -1:
            signal = "SELL"

        else:
            signal = "HOLD"

        confidence = max(
            probabilities.values()
        ) if probabilities else 0

        return {
            "prediction": pred_int,
            "signal": signal,
            "confidence": confidence * 100,
            "probabilities": probabilities
        }

    except Exception as e:

        print("Prediction error:", e)

        return {
            "prediction": None,
            "signal": "ERROR",
            "confidence": 0,
            "probabilities": {}
        }


# ============================================================
# DASHBOARD DATA
# ============================================================

def dashboard_data():

    latest = get_latest_data()
    prediction = predict_latest()

    trades = safe_read_csv(TRADES_FILE)
    equity = safe_read_csv(EQUITY_FILE)

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    price = 0

    if latest is not None:

        for col in ["Close", "close", "price"]:

            if col in latest.index:

                try:
                    price = float(latest[col])
                    break
                except:
                    pass

    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    current_time = "N/A"

    if latest is not None:

        for col in [
            "timestamp",
            "time",
            "datetime",
            "date"
        ]:

            if col in latest.index:

                current_time = str(latest[col])
                break

    # --------------------------------------------------------
    # Paper trading balance
    # --------------------------------------------------------

    balance = 1000.0

    if not equity.empty:

        possible_balance_columns = [
            "balance",
            "equity",
            "balance_after",
            "portfolio_value"
        ]

        for col in possible_balance_columns:

            if col in equity.columns:

                try:
                    balance = float(
                        equity[col].iloc[-1]
                    )
                    break
                except:
                    pass

    # --------------------------------------------------------
    # Trade statistics
    # --------------------------------------------------------

    total_trades = len(trades)

    winning_trades = 0
    losing_trades = 0
    total_pnl = 0

    if not trades.empty:

        if "net_pnl" in trades.columns:

            pnl = pd.to_numeric(
                trades["net_pnl"],
                errors="coerce"
            ).fillna(0)

            total_pnl = float(pnl.sum())

            winning_trades = int(
                (pnl > 0).sum()
            )

            losing_trades = int(
                (pnl < 0).sum()
            )

    win_rate = 0

    if total_trades > 0:
        win_rate = (
            winning_trades /
            total_trades
        ) * 100

    # --------------------------------------------------------
    # Recent trades
    # --------------------------------------------------------

    recent_trades = []

    if not trades.empty:

        recent = trades.tail(10)

        for _, row in recent.iterrows():

            recent_trades.append({

                "side": str(
                    row.get("side", "-")
                ),

                "entry_price": float(
                    pd.to_numeric(
                        row.get(
                            "entry_price",
                            0
                        ),
                        errors="coerce"
                    ) or 0
                ),

                "exit_price": float(
                    pd.to_numeric(
                        row.get(
                            "exit_price",
                            0
                        ),
                        errors="coerce"
                    ) or 0
                ),

                "pnl": float(
                    pd.to_numeric(
                        row.get(
                            "net_pnl",
                            0
                        ),
                        errors="coerce"
                    ) or 0
                ),

                "reason": str(
                    row.get(
                        "exit_reason",
                        "-"
                    )
                )

            })

    return {

        "price": price,

        "time": current_time,

        "signal": prediction["signal"],

        "prediction": prediction["prediction"],

        "confidence": prediction["confidence"],

        "probabilities": prediction["probabilities"],

        "balance": balance,

        "total_trades": total_trades,

        "winning_trades": winning_trades,

        "losing_trades": losing_trades,

        "win_rate": win_rate,

        "total_pnl": total_pnl,

        "recent_trades": recent_trades,

        "features": get_model_features()

    }


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Solana AI Trading Dashboard</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #050816,
            #0b1024,
            #111827
        );

    color: #ffffff;

    min-height: 100vh;

}

.container {

    width: 94%;

    max-width: 1500px;

    margin: auto;

    padding: 30px 0;

}

.header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 30px;

}

.title {

    font-size: 30px;

    font-weight: bold;

}

.subtitle {

    color: #9ca3af;

    margin-top: 6px;

}

.status {

    padding: 10px 16px;

    border-radius: 20px;

    background: rgba(34,197,94,0.15);

    color: #4ade80;

    border: 1px solid rgba(34,197,94,0.3);

}

.cards {

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(220px, 1fr)
        );

    gap: 18px;

    margin-bottom: 25px;

}

.card {

    background:
        rgba(17,24,39,0.85);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 10px 30px
        rgba(0,0,0,0.25);

}

.card-title {

    color: #9ca3af;

    font-size: 14px;

    margin-bottom: 10px;

}

.card-value {

    font-size: 27px;

    font-weight: bold;

}

.dashboard {

    display: grid;

    grid-template-columns:
        1.4fr 1fr;

    gap: 20px;

}

.panel {

    background:
        rgba(17,24,39,0.85);

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 25px;

    margin-bottom: 20px;

}

.panel h2 {

    margin-top: 0;

}

.signal {

    text-align: center;

    padding: 35px 20px;

    border-radius: 18px;

    background:
        rgba(255,255,255,0.03);

}

.signal-value {

    font-size: 52px;

    font-weight: bold;

    margin: 15px;

}

.buy {

    color: #22c55e;

}

.sell {

    color: #ef4444;

}

.hold {

    color: #facc15;

}

.probability {

    margin: 15px 0;

}

.bar {

    width: 100%;

    height: 10px;

    background: #1f2937;

    border-radius: 10px;

    overflow: hidden;

    margin-top: 7px;

}

.bar-inner {

    height: 100%;

    width: 0%;

    transition: width 0.5s;

}

.buybar {

    background: #22c55e;

}

.sellbar {

    background: #ef4444;

}

.holdbar {

    background: #facc15;

}

table {

    width: 100%;

    border-collapse: collapse;

}

th, td {

    padding: 12px;

    text-align: left;

    border-bottom:
        1px solid
        rgba(255,255,255,0.06);

}

th {

    color: #9ca3af;

    font-size: 13px;

}

.positive {

    color: #4ade80;

}

.negative {

    color: #f87171;

}

.features {

    display: flex;

    flex-wrap: wrap;

    gap: 8px;

}

.feature {

    background: #1f2937;

    padding: 7px 10px;

    border-radius: 8px;

    font-size: 12px;

    color: #d1d5db;

}

.footer {

    text-align: center;

    color: #6b7280;

    padding: 20px;

}

@media(max-width: 900px) {

    .dashboard {

        grid-template-columns: 1fr;

    }

    .header {

        flex-direction: column;

        align-items: flex-start;

        gap: 15px;

    }

}

</style>

</head>


<body>


<div class="container">


<div class="header">

<div>

<div class="title">
🤖 Solana AI Trading Agent
</div>

<div class="subtitle">
AI-powered SOL/USDT paper trading dashboard
</div>

</div>

<div class="status">
● PAPER TRADING
</div>

</div>


<!-- ================================================= -->
<!-- TOP CARDS -->
<!-- ================================================= -->

<div class="cards">


<div class="card">

<div class="card-title">
SOL PRICE
</div>

<div class="card-value">
$<span id="price">0.00</span>
</div>

</div>


<div class="card">

<div class="card-title">
PAPER BALANCE
</div>

<div class="card-value">
$<span id="balance">0.00</span>
</div>

</div>


<div class="card">

<div class="card-title">
TOTAL PNL
</div>

<div class="card-value">
$<span id="pnl">0.00</span>
</div>

</div>


<div class="card">

<div class="card-title">
WIN RATE
</div>

<div class="card-value">
<span id="winrate">0.00</span>%
</div>

</div>


</div>


<div class="dashboard">


<!-- ================================================= -->
<!-- LEFT -->
<!-- ================================================= -->

<div>


<div class="panel">

<h2>
🤖 AI Prediction
</h2>


<div class="signal">

<div class="card-title">
CURRENT AI SIGNAL
</div>

<div
id="signal"
class="signal-value hold"
>
HOLD
</div>

<div>

Confidence:
<strong>
<span id="confidence">0</span>%
</strong>

</div>

</div>


<div class="probability">

SELL:
<strong>
<span id="sell">0</span>%
</strong>

<div class="bar">
<div
id="sellbar"
class="bar-inner sellbar">
</div>
</div>

</div>


<div class="probability">

HOLD:
<strong>
<span id="hold">0</span>%
</strong>

<div class="bar">
<div
id="holdbar"
class="bar-inner holdbar">
</div>
</div>

</div>


<div class="probability">

BUY:
<strong>
<span id="buy">0</span>%
</strong>

<div class="bar">
<div
id="buybar"
class="bar-inner buybar">
</div>
</div>

</div>


</div>


<!-- ================================================= -->
<!-- TRADES -->
<!-- ================================================= -->

<div class="panel">

<h2>
📋 Recent Trades
</h2>

<table>

<thead>

<tr>

<th>Side</th>

<th>Entry</th>

<th>Exit</th>

<th>PNL</th>

<th>Reason</th>

</tr>

</thead>

<tbody id="trades">

</tbody>

</table>

</div>


</div>


<!-- ================================================= -->
<!-- RIGHT -->
<!-- ================================================= -->

<div>


<div class="panel">

<h2>
📊 Trading Statistics
</h2>

<p>
Total Trades:
<strong id="totalTrades">
0
</strong>
</p>

<p>
Winning Trades:
<strong id="wins">
0
</strong>
</p>

<p>
Losing Trades:
<strong id="losses">
0
</strong>
</p>

<p>
Win Rate:
<strong id="statsWinrate">
0%
</strong>
</p>

<p>
Total PNL:
<strong id="statsPnl">
$0
</strong>
</p>

</div>


<div class="panel">

<h2>
🧠 AI Features
</h2>

<div
id="features"
class="features"
>
</div>

</div>


<div class="panel">

<h2>
🕐 Latest Data
</h2>

<p>
Time:
</p>

<strong id="time">
N/A
</strong>

<p style="margin-top:20px">

Prediction:
<strong id="prediction">
N/A
</strong>

</p>

</div>


</div>


</div>


<div class="footer">

Solana AI Trading Agent •
Paper Trading Only •
No real money connected

</div>


</div>


<script>


async function updateDashboard() {

    try {

        const response =
            await fetch("/api/data");

        const data =
            await response.json();


        // ----------------------------------------------
        // TOP CARDS
        // ----------------------------------------------

        document.getElementById(
            "price"
        ).textContent =
            Number(data.price)
            .toFixed(4);


        document.getElementById(
            "balance"
        ).textContent =
            Number(data.balance)
            .toFixed(2);


        document.getElementById(
            "pnl"
        ).textContent =
            Number(data.total_pnl)
            .toFixed(2);


        document.getElementById(
            "winrate"
        ).textContent =
            Number(data.win_rate)
            .toFixed(2);


        // ----------------------------------------------
        // SIGNAL
        // ----------------------------------------------

        const signal =
            document.getElementById(
                "signal"
            );

        signal.textContent =
            data.signal;


        signal.className =
            "signal-value " +
            data.signal.toLowerCase();


        document.getElementById(
            "confidence"
        ).textContent =
            Number(data.confidence)
            .toFixed(2);


        // ----------------------------------------------
        // PROBABILITIES
        // ----------------------------------------------

        let sell = 0;
        let hold = 0;
        let buy = 0;


        if (data.probabilities) {

            sell =
                Number(
                    data.probabilities["-1"] || 0
                ) * 100;

            hold =
                Number(
                    data.probabilities["0"] || 0
                ) * 100;

            buy =
                Number(
                    data.probabilities["1"] || 0
                ) * 100;

        }


        document.getElementById(
            "sell"
        ).textContent =
            sell.toFixed(2);


        document.getElementById(
            "hold"
        ).textContent =
            hold.toFixed(2);


        document.getElementById(
            "buy"
        ).textContent =
            buy.toFixed(2);


        document.getElementById(
            "sellbar"
        ).style.width =
            sell + "%";


        document.getElementById(
            "holdbar"
        ).style.width =
            hold + "%";


        document.getElementById(
            "buybar"
        ).style.width =
            buy + "%";


        // ----------------------------------------------
        // STATISTICS
        // ----------------------------------------------

        document.getElementById(
            "totalTrades"
        ).textContent =
            data.total_trades;


        document.getElementById(
            "wins"
        ).textContent =
            data.winning_trades;


        document.getElementById(
            "losses"
        ).textContent =
            data.losing_trades;


        document.getElementById(
            "statsWinrate"
        ).textContent =
            Number(data.win_rate)
            .toFixed(2) + "%";


        document.getElementById(
            "statsPnl"
        ).textContent =
            "$" +
            Number(data.total_pnl)
            .toFixed(2);


        // ----------------------------------------------
        // LATEST DATA
        // ----------------------------------------------

        document.getElementById(
            "time"
        ).textContent =
            data.time;


        document.getElementById(
            "prediction"
        ).textContent =
            data.prediction;


        // ----------------------------------------------
        // FEATURES
        // ----------------------------------------------

        const features =
            document.getElementById(
                "features"
            );

        features.innerHTML = "";


        data.features.forEach(
            function(feature) {

                const span =
                    document.createElement(
                        "span"
                    );

                span.className =
                    "feature";

                span.textContent =
                    feature;

                features.appendChild(
                    span
                );

            }
        );


        // ----------------------------------------------
        // TRADES
        // ----------------------------------------------

        const trades =
            document.getElementById(
                "trades"
            );

        trades.innerHTML = "";


        data.recent_trades.forEach(
            function(trade) {

                const row =
                    document.createElement(
                        "tr"
                    );


                const pnlClass =
                    trade.pnl >= 0
                    ? "positive"
                    : "negative";


                row.innerHTML = `

                    <td>
                        ${trade.side}
                    </td>

                    <td>
                        $${Number(
                            trade.entry_price
                        ).toFixed(4)}
                    </td>

                    <td>
                        $${Number(
                            trade.exit_price
                        ).toFixed(4)}
                    </td>

                    <td class="${pnlClass}">
                        $${Number(
                            trade.pnl
                        ).toFixed(2)}
                    </td>

                    <td>
                        ${trade.reason}
                    </td>

                `;


                trades.appendChild(row);

            }
        );


    }

    catch(error) {

        console.error(
            "Dashboard error:",
            error
        );

    }

}


// Initial load
updateDashboard();


// Refresh every 10 seconds
setInterval(
    updateDashboard,
    10000
);


</script>


</body>

</html>

"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


@app.route("/api/data")
def api_data():

    return jsonify(
        dashboard_data()
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SOLANA AI TRADING DASHBOARD")
    print("=" * 60)
    print()
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print()
    print("Paper trading only.")
    print("No real money is connected.")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )