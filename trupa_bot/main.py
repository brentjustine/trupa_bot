import os
import requests
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import telegram
from telegram.ext import Dispatcher, CommandHandler

# === ENV VARS ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
RENDER_HOST = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.environ.get("PORT", 10000))

# === Flask App Init ===
app = Flask(__name__)
bot = telegram.Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

# === TP/SL Generator ===
def generate_dynamic_tp_sl(entry, direction, atr, rr=2.0, strategy=None):
    rr_map = {"breakout": 2.0, "rsi_reversal": 1.5, "engulfing": 1.5, "grid_bias": 1.2, "squeeze": 2.5}
    rr = rr_map.get(strategy, rr)
    tp = entry + atr * rr if direction == "buy" else entry - atr * rr
    sl = entry - atr if direction == "buy" else entry + atr
    return tp, sl

# === Signal Strategies ===
def breakout_signal(df, i, lookback=50):
    if i < lookback: return None
    recent_high = df["high"].iloc[i-lookback:i].max()
    high, close = df["high"].iloc[i], df["close"].iloc[i]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
    if high > recent_high:
        tp, sl = generate_dynamic_tp_sl(close, "buy", atr, strategy="breakout")
        return {"direction": "buy", "confidence": 0.85, "entry": close, "tp": tp, "sl": sl, "strategy": "breakout"}
    return None

def rsi_reversal_signal(df, i, rsi_period=14, overbought=70, oversold=30):
    if i < rsi_period: return None
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[i]
    close = df["close"].iloc[i]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
    if current_rsi < oversold:
        tp, sl = generate_dynamic_tp_sl(close, "buy", atr, strategy="rsi_reversal")
        return {"direction": "buy", "confidence": 0.75, "entry": close, "tp": tp, "sl": sl, "strategy": "rsi_reversal"}
    elif current_rsi > overbought:
        tp, sl = generate_dynamic_tp_sl(close, "sell", atr, strategy="rsi_reversal")
        return {"direction": "sell", "confidence": 0.75, "entry": close, "tp": tp, "sl": sl, "strategy": "rsi_reversal"}
    return None

def grid_bias_signal(df, i, grid_size=5):
    if i < grid_size: return None
    midline = df["close"].rolling(grid_size).mean().iloc[i]
    close = df["close"].iloc[i]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
    if close < midline * 0.995:
        tp, sl = generate_dynamic_tp_sl(close, "buy", atr, strategy="grid_bias")
        return {"direction": "buy", "confidence": 0.7, "entry": close, "tp": tp, "sl": sl, "strategy": "grid_bias"}
    elif close > midline * 1.005:
        tp, sl = generate_dynamic_tp_sl(close, "sell", atr, strategy="grid_bias")
        return {"direction": "sell", "confidence": 0.7, "entry": close, "tp": tp, "sl": sl, "strategy": "grid_bias"}
    return None

def engulfing_signal(df, i):
    if i < 2: return None
    o1, c1 = df["open"].iloc[i - 1], df["close"].iloc[i - 1]
    o2, c2 = df["open"].iloc[i], df["close"].iloc[i]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
    if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
        tp, sl = generate_dynamic_tp_sl(c2, "buy", atr, strategy="engulfing")
        return {"direction": "buy", "confidence": 0.65, "entry": c2, "tp": tp, "sl": sl, "strategy": "engulfing"}
    elif c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
        tp, sl = generate_dynamic_tp_sl(c2, "sell", atr, strategy="engulfing")
        return {"direction": "sell", "confidence": 0.65, "entry": c2, "tp": tp, "sl": sl, "strategy": "engulfing"}
    return None

def squeeze_signal(df, i, short=10, long=30):
    if i < long: return None
    short_range = (df["high"] - df["low"]).rolling(short).mean().iloc[i]
    long_range = (df["high"] - df["low"]).rolling(long).mean().iloc[i]
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[i]
    close = df["close"].iloc[i]
    if short_range < 0.5 * long_range:
        direction = "buy" if close > df["close"].iloc[i - 1] else "sell"
        tp, sl = generate_dynamic_tp_sl(close, direction, atr, strategy="squeeze")
        return {"direction": direction, "confidence": 0.7, "entry": close, "tp": tp, "sl": sl, "strategy": "squeeze"}
    return None

# === Ensemble Signal Engine ===
def generate_ensemble_signal(df, i):
    strategies = [
        breakout_signal, rsi_reversal_signal, grid_bias_signal,
        engulfing_signal, squeeze_signal
    ]
    signals = [s(df, i) for s in strategies if s(df, i)]
    if not signals: return None
    best = max(signals, key=lambda x: x["confidence"])
    best["strategy_votes"] = len(signals)
    return best

# === Fetch Live Data ===
def fetch_data(symbol="XAU/USD", interval="5min", apikey=None):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={apikey}&outputsize=500"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise Exception("❌ Data fetch failed.")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns=str.lower)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df.rename(columns={"datetime": "timestamp"})

# === Telegram Sender ===
def send_telegram_signal(signal):
    if not signal: return
    msg = (
        f"✅ Signal at {signal['timestamp']}\n"
        f"Strategy: {signal['strategy']}\n"
        f"Direction: {signal['direction']}\n"
        f"Entry: {round(signal['entry'], 2)}\n"
        f"TP: {round(signal['tp'], 2)}\n"
        f"SL: {round(signal['sl'], 2)}\n"
        f"Confidence: {signal['confidence']}\n"
        f"Votes: {signal['strategy_votes']}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# === Flask Routes ===
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "alive"}), 200

@app.route("/predict", methods=["GET"])
def predict():
    try:
        df = fetch_data(apikey=TWELVE_DATA_API_KEY)
        signal = generate_ensemble_signal(df, -1)
        if signal:
            signal["timestamp"] = df["timestamp"].iloc[-1]
            send_telegram_signal(signal)
        return jsonify({"status": "ok", "signal": signal}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

# === /start Command ===
def start(update, context):
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=(
            "👋 Welcome to the SignalBot!\n\n"
            "Use /predict to get the latest trading signal based on multiple strategies.\n"
            "✅ Strategies: breakout, RSI reversal, engulfing, grid bias, squeeze"
        )
    )

dispatcher.add_handler(CommandHandler("start", start))

# === Entrypoint ===
if __name__ == "__main__":
    if RENDER_HOST:
        bot.set_webhook(f"https://{RENDER_HOST}/{TELEGRAM_TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
