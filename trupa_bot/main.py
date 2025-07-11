import os
import time
import numpy as np
import pandas as pd
import requests
from flask import Flask, request
import telegram
from telegram.ext import Dispatcher, CommandHandler

# === Environment Variables ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

# === Telegram Bot Initialization ===
bot = telegram.Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)

# === Dynamic TP/SL Generator ===
def generate_dynamic_tp_sl(entry, direction, atr, rr=2.0, strategy=None):
    strategy_rr = {
        "breakout": 2.0,
        "rsi_reversal": 1.5,
        "engulfing": 1.5,
        "grid_bias": 1.2,
        "squeeze": 2.5
    }
    rr = strategy_rr.get(strategy, rr)
    if direction == "buy":
        tp = entry + atr * rr
        sl = entry - atr * 1.0
    else:
        tp = entry - atr * rr
        sl = entry + atr * 1.0
    return tp, sl

# === Strategy Definitions ===
def breakout_signal(df, i, lookback=50):
    if i < lookback: return None
    recent_high = df["high"].iloc[i - lookback:i].max()
    high = df["high"].iloc[i]
    close = df["close"].iloc[i]
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

def send_telegram_signal(signal):
    if not signal: return
    msg = f"\u2705 Signal at {signal['timestamp']}\nStrategy: {signal['strategy']}\nDirection: {signal['direction']}\nEntry: {round(signal['entry'], 2)}\nTP: {round(signal['tp'], 2)}\nSL: {round(signal['sl'], 2)}\nConfidence: {signal['confidence']}\nVotes: {signal['strategy_votes']}"
    bot.send_message(chat_id=CHAT_ID, text=msg)

# === Flask App ===
app = Flask(__name__)

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "OK", 200

@app.route("/predict", methods=["GET"])
def predict():
    try:
        df = fetch_data(apikey=TWELVE_DATA_API_KEY)
        signal = generate_ensemble_signal(df, -1)
        if signal:
            signal["timestamp"] = df["timestamp"].iloc[-1]
            send_telegram_signal(signal)
            return {"status": "ok", "signal": signal}, 200
        else:
            return {"status": "ok", "signal": None}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    domain = os.getenv("WEBHOOK_DOMAIN")  # e.g., https://your-render-url
    if not domain:
        return {"error": "WEBHOOK_DOMAIN not set"}, 400
    webhook_url = f"{domain}/{TELEGRAM_TOKEN}"
    success = bot.set_webhook(webhook_url)
    return {"webhook_url": webhook_url, "success": success}, 200

@app.route("/", methods=["GET"])
def home():
    return "SignalBot is running.", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
