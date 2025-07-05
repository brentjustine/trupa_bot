import os
import time
import threading
import datetime
import schedule
import gdown
import pandas as pd
from zipfile import ZipFile
from flask import Flask, request
from telegram import Bot, Update, InputFile
from telegram.ext import Dispatcher, CommandHandler, CallbackContext

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import GoldTradingEnv, add_indicators, fetch_data_twelvedata
import numpy as np

# === Flask Ping Server ===
app = Flask(__name__)
@app.route("/")
def home():
    return "✅ Gold Trading Bot is alive!\nVisit: https://trupa-bot.onrender.com", 200

@app.route(f"/{os.getenv('TELEGRAM_TOKEN')}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "OK", 200

# === Download model files from Google Drive if missing ===
def download_model_files():
    if not os.path.exists("gold_ppo_model_retrained.zip"):
        gdown.download(id="13BWWyOspY0yZW0yNrdFqbA6QEnMn1Mfh", output="gold_ppo_model_retrained.zip", quiet=False)
        with ZipFile("gold_ppo_model_retrained.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    if not os.path.exists("vec_normalize.pkl"):
        gdown.download(id="1Q-wl0aqr4QAv6xDiQr1DX8T9cdVAQrZ1", output="vec_normalize.pkl", quiet=False)

# === Signal Logging ===
def log_signal(action, price, rsi, macd, ema, tp=None, sl=None, source="manual", trade_status="open", update_last=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "signal_log.csv"
    header = "datetime,source,price,rsi,macd,ema,obv,bb_width,vwap,fib_0,fib_618,fib_100,action,tp,sl,status\n"

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(header)

    df_live = add_indicators(fetch_data_twelvedata(interval="1h"))
    latest = df_live.iloc[-1]

    if update_last:
        df = pd.read_csv(file_path)
        if len(df) > 0:
            df.iloc[-1] = [
                timestamp, source, f"{price:.2f}", f"{rsi:.2f}", f"{macd:.4f}", f"{ema:.2f}",
                latest["obv"], latest["bb_width"], latest["vwap"],
                latest["fib_0"], latest["fib_618"], latest["fib_100"],
                action, tp or '', sl or '', trade_status
            ]
            df.to_csv(file_path, index=False)
            return

    with open(file_path, "a") as f:
        f.write(f"{timestamp},{source},{price:.2f},{rsi:.2f},{macd:.4f},{ema:.2f},{latest['obv']},{latest['bb_width']},{latest['vwap']},{latest['fib_0']},{latest['fib_618']},{latest['fib_100']},{action},{tp or ''},{sl or ''},{trade_status}\n")

# === Load Model and Environment ===
download_model_files()
df = add_indicators(fetch_data_twelvedata(interval="1h"))

def make_env():
    return GoldTradingEnv(df)

dummy_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False
model = PPO.load("gold_ppo_model_retrained", env=vec_env)

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)

# === Trade Memory State ===
trade_state = {"open_position": None, "entry_price": None, "tp": None, "sl": None}
waiting_steps = 0
WAITING_THRESHOLD = 4
spread_tolerance = 0.3

# === TP/SL Calculation ===
def calculate_tp_sl(action_name, price):
    tp_buffer = 4.0
    sl_buffer = 3.0
    if action_name == "Buy":
        tp = round(price + tp_buffer, 2)
        sl = round(price - sl_buffer, 2)
    elif action_name == "Sell":
        tp = round(price - tp_buffer, 2)
        sl = round(price + sl_buffer, 2)
    else:
        tp = sl = None
    return tp, sl

# === Auto Trading Logic ===
def check_market_and_send_signal():
    global waiting_steps
    df_live = add_indicators(fetch_data_twelvedata(interval="1h"))
    latest = df_live.iloc[-1]

    if trade_state["open_position"]:
        high = latest["high"]
        low = latest["low"]
        current_price = latest["close"]
        pos = trade_state["open_position"]

        tp_hit = sl_hit = False
        if pos == "Buy":
            tp_hit = high >= (trade_state["tp"] - spread_tolerance)
            sl_hit = low <= (trade_state["sl"] + spread_tolerance)
        elif pos == "Sell":
            tp_hit = low <= (trade_state["tp"] + spread_tolerance)
            sl_hit = high >= (trade_state["sl"] - spread_tolerance)

        if tp_hit or sl_hit:
            result = "✅ TP hit" if tp_hit else "🛑 SL hit"
            log_signal(pos, current_price, latest["rsi"], latest["macd"], latest["ema_20"], trade_state["tp"], trade_state["sl"], source="auto", trade_status=result, update_last=True)
            bot.send_message(chat_id=CHAT_ID, text=f"📤 Trade closed: {result}")
            trade_state.update({"open_position": None, "entry_price": None, "tp": None, "sl": None})
            waiting_steps = 0
        return

    obs = latest[["open", "high", "low", "close", "ema_50", "rsi", "obv", "bb_width", "vwap", "fib_0", "fib_618", "fib_100"]].values.astype(np.float32)
    obs = obs.reshape(1, -1)
    obs = vec_env.normalize_obs(obs)
    action, _ = model.predict(obs, deterministic=True)
    action_name = "Buy" if action == 0 else "Sell"
    tp, sl = calculate_tp_sl(action_name, latest["close"])

    waiting_steps += 1
    if waiting_steps >= WAITING_THRESHOLD:
        bot.send_message(chat_id=CHAT_ID, text="⏳ Wait. No good entry.")
        waiting_steps = 0
        return

    trade_state.update({
        "open_position": action_name,
        "entry_price": latest["close"],
        "tp": tp,
        "sl": sl
    })

    waiting_steps = 0
    log_signal(action_name, latest["close"], latest["rsi"], latest["macd"], latest["ema_20"], tp, sl, source="auto")
    msg = (
        f"📈 Auto Signal: {action_name}\n"
        f"💰 Price: {latest['close']:.2f}\n"
        f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
    )
    bot.send_message(chat_id=CHAT_ID, text=msg)

# === Telegram Bot Commands ===
def start(update: Update, context: CallbackContext):
    msg = (
        "🤖 Welcome to the Gold Trading Bot!\n"
        "Available commands:\n"
        "/predict — Get the latest trading signal.\n"
        "/export — Download the signal log as CSV.\n"
        "🌐 Dashboard: https://trupa-bot.onrender.com"
    )
    update.message.reply_text(msg)

def predict(update: Update, context: CallbackContext):
    global waiting_steps
    df_live = add_indicators(fetch_data_twelvedata(interval="1h"))
    latest = df_live.iloc[-1]

    obs = latest[["open", "high", "low", "close", "ema_50", "rsi", "obv", "bb_width", "vwap", "fib_0", "fib_618", "fib_100"]].values.astype(np.float32)
    obs = obs.reshape(1, -1)
    obs = vec_env.normalize_obs(obs)

    action, _ = model.predict(obs, deterministic=True)
    action_name = "Buy" if action == 0 else "Sell"
    tp, sl = calculate_tp_sl(action_name, latest["close"])

    trade_state.update({
        "open_position": action_name,
        "entry_price": latest["close"],
        "tp": tp,
        "sl": sl
    })

    waiting_steps = 0

    log_signal(action_name, latest["close"], latest["rsi"], latest["macd"], latest["ema_20"], tp, sl, source="manual")

    msg = (
        f"📈 Signal: {action_name}\n"
        f"💰 Price: {latest['close']:.2f}\n"
        f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
    )
    update.message.reply_text(msg)

def export_log(update: Update, context: CallbackContext):
    file_path = "signal_log.csv"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            update.message.reply_document(InputFile(f, filename="signal_log.csv"))
    else:
        update.message.reply_text("⚠️ No signal log available yet.")

# === Scheduler ===
def run_scheduler():
    schedule.every(15).minutes.do(check_market_and_send_signal)
    while True:
        schedule.run_pending()
        time.sleep(10)

# === Attach Commands ===
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("predict", predict))
dispatcher.add_handler(CommandHandler("export", export_log))

# === Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    bot.set_webhook(f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{TELEGRAM_TOKEN}")
    threading.Thread(target=run_scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=port)

