import os
import time
import threading
import datetime
import schedule
import gdown
import logging
import pandas as pd
from zipfile import ZipFile
from flask import Flask
from telegram import Bot, Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import GoldTradingEnv, add_indicators, fetch_data_twelvedata

# === Flask Ping Server ===
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Gold Trading Bot is alive!", 200

# === Download model files from Google Drive if not present ===
if not os.path.exists("gold_ppo_model_retrained.zip"):
    gdown.download(id="13BWWyOspY0yZW0yNrdFqbA6QEnMn1Mfh", output="gold_ppo_model_retrained.zip", quiet=False)
    with ZipFile("gold_ppo_model_retrained.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

if not os.path.exists("vec_normalize.pkl"):
    gdown.download(id="1Q-wl0aqr4QAv6xDiQr1DX8T9cdVAQrZ1", output="vec_normalize.pkl", quiet=False)

# === Signal Logging ===
def log_signal(action, price, rsi, macd, ema, tp=None, sl=None, source="manual"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "signal_log.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("datetime,source,price,rsi,macd,ema,action,tp,sl\n")
    with open(file_path, "a") as f:
        f.write(f"{timestamp},{source},{price:.2f},{rsi:.2f},{macd:.4f},{ema:.2f},{action},{tp or ''},{sl or ''}\n")

# === Load Data & Create Env ===
df = add_indicators(fetch_data_twelvedata())

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

# === Telegram Commands ===
def start(update: Update, context: CallbackContext):
    update.message.reply_text("🤖 Gold Trading Bot Active! Use /predict to get a trading signal.")
    update.message.reply_text(f"Your chat ID is: {update.message.chat_id}")

def predict(update: Update, context: CallbackContext):
    obs = vec_env.reset()
    action, _ = model.predict(obs)
    action_name = ["Hold", "Buy", "Sell"][action[0]]
    latest = df.iloc[-1]
    close_price, rsi, macd, ema = latest['close'], latest['rsi'], latest['macd'], latest['ema_20']

    if action_name in ["Buy", "Sell"]:
        tp = close_price + 4 if action_name == "Buy" else close_price - 4
        sl = close_price - 3 if action_name == "Buy" else close_price + 3
        tp_sl_line = f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
    else:
        tp = sl = None
        tp_sl_line = "📌 No TP/SL — holding"

    msg = (
        f"📊 Signal: {action_name}\n"
        f"💰 Price: {close_price:.2f}\n"
        f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}\n"
        f"{tp_sl_line}"
    )
    update.message.reply_text(msg)
    log_signal(action_name, close_price, rsi, macd, ema, tp, sl, source="manual")

def export_log(update: Update, context: CallbackContext):
    file_path = "signal_log.csv"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            update.message.reply_document(InputFile(f, filename="signal_log.csv"))
    else:
        update.message.reply_text("⚠️ No signal log available yet.")

# === Scheduler Job ===
last_signal = {"timestamp": None}

def check_market_and_send_signal():
    global last_signal
    try:
        df_live = add_indicators(fetch_data_twelvedata())
        latest = df_live.iloc[-1]
        current_time = latest['datetime']

        if last_signal["timestamp"] == current_time:
            return

        obs = vec_env.reset()
        action, _ = model.predict(obs)
        action_name = ["Hold", "Buy", "Sell"][action[0]]
        close_price, rsi, macd, ema = latest['close'], latest['rsi'], latest['macd'], latest['ema_20']

        if action_name in ["Buy", "Sell"]:
            tp = close_price + 4 if action_name == "Buy" else close_price - 4
            sl = close_price - 3 if action_name == "Buy" else close_price + 3
            tp_sl_line = f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
        else:
            tp = sl = None
            tp_sl_line = "📌 No TP/SL — holding"

        msg = (
            f"📊 Auto Signal: {action_name}\n"
            f"💰 Entry Price: {close_price:.2f}\n"
            f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}\n"
            f"{tp_sl_line}"
        )

        bot.send_message(chat_id=CHAT_ID, text=msg)
        log_signal(action_name, close_price, rsi, macd, ema, tp, sl, source="auto")
        last_signal["timestamp"] = current_time

    except Exception as e:
        print(f"[Monitor Error] {e}")

def run_scheduler():
    schedule.every().hour.at(":00").do(check_market_and_send_signal)
    while True:
        schedule.run_pending()
        time.sleep(10)

# === Main Bot Webhook + Scheduler ===
def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("export", export_log))

    threading.Thread(target=run_scheduler, daemon=True).start()

    PORT = int(os.environ.get("PORT", 8443))
    WEBHOOK_URL = f"https://trupa-bot.onrender.com/{TELEGRAM_TOKEN}"

    bot.delete_webhook()
    updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=TELEGRAM_TOKEN,
        webhook_url=WEBHOOK_URL
    )
    updater.idle()

# === Entry Point ===
if __name__ == "__main__":
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8443))),
        daemon=True
    ).start()

    main()

I should paste this on github and commit then redeploy and everything will work now flawlessly?
