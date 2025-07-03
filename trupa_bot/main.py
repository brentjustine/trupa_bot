import logging
import pandas as pd
import datetime
import os
import time
import threading
import schedule
from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import GoldTradingEnv, add_indicators, fetch_data_twelvedata

import gdown

# === Download model files from Google Drive if not present ===
if not os.path.exists("gold_ppo_model_retrained.zip"):
    gdown.download(id="13BWWyOspY0yZW0yNrdFqbA6QEnMn1Mfh", output="gold_ppo_model_retrained.zip", quiet=False)
    from zipfile import ZipFile
    with ZipFile("gold_ppo_model_retrained.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

if not os.path.exists("vec_normalize.pkl"):
    gdown.download(id="1Q-wl0aqr4QAv6xDiQr1DX8T9cdVAQrZ1", output="vec_normalize.pkl", quiet=False)

# === Logging Setup ===
logging.basicConfig(filename='trade_signals.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_signal(message):
    print(message)
    logging.info(message)
    with open("signal_log.csv", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{message}\n")

# === Load Data and Indicators for Initial Setup ===
df = add_indicators(fetch_data_twelvedata())

# === Create Environment ===
def make_env():
    return GoldTradingEnv(df)

dummy_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load("gold_ppo_model_retrained", env=vec_env)

# === Signal Filter ===
def is_good_trade(row, action):
    rsi = row['rsi']
    macd = row['macd']
    ema = row['ema_20']
    close = row['close']
    if action == 1:  # Buy
        return rsi < 30 and macd > 0 and close > ema
    elif action == 2:  # Sell
        return rsi > 70 and macd < 0 and close < ema
    return False

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))

bot = Bot(token=TELEGRAM_TOKEN)

# === Manual Command: /start ===
def start(update: Update, context: CallbackContext):
    update.message.reply_text("🤖 Gold Trading Bot Active! Use /predict to get a trading signal.")
    update.message.reply_text(f"Your chat ID is: {update.message.chat_id}")

# === Manual Command: /predict ===
def predict(update: Update, context: CallbackContext):
    obs = vec_env.reset()
    action, _ = model.predict(obs)
    action_name = ["Hold", "Buy", "Sell"][action[0]]

    latest = df.iloc[-1]
    close_price = latest['close']
    rsi = latest['rsi']
    macd = latest['macd']
    ema = latest['ema_20']

    if action_name in ["Buy", "Sell"]:
        tp = close_price + 4 if action_name == "Buy" else close_price - 4
        sl = close_price - 3 if action_name == "Buy" else close_price + 3
        tp_sl_line = f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
    else:
        tp_sl_line = "📌 No TP/SL — holding"

    msg = (
        f"📊 Signal: {action_name}\n"
        f"💰 Price: {close_price:.2f}\n"
        f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}\n"
        f"{tp_sl_line}"
    )

    update.message.reply_text(msg)
    log_signal(msg)


# === Scheduled Monitoring Job ===
last_signal = {"timestamp": None, "action": None}

def check_market_and_send_signal():
    global last_signal
    try:
        df_live = add_indicators(fetch_data_twelvedata())
        latest = df_live.iloc[-1]
        current_time = latest['datetime']

        # Avoid duplicate signals per candle
        if last_signal["timestamp"] == current_time:
            return

        obs = vec_env.reset()
        action, _ = model.predict(obs)
        action = action[0]
        action_name = ["Hold", "Buy", "Sell"][action]

        if action_name == "Hold":
            return

        close_price = latest['close']
        rsi = latest['rsi']
        macd = latest['macd']
        ema = latest['ema_20']

        tp = close_price + 4 if action_name == "Buy" else close_price - 4
        sl = close_price - 3 if action_name == "Buy" else close_price + 3

        msg = (
            f"📊 Auto Signal: {action_name}\n"
            f"💰 Entry Price: {close_price:.2f}\n"
            f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}\n"
            f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
        )

        bot.send_message(chat_id=CHAT_ID, text=msg)
        log_signal(msg)

        last_signal = {"timestamp": current_time, "action": action_name}

    except Exception as e:
        print(f"[Monitor Error] {e}")


# === Background Scheduler Thread ===
def run_scheduler():
    schedule.every().hour.at(":00").do(check_market_and_send_signal)
    while True:
        schedule.run_pending()
        time.sleep(10)

# === Main Entrypoint ===
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))

    # Start scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
