import logging
import pandas as pd
import datetime
import os
import time
import threading
import schedule
import gdown
from zipfile import ZipFile

from telegram import Bot, Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import GoldTradingEnv, add_indicators, fetch_data_twelvedata

# === Download model files if missing ===
if not os.path.exists("gold_ppo_model_retrained.zip"):
    gdown.download(id="13BWWyOspY0yZW0yNrdFqbA6QEnMn1Mfh", output="gold_ppo_model_retrained.zip", quiet=False)
    with ZipFile("gold_ppo_model_retrained.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

if not os.path.exists("vec_normalize.pkl"):
    gdown.download(id="1Q-wl0aqr4QAv6xDiQr1DX8T9cdVAQrZ1", output="vec_normalize.pkl", quiet=False)

# === Logging ===
def log_signal(action, price, rsi, macd, ema, tp=None, sl=None, source="manual"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "signal_log.csv"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("datetime,source,price,rsi,macd,ema,action,tp,sl\n")
    with open(file_path, "a") as f:
        f.write(f"{timestamp},{source},{price:.2f},{rsi:.2f},{macd:.4f},{ema:.2f},{action},{tp or ''},{sl or ''}\n")

# === Load initial model ===
dummy_env = DummyVecEnv([lambda: GoldTradingEnv(add_indicators(fetch_data_twelvedata()))])
vec_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False
model = PPO.load("gold_ppo_model_retrained", env=vec_env)

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
bot = Bot(token=TELEGRAM_TOKEN)

# === /start ===
def start(update: Update, context: CallbackContext):
    welcome_msg = (
        "🤖 *Welcome to the Gold Trading Bot!*\n\n"
        "This bot uses a trained reinforcement learning AI to generate real-time trading signals for *XAU/USD*.\n\n"
        "📌 *Available Commands:*\n"
        "• /predict — Get the latest AI signal using live market data\n"
        "• /export — Export the signal log as CSV\n"
        "• /start — Show this help message\n\n"
        "⏱️ Signals are also sent automatically every hour (on the hour).\n"
        "💡 Make sure notifications are enabled so you never miss a trade idea.\n\n"
        f"🆔 *Your Chat ID:* `{update.message.chat_id}`"
    )
    update.message.reply_text(welcome_msg, parse_mode='Markdown')


# === /predict ===
def predict(update: Update, context: CallbackContext):
    try:
        df_live = add_indicators(fetch_data_twelvedata())
        latest = df_live.iloc[-1]
        close_price = latest['close']
        rsi = latest['rsi']
        macd = latest['macd']
        ema = latest['ema_20']

        temp_env = DummyVecEnv([lambda: GoldTradingEnv(df_live)])
        temp_env = VecNormalize.load("vec_normalize.pkl", temp_env)
        temp_env.training = False
        temp_env.norm_reward = False

        env = temp_env.envs[0]
        env.reset()  # 🛠️ Initialize internal vars
        env.current_step = len(df_live) - 2
        obs = env._get_obs().reshape(1, -1)

        action, _ = model.predict(obs)
        action_name = ["Hold", "Buy", "Sell"][action[0]]

        if action_name in ["Buy", "Sell"]:
            tp = close_price + 4 if action_name == "Buy" else close_price - 4
            sl = close_price - 3 if action_name == "Buy" else close_price + 3
            tp_sl_line = f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
        else:
            tp = sl = None
            tp_sl_line = "📌 No TP/SL — holding"

        msg = (
            f"📊 Live Signal: {action_name}\n"
            f"💰 Price: {close_price:.2f}\n"
            f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}\n"
            f"{tp_sl_line}"
        )

        update.message.reply_text(msg)
        log_signal(action_name, close_price, rsi, macd, ema, tp, sl, source="manual-live")

    except Exception as e:
        update.message.reply_text(f"❌ Error during prediction: {e}")

# === /export ===
def export_log(update: Update, context: CallbackContext):
    file_path = "signal_log.csv"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            update.message.reply_document(InputFile(f, filename="signal_log.csv"))
    else:
        update.message.reply_text("⚠️ No signal log available yet.")

# === Auto Signal ===
last_signal = {"timestamp": None}

def check_market_and_send_signal():
    global last_signal
    try:
        df_live = add_indicators(fetch_data_twelvedata())
        latest = df_live.iloc[-1]
        current_time = latest['datetime']
        if last_signal["timestamp"] == current_time:
            return

        temp_env = DummyVecEnv([lambda: GoldTradingEnv(df_live)])
        temp_env = VecNormalize.load("vec_normalize.pkl", temp_env)
        temp_env.training = False
        temp_env.norm_reward = False

        env = temp_env.envs[0]
        env.reset()  # ✅ Fix: initialize current_step
        env.current_step = len(df_live) - 2
        obs = env._get_obs().reshape(1, -1)

        action, _ = model.predict(obs)
        action_name = ["Hold", "Buy", "Sell"][action[0]]

        close_price = latest['close']
        rsi = latest['rsi']
        macd = latest['macd']
        ema = latest['ema_20']

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

# === Scheduler ===
def run_scheduler():
    schedule.every().hour.at(":00").do(check_market_and_send_signal)
    while True:
        schedule.run_pending()
        time.sleep(10)

# === Run Bot ===
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("export", export_log))
    threading.Thread(target=run_scheduler, daemon=True).start()
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
