import logging
import pandas as pd
import datetime
import os
from telegram.ext import Updater, CommandHandler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import GoldTradingEnv, add_indicators, fetch_data_twelvedata

# === Logging Setup ===
logging.basicConfig(filename='trade_signals.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_signal(message):
    print(message)
    logging.info(message)
    with open("signal_log.csv", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{message}\n")

# === Load Data and Indicators ===
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
    return True

# === Telegram Bot ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
updater = Updater(TELEGRAM_TOKEN, use_context=True)
dispatcher = updater.dispatcher

# === Commands ===
def start(update, context):
    update.message.reply_text("🤖 Gold Trading Bot Active! Use /predict to get a trading signal.")

def predict(update, context):
    obs = vec_env.reset()
    action, _ = model.predict(obs)
    action_name = ["Hold", "Buy", "Sell"][action[0]]

    latest = df.iloc[-1]
    close_price = latest['close']
    rsi = latest['rsi']
    macd = latest['macd']
    ema = latest['ema_20']

    if not is_good_trade(latest, action[0]):
        action_name = "Hold (Filtered)"
        log_signal("⚠️ Signal filtered due to weak setup.")
        update.message.reply_text("🛑 Weak signal. Holding position.")
    else:
        pip = 0.01
        sl_pips = 30 * pip
        tp_pips = 40 * pip

        if action[0] == 1:
            tp = close_price + tp_pips
            sl = close_price - sl_pips
        elif action[0] == 2:
            tp = close_price - tp_pips
            sl = close_price + sl_pips
        else:
            tp = sl = close_price

        msg = (
            f"📊 Signal: {action_name}\n"
            f"💰 Price: {close_price:.2f}\n"
            f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}\n"
        )

        if action[0] in [1, 2]:
            msg += f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"

        update.message.reply_text(msg)
        log_signal(f"{action_name} | Price: {close_price:.2f} | TP: {tp:.2f} | SL: {sl:.2f} | RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA20: {ema:.2f}")

# === Register Commands ===
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("predict", predict))

# === Start Bot ===
updater.start_polling()
updater.idle()
