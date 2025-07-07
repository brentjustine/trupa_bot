import logging
import pandas as pd
import datetime
import os
from telegram import Bot, Update, InputFile
from telegram.ext import Updater, CommandHandler, CallbackContext
from flask import Flask, request
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import TPSSLTradingEnv, add_indicators, fetch_data_twelvedata

# === Flask App ===
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Gold Trading Bot is alive!\nVisit: https://trupa-bot.onrender.com", 200

@app.route(f"/{os.getenv('TELEGRAM_TOKEN')}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dp.process_update(update)
    return "OK", 200

# === Load Model and VecNormalize ===
dummy_env = DummyVecEnv([lambda: TPSSLTradingEnv(add_indicators(fetch_data_twelvedata()))])
vec_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False
model = PPO.load("gold_ppo_model_retrained", env=vec_env)

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
bot = Bot(token=TELEGRAM_TOKEN)

# === /start Command ===
def start(update: Update, context: CallbackContext):
    welcome_msg = (
        "🤖 *Welcome to the Gold Trading Bot!*\n\n"
        "This bot uses a trained RL AI to generate real-time trading signals for *XAU/USD*.\n\n"
        "📌 *Available Commands:*\n"
        "• /predict — Get the latest AI signal using live market data\n"
        "• /start — Show this help message\n\n"
        f"🆔 *Your Chat ID:* `{update.message.chat_id}`"
    )
    update.message.reply_text(welcome_msg, parse_mode='Markdown')

# === /predict Command ===
def predict(update: Update, context: CallbackContext):
    try:
        df_live = add_indicators(fetch_data_twelvedata())
        latest = df_live.iloc[-1]
        close_price = latest['close']
        rsi = latest['rsi']
        macd = latest['macd']
        ema_50 = latest['ema_50']

        state = latest.values.reshape(1, -1)
        state = vec_env.normalize_obs(state)
        action, _states = model.predict(state, deterministic=True)
        action_name = "Buy" if action == 1 else "Sell" if action == 2 else "Hold"

        tp = sl = None
        if action_name == "Buy":
            tp = close_price + 4.0
            sl = close_price - 3.0
        elif action_name == "Sell":
            tp = close_price - 4.0
            sl = close_price + 3.0

        tp_sl_line = f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}" if tp and sl else "📌 No TP/SL — holding"

        msg = (
            f"📊 Live Signal: {action_name}\n"
            f"💰 Price: {close_price:.2f}\n"
            f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA50: {ema_50:.2f}\n"
            f"{tp_sl_line}"
        )
        update.message.reply_text(msg)

    except Exception as e:
        update.message.reply_text(f"❌ Error during prediction: {e}")

# === Main ===
if __name__ == "__main__":
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))

    bot.set_webhook(f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{TELEGRAM_TOKEN}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
