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
from flask import Flask, request
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from trading_env import TPSSLTradingEnv, add_indicators, fetch_data_twelvedata

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Gold Trading Bot is alive!\nVisit: https://trupa-bot.onrender.com", 200

@app.route(f"/{os.getenv('TELEGRAM_TOKEN')}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dp.process_update(update)
    return "OK", 200

# === Download model files if missing ===
if not os.path.exists("gold_ppo_model_retrained.zip"):
    gdown.download(id="1t4wHEXStdKQX7mtDxAWE8eYtxSvkloXq", output="gold_ppo_model_retrained.zip", quiet=False)
    with ZipFile("gold_ppo_model_retrained.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

if not os.path.exists("vec_normalize.pkl"):
    gdown.download(id="1p6LOB3pM5-YhgNrLgF68X9dFT_hjhSqR", output="vec_normalize.pkl", quiet=False)

# === Logging ===
def log_signal(action, price, rsi, macd, ema, tp=None, sl=None, source="manual", trade_status="open", update_last=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "signal_log.csv"

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("datetime,source,price,rsi,macd,ema,action,tp,sl,status\n")

    def safe_str(val, fmt):
        return fmt.format(val) if val is not None else ""

    price_str = safe_str(price, "{:.2f}")
    rsi_str = safe_str(rsi, "{:.2f}")
    macd_str = safe_str(macd, "{:.4f}")
    ema_str = safe_str(ema, "{:.2f}")
    tp_str = safe_str(tp, "{:.2f}")
    sl_str = safe_str(sl, "{:.2f}")

    if update_last:
        df = pd.read_csv(file_path)
        if len(df) > 0:
            df.iloc[-1] = [timestamp, source, price_str, rsi_str, macd_str, ema_str, action, tp_str, sl_str, trade_status]
            df.to_csv(file_path, index=False)
            return

    with open(file_path, "a") as f:
        f.write(f"{timestamp},{source},{price_str},{rsi_str},{macd_str},{ema_str},{action},{tp_str},{sl_str},{trade_status}\n")

# === Load initial model ===
dummy_env = DummyVecEnv([lambda: TPSSLTradingEnv(add_indicators(fetch_data_twelvedata()))])
vec_env = VecNormalize.load("vec_normalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False
model = PPO.load("gold_ppo_model_retrained", env=vec_env)

# === Telegram Setup ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))
bot = Bot(token=TELEGRAM_TOKEN)

# === Global Trade State ===
trade_lock = threading.Lock()
trade_open = False
current_action = None
current_tp = None
current_sl = None
trade_entry_price = None
trade_timestamp = None

# === /start ===
def start(update: Update, context: CallbackContext):
    welcome_msg = (
        "🤖 *Welcome to the Gold Trading Bot!*\n\n"
        "This bot uses a trained reinforcement learning AI to generate real-time trading signals for *XAU/USD*.\n\n"
        "📌 *Available Commands:*\n"
        "• /predict — Get the latest AI signal using live market data\n"
        "• /export — Export the signal log as CSV\n"
        "• /start — Show this help message\n\n"
        "⏱️ Signals are also sent automatically every 15 minutes.\n"
        "💡 Make sure notifications are enabled so you never miss a signal.\n\n"
        f"🆔 *Your Chat ID:* `{update.message.chat_id}`"
    )
    update.message.reply_text(welcome_msg, parse_mode='Markdown')

# === /predict ===
def predict(update: Update, context: CallbackContext):
    global trade_open, current_action, current_tp, current_sl, trade_entry_price, trade_timestamp
    try:
        with trade_lock:
            df_live = add_indicators(fetch_data_twelvedata())
            latest = df_live.iloc[-1]
            close_price = latest['close']
            high = latest['high']
            low = latest['low']
            rsi = latest['rsi']
            macd = latest['macd']
            ema_50 = latest['ema_50']
            spread = 0.3

            if trade_open:
                if current_action == "Buy":
                    if high >= current_tp - spread:
                        bot.send_message(chat_id=CHAT_ID, text="🎯 Buy TP hit (spread-adjusted)!")
                        log_signal("Buy", close_price, None, None, None, current_tp, current_sl, trade_status="TP Hit", update_last=True)
                        trade_open = False
                        return
                    elif low <= current_sl + spread:
                        bot.send_message(chat_id=CHAT_ID, text="🛑 Buy SL hit (spread-adjusted)!")
                        log_signal("Buy", close_price, None, None, None, current_tp, current_sl, trade_status="SL Hit", update_last=True)
                        trade_open = False
                        return

                elif current_action == "Sell":
                    if low <= current_tp + spread:
                        bot.send_message(chat_id=CHAT_ID, text="🎯 Sell TP hit (spread-adjusted)!")
                        log_signal("Sell", close_price, None, None, None, current_tp, current_sl, trade_status="TP Hit", update_last=True)
                        trade_open = False
                        return
                    elif high >= current_sl - spread:
                        bot.send_message(chat_id=CHAT_ID, text="🛑 Sell SL hit (spread-adjusted)!")
                        log_signal("Sell", close_price, None, None, None, current_tp, current_sl, trade_status="SL Hit", update_last=True)
                        trade_open = False
                        return

                msg = (
                    f"📊 Trade Open: {current_action}\n"
                    f"💰 Entry Price: {trade_entry_price:.2f}\n"
                    f"🎯 TP: {current_tp:.2f} | 🛑 SL: {current_sl:.2f}\n"
                    f"📉 Current Price: {close_price:.2f}"
                )
                update.message.reply_text(msg)
                return

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

            if action_name != "Hold":
                trade_open = True
                current_action = action_name
                current_tp = tp
                current_sl = sl
                trade_entry_price = close_price
                trade_timestamp = datetime.datetime.now()

            tp_sl_line = f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}" if tp and sl else "📌 No TP/SL — holding"

            msg = (
                f"📊 Live Signal: {action_name}\n"
                f"💰 Price: {close_price:.2f}\n"
                f"📈 RSI: {rsi:.2f} | MACD: {macd:.4f} | EMA50: {ema_50:.2f}\n"
                f"{tp_sl_line}"
            )
            update.message.reply_text(msg)

            if action_name != "Hold":
                log_signal(action_name, close_price, rsi, macd, ema_50, tp, sl, source="manual-live")

    except Exception as e:
        update.message.reply_text(f"❌ Error during prediction: {e}")


# === /export ===
def export_log(update: Update, context: CallbackContext):
    try:
        log_file_path = "signal_log.csv"
        if os.path.exists(log_file_path):
            with open(log_file_path, "rb") as f:
                bot.send_document(chat_id=CHAT_ID, document=InputFile(f, filename="signal_log.csv"))
            update.message.reply_text("✅ The signal log has been exported.")
        else:
            update.message.reply_text("❌ No signal log file found.")
    except Exception as e:
        update.message.reply_text(f"❌ Error during log export: {e}")

# === Auto Signal ===
def check_market_and_send_signal():
    global trade_open, current_action, current_tp, current_sl, trade_entry_price, trade_timestamp
    try:
        with trade_lock:
            df_live = add_indicators(fetch_data_twelvedata())
            latest = df_live.iloc[-1]
            close_price = latest['close']
            high = latest['high']
            low = latest['low']
            rsi = latest['rsi']
            macd = latest['macd']
            ema_50 = latest['ema_50']
            spread = 0.3

            if trade_open:
                if current_action == "Buy":
                    if high >= current_tp - spread:
                        bot.send_message(chat_id=CHAT_ID, text="🎯 Buy TP hit (spread-adjusted)!")
                        log_signal("Buy", close_price, None, None, None, current_tp, current_sl, trade_status="TP Hit", update_last=True)
                        trade_open = False
                        return
                    elif low <= current_sl + spread:
                        bot.send_message(chat_id=CHAT_ID, text="🛑 Buy SL hit (spread-adjusted)!")
                        log_signal("Buy", close_price, None, None, None, current_tp, current_sl, trade_status="SL Hit", update_last=True)
                        trade_open = False
                        return

                elif current_action == "Sell":
                    if low <= current_tp + spread:
                        bot.send_message(chat_id=CHAT_ID, text="🎯 Sell TP hit (spread-adjusted)!")
                        log_signal("Sell", close_price, None, None, None, current_tp, current_sl, trade_status="TP Hit", update_last=True)
                        trade_open = False
                        return
                    elif high >= current_sl - spread:
                        bot.send_message(chat_id=CHAT_ID, text="🛑 Sell SL hit (spread-adjusted)!")
                        log_signal("Sell", close_price, None, None, None, current_tp, current_sl, trade_status="SL Hit", update_last=True)
                        trade_open = False
                        return

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

            if action_name != "Hold" and not trade_open:
                trade_open = True
                current_action = action_name
                current_tp = tp
                current_sl = sl
                trade_entry_price = close_price
                trade_timestamp = datetime.datetime.now()

                msg = (
                    f"📊 Auto Signal: {action_name}\n"
                    f"💰 Price: {close_price:.2f}\n"
                    f"🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f}"
                )
                bot.send_message(chat_id=CHAT_ID, text=msg)
                log_signal(action_name, close_price, rsi, macd, ema_50, tp, sl, source="auto")

    except Exception as e:
        bot.send_message(chat_id=CHAT_ID, text=f"❌ Error during auto signal: {e}")

def run_scheduler():
    schedule.every(15).minutes.do(check_market_and_send_signal)
    while True:
        schedule.run_pending()
        time.sleep(1)

def clear_log_file():
    log_file_path = "signal_log.csv"
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            print(f"✅ Log file '{log_file_path}' cleared.")
        except Exception as e:
            print(f"❌ Error while clearing log file: {e}")

if __name__ == "__main__":
    clear_log_file()

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("export", export_log))

    bot.set_webhook(f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/{TELEGRAM_TOKEN}")

    threading.Thread(target=run_scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
