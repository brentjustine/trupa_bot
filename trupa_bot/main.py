import os
import time
import threading
import datetime
import schedule
import gdown
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

# === Download model files from Google Drive if missing ===
def download_model_files():
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

# === Load Model and Environment ===
download_model_files()
df = add_indicators(fetch_data_twelvedata(interval="15min"))

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

# === Trade Memory State ===
trade_state = {
    "open_position": None,
    "entry_price": None,
    "tp": None,
    "sl": None
}

# === Telegram Bot Commands ===
def start(update: Update, context: CallbackContext):
    msg = (
        "🤖 Gold Trading Bot Active!\n"
        "Use /predict to get a trading signal.\n"
        f"Your Chat ID: `{update.message.chat_id}`"
    )
    update.message.reply_text(msg, parse_mode="Markdown")

def predict(update: Update, context: CallbackContext):
    global trade_state
    df_live = add_indicators(fetch_data_twelvedata(interval="15min"))
    latest = df_live.iloc[-1]
    high = latest["high"]
    low = latest["low"]
    close_price = latest["close"]

    # If in a trade, check TP/SL
    if trade_state["open_position"]:
        pos = trade_state["open_position"]
        if (pos == "Buy" and (high >= trade_state["tp"] or low <= trade_state["sl"])) or \
           (pos == "Sell" and (low <= trade_state["tp"] or high >= trade_state["sl"])):
            result = "✅ TP hit" if (pos == "Buy" and high >= trade_state["tp"]) or \
                                     (pos == "Sell" and low <= trade_state["tp"]) else "🛑 SL hit"
            trade_state = {"open_position": None, "entry_price": None, "tp": None, "sl": None}
            update.message.reply_text(f"📤 Trade closed: {result}")
        else:
            update.message.reply_text("⏳ Trade ongoing. Waiting for TP/SL.")
            return

    obs = vec_env.reset()
    action, _ = model.predict(obs)
    action_name = ["Hold", "Buy", "Sell"][action[0]]

    rsi = latest["rsi"]
    macd = latest["macd"]
    ema = latest["ema_20"]

    if action_name in ["Buy", "Sell"]:
        tp = close_price + 4 if action_name == "Buy" else close_price - 4
        sl = close_price - 3 if action_name == "Buy" else close_price + 3
        trade_state = {
            "open_position": action_name,
            "entry_price": close_price,
            "tp": tp,
            "sl": sl
        }
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

# === Auto Signal Scheduler ===
last_signal = {"timestamp": None}

def check_market_and_send_signal():
    global trade_state, last_signal
    try:
        df_live = add_indicators(fetch_data_twelvedata(interval="15min"))
        latest = df_live.iloc[-1]
        high = latest["high"]
        low = latest["low"]
        close_price = latest["close"]
        current_time = latest["datetime"]

        # Check for open trade TP/SL
        if trade_state["open_position"]:
            pos = trade_state["open_position"]
            if (pos == "Buy" and (high >= trade_state["tp"] or low <= trade_state["sl"])) or \
               (pos == "Sell" and (low <= trade_state["tp"] or high >= trade_state["sl"])):
                result = "✅ TP hit" if (pos == "Buy" and high >= trade_state["tp"]) or \
                                         (pos == "Sell" and low <= trade_state["tp"]) else "🛑 SL hit"
                bot.send_message(chat_id=CHAT_ID, text=f"📤 Auto trade closed: {result}")
                trade_state = {"open_position": None, "entry_price": None, "tp": None, "sl": None}
                return  # Wait for next entry opportunity
            else:
                return  # Trade is ongoing

        if last_signal["timestamp"] == current_time:
            return  # already checked this interval

        obs = vec_env.reset()
        action, _ = model.predict(obs)
        action_name = ["Hold", "Buy", "Sell"][action[0]]

        rsi = latest["rsi"]
        macd = latest["macd"]
        ema = latest["ema_20"]

        if action_name in ["Buy", "Sell"]:
            tp = close_price + 4 if action_name == "Buy" else close_price - 4
            sl = close_price - 3 if action_name == "Buy" else close_price + 3
            trade_state = {
                "open_position": action_name,
                "entry_price": close_price,
                "tp": tp,
                "sl": sl
            }
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
    schedule.every(15).minutes.do(check_market_and_send_signal)
    while True:
        schedule.run_pending()
        time.sleep(10)

# === Main Bot and Flask Server Launcher ===
def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("export", export_log))

    # Start the scheduler thread
    threading.Thread(target=run_scheduler, daemon=True).start()

    # Start Telegram polling
    updater.start_polling()
    updater.idle()

# === Entry Point ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port),
        daemon=True
    ).start()
    main()
