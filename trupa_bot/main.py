import os
import requests
import numpy as np
import pandas as pd
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import telegram
from telegram.ext import Dispatcher, CommandHandler
from datetime import datetime
import pytz

# === ENV VARS ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
RENDER_HOST = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.environ.get("PORT", 10000))

# === Globals ===
latest_model = None
latest_stats = {}

# === Flask App Init ===
app = Flask(__name__)
bot = telegram.Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

# === Fetch Live Data ===
def fetch_data(symbol="XAU/USD", interval="1min", apikey=None, outputsize=5000):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={apikey}&outputsize={outputsize}"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise Exception("❌ Data fetch failed.")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns=str.lower)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["datetime"])
    return df.sort_values("timestamp").reset_index(drop=True)

# === Feature Engineering ===
def compute_features(df):
    df['range'] = df['high'] - df['low']
    df['ma10'] = df['close'].rolling(10).mean()
    df['trend'] = (df['close'] > df['ma10']).astype(int)
    df['momentum'] = (df['close'] > df['close'].shift(5)).astype(int)
    df['avg_range_10'] = df['range'].rolling(10).mean()
    df['volatility'] = (df['range'] > df['avg_range_10']).astype(int)
    df['body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    df['structure'] = np.where(df['body_ratio'] > 0.6, 1, np.where(df['body_ratio'] < 0.3, -1, 0))
    df['position_ratio'] = (df['close'] - df['low']) / df['range'].replace(0, np.nan)
    df['sr_position'] = pd.Series(np.select([
        df['position_ratio'] > 0.66,
        df['position_ratio'] < 0.33
    ], ['top', 'bottom'], default='middle'), index=df.index).map({'top': 1, 'middle': 0, 'bottom': -1})
    df['compression'] = (df['range'] < 0.5 * df['range'].rolling(20).mean()).astype(int)
    df['momentum_score'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)
    df['breakout_score'] = (df['high'] > df['high'].rolling(50).max()).astype(int)
    df['atr'] = df['range'].rolling(14).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    stddev = df['close'].rolling(20).std()
    upper_bb = df['close'].rolling(20).mean() + 2 * stddev
    lower_bb = df['close'].rolling(20).mean() - 2 * stddev
    df['bb_width'] = upper_bb - lower_bb
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)).astype(int)
    df['body_strength'] = df['body'].rolling(20).apply(lambda x: (x.rank().iloc[-1]) / len(x), raw=False)
    return df

# === Trade Simulation and Labeling ===
def simulate_trades(df):
    entries = []
    for i in range(50, len(df) - 50):
        row = df.iloc[i]
        atr = row['atr']
        if np.isnan(atr) or atr == 0:
            continue
        entry = row['close']
        sl = entry - 2.0
        tp = entry + 3.0
        label = None
        for j in range(i + 1, i + 50):
            future = df.iloc[j]
            if future['low'] <= sl:
                label = 0
                break
            if future['high'] >= tp:
                label = 1
                break
        if label is not None:
            entry_data = row[['timestamp', 'momentum_score', 'breakout_score', 'compression', 'trend', 'structure', 'sr_position', 'volatility', 'rsi', 'macd_hist', 'bb_squeeze', 'body_strength']].to_dict()
            entry_data.update({'entry': entry, 'tp': tp, 'sl': sl, 'label': label})
            entries.append(entry_data)
    return pd.DataFrame(entries)

# === Training ===
def train_model():
    global latest_model, latest_stats
    df = fetch_data(apikey=TWELVE_DATA_API_KEY)
    df = compute_features(df)
    trade_data = simulate_trades(df)
    features = ['momentum_score', 'breakout_score', 'compression', 'trend', 'structure', 'sr_position', 'volatility', 'rsi', 'macd_hist', 'bb_squeeze', 'body_strength']
    X = trade_data[features]
    y = trade_data['label']
    model = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss', use_label_encoder=False)
    model.fit(X, y)
    predictions = model.predict(X)
    winrate = (predictions == y).mean()
    latest_model = model
    latest_stats = {
        "winrate": round(float(winrate) * 100, 2),
        "trades": len(trade_data),
        "high_conf": int((model.predict_proba(X)[:, 1] > 0.6).sum())
    }

# === Signal Generation ===
def get_signal():
    if latest_model is None:
        return None
    df = fetch_data(apikey=TWELVE_DATA_API_KEY, outputsize=150)
    df = compute_features(df)
    row = df.iloc[-1]
    features = ['momentum_score', 'breakout_score', 'compression', 'trend', 'structure', 'sr_position', 'volatility', 'rsi', 'macd_hist', 'bb_squeeze', 'body_strength']
    X_live = row[features].values.reshape(1, -1)
    prob = latest_model.predict_proba(X_live)[0, 1]
    direction = "buy" if row['trend'] == 1 else "sell"
    tp = row['close'] + 3.0 if direction == "buy" else row['close'] - 3.0
    sl = row['close'] - 2.0 if direction == "buy" else row['close'] + 2.0
    return {
        "timestamp": str(row['timestamp']),
        "direction": direction,
        "entry": round(row['close'], 2),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "win_prob": round(prob * 100, 1)
    } if prob > 0.6 else None

# === Telegram Commands ===
def start(update, context):
    msg = (
        "📱 Adaptive AI SignalBot Activated!\n\n"
        "This bot uses XGBoost with dynamic retraining every 3 minutes based on the latest 5000 candles.\n\n"
        "• TP = $3 | SL = $2\n"
        "• /predict - Get the latest signal\n"
        "• /monitor - Check model performance"
    )
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode="Markdown")

def predict(update, context):
    try:
        signal = get_signal()
        if signal:
            msg = (
                f"✅ Signal at {signal['timestamp']}\n"
                f"Direction: {signal['direction']}\n"
                f"Entry: {signal['entry']}\n"
                f"TP: {signal['tp']} | SL: {signal['sl']}\n"
                f"Confidence: {signal['win_prob']}%"
            )
        else:
            msg = "🤖 No high-confidence signal right now."
        context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
    except Exception as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Error: {e}")

def monitor(update, context):
    if not latest_stats:
        context.bot.send_message(chat_id=update.effective_chat.id, text="ℹ️ No training yet.")
        return
    msg = (
        f"📊 Model Monitor\n"
        f"Trades Simulated: {latest_stats['trades']}\n"
        f"High Confidence: {latest_stats['high_conf']}\n"
        f"Winrate: {latest_stats['winrate']}%"
    )
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode="Markdown")

# === Register Telegram Handlers ===
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("predict", predict))
dispatcher.add_handler(CommandHandler("monitor", monitor))

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return "✅ SignalBot is Running", 200

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

# === Scheduler ===
scheduler = BackgroundScheduler(timezone=pytz.utc)
scheduler.add_job(train_model, 'interval', minutes=3)
scheduler.start()

if __name__ == "__main__":
    if RENDER_HOST:
        bot.set_webhook(f"https://{RENDER_HOST}/{TELEGRAM_TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
    
