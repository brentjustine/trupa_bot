import os
import pandas as pd
import numpy as np
import requests
import telegram
from flask import Flask, request
from telegram.ext import Dispatcher, CommandHandler
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import pytz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# === ENVIRONMENT ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
RENDER_HOST = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.environ.get("PORT", 10000))

# === GLOBAL STATE ===
model = None
last_report = ""

# === APP INIT ===
app = Flask(__name__)
bot = telegram.Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

# === Fetch & Process ===
def fetch_data(symbol="XAU/USD", interval="1min", size=5000):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVE_DATA_API_KEY}&outputsize={size}"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise ValueError("Data fetch failed")
    df = pd.DataFrame(data["values"])
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"datetime": "timestamp"})
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)

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
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df['macd_hist'] = macd - signal
    stddev = df['close'].rolling(20).std()
    upper_bb = df['close'].rolling(20).mean() + 2 * stddev
    lower_bb = df['close'].rolling(20).mean() - 2 * stddev
    df['bb_width'] = upper_bb - lower_bb
    df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)).astype(int)
    df['body_strength'] = df['body'].rolling(20).apply(lambda x: (x.rank().iloc[-1]) / len(x), raw=False)
    return df

def simulate_trades(df):
    entries = []
    for i in range(50, len(df) - 50):
        row = df.iloc[i]
        atr = row['atr']
        if np.isnan(atr) or atr == 0:
            continue
        entry = row['close']
        sl = entry - 2
        tp = entry + 3
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
            entry_data = row[['timestamp', 'momentum_score', 'breakout_score', 'compression', 'trend',
                              'structure', 'sr_position', 'volatility', 'rsi', 'macd_hist',
                              'bb_squeeze', 'body_strength']].to_dict()
            entry_data.update({'entry': entry, 'tp': tp, 'sl': sl, 'label': label})
            entries.append(entry_data)
    return pd.DataFrame(entries)

def train_model():
    global model, last_report
    df = fetch_data()
    df = compute_features(df)
    trades = simulate_trades(df)
    if len(trades) < 100:
        return
    features = ['momentum_score', 'breakout_score', 'compression', 'trend',
                'structure', 'sr_position', 'volatility', 'rsi', 'macd_hist',
                'bb_squeeze', 'body_strength']
    X = trades[features]
    y = trades['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    model = XGBClassifier(n_estimators=50, max_depth=6, scale_pos_weight=(len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1e-6)),
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    winrate = (y_pred[y_pred == 1] == y_test[y_pred == 1]).mean()
    last_report = f"✅ Model retrained.\nSamples: {len(trades)}\nAccuracy: {acc:.2%}\nWinrate (pred=1): {winrate:.2%}"

# === APScheduler (fixed with pytz) ===
scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Manila'))
scheduler.add_job(train_model, trigger=IntervalTrigger(minutes=3))
scheduler.start()

# === Bot Commands ===
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="🤖 Model-based XAU/USD SignalBot is running.\nUse /predict for signals.\nUse /monitor for model status.")

def monitor(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=last_report or "📉 Model not trained yet.")

def predict(update, context):
    if model is None:
        context.bot.send_message(chat_id=update.effective_chat.id, text="❌ Model not trained yet.")
        return
    df = fetch_data()
    df = compute_features(df)
    latest = df.iloc[-1]
    features = ['momentum_score', 'breakout_score', 'compression', 'trend',
                'structure', 'sr_position', 'volatility', 'rsi', 'macd_hist',
                'bb_squeeze', 'body_strength']
    proba = model.predict_proba(latest[features].values.reshape(1, -1))[0][1]
    msg = f"📊 XAU/USD Prediction\nTimestamp: {latest['timestamp']}\nWin Prob: {proba:.2%}"
    context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("predict", predict))
dispatcher.add_handler(CommandHandler("monitor", monitor))

# === Routes ===
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

@app.route("/", methods=["GET"])
def root():
    return "✅ XAU/USD Model-Based SignalBot is Live"

# === Entrypoint ===
if __name__ == "__main__":
    if RENDER_HOST:
        bot.set_webhook(f"https://{RENDER_HOST}/{TELEGRAM_TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
