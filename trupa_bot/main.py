import os
import requests
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import telegram
from telegram.ext import Dispatcher, CommandHandler
from xgboost import XGBClassifier
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === ENV VARS ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
RENDER_HOST = os.getenv("RENDER_EXTERNAL_HOSTNAME")
PORT = int(os.environ.get("PORT", 10000))

# === Globals ===
model = None
last_report = "Not trained yet."
data_cache = None

# === Flask App Init ===
app = Flask(__name__)
bot = telegram.Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, workers=0)

# === Data Processing ===
def fetch_data(symbol="XAU/USD", interval="1min", apikey=None):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={apikey}&outputsize=5000"
    response = requests.get(url)
    data = response.json()
    if "values" not in data:
        raise Exception("❌ Data fetch failed.")
    df = pd.DataFrame(data["values"])
    df = df.rename(columns=str.lower)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def compute_features(df):
    df['range'] = df['high'] - df['low']
    df['ma10'] = df['close'].rolling(10).mean()
    df['trend'] = (df['close'] > df['ma10']).astype(int)
    df['momentum_score'] = (df['close'] > df['close'].rolling(20).mean()).astype(int)
    df['breakout_score'] = (df['high'] > df['high'].rolling(50).max()).astype(int)
    df['compression'] = (df['range'] < 0.5 * df['range'].rolling(20).mean()).astype(int)
    df['atr'] = df['range'].rolling(14).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal

    df['body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    df['structure'] = np.where(df['body_ratio'] > 0.6, 1, np.where(df['body_ratio'] < 0.3, -1, 0))
    df['position_ratio'] = (df['close'] - df['low']) / df['range'].replace(0, np.nan)
    df['sr_position'] = pd.Series(np.select([
        df['position_ratio'] > 0.66,
        df['position_ratio'] < 0.33
    ], ['top', 'bottom'], default='middle'), index=df.index).map({'top': 1, 'middle': 0, 'bottom': -1})
    df['bb_squeeze'] = (df['range'].rolling(20).std() < df['range'].rolling(50).std().quantile(0.25)).astype(int)
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
        sl = entry - (0.2 * 10.0)
        tp = entry + (0.3 * 10.0)
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
                              'structure', 'sr_position', 'rsi', 'macd_hist', 'bb_squeeze', 'body_strength']].to_dict()
            entry_data.update({'entry': entry, 'tp': tp, 'sl': sl, 'label': label})
            entries.append(entry_data)
    return pd.DataFrame(entries)

def train_classifier(data):
    global model, last_report
    features = ['momentum_score', 'breakout_score', 'compression', 'trend',
                'structure', 'sr_position', 'rsi', 'macd_hist', 'bb_squeeze', 'body_strength']
    X = data[features]
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                          use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=3)
    last_report = f"🧠 Model trained {datetime.now().strftime('%H:%M:%S')}\n\n{report}"

def retrain():
    try:
        df = fetch_data(apikey=TWELVE_DATA_API_KEY)
        df = compute_features(df)
        data = simulate_trades(df)
        train_classifier(data)
        print("[Retrain] Success")
    except Exception as e:
        print(f"[Retrain] Failed: {e}")

# === Telegram Commands ===
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="📡 SignalBot is live. Use /monitor to check model performance.")

def monitor(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=last_report)

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("monitor", monitor))

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

@app.route("/", methods=["GET"])
def health():
    return "✅ SignalBot running with auto-retraining.", 200

# === Schedule Auto-Retraining ===
scheduler = BackgroundScheduler()
scheduler.add_job(retrain, 'interval', minutes=3)
scheduler.start()

# === Entry Point ===
if __name__ == "__main__":
    retrain()
    if RENDER_HOST:
        bot.set_webhook(f"https://{RENDER_HOST}/{TELEGRAM_TOKEN}")
    app.run(host="0.0.0.0", port=PORT)
  
