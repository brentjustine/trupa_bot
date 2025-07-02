from flask import Flask
import threading
import os
import main  # this runs your trading bot

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is alive."

if __name__ == '__main__':
    # Run bot in background thread
    threading.Thread(target=main.main).start()

    # Run Flask web server for Render to ping
    PORT = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=PORT)
