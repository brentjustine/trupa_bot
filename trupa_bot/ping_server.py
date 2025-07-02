from flask import Flask
import threading
import main  # this runs your trading bot

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Bot is alive."

if __name__ == '__main__':
    # Run your bot in a background thread
    threading.Thread(target=main.main).start()
    # Start the web server
    app.run(host='0.0.0.0', port=10000)
