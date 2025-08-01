import requests
import pandas as pd
import time
import ta
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo  # ใช้ได้ใน Python 3.9+
from flask import Flask, jsonify
from threading import Thread

load_dotenv()

# 🔐 ใส่ TOKEN และ CHAT_ID ของ Telegram Bot
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

# โหลดโมเดล AI
model = joblib.load("xau_model.pkl")

app = Flask(__name__)
last_signal = None

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.status_code

def get_latest_xau():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=50&apikey={API_KEY}"
    res = requests.get(url).json()

    df = pd.DataFrame(res['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    now_utc = pd.Timestamp.utcnow().replace(tzinfo=None)
    df = df[df['datetime'] <= now_utc]
    df = df.sort_values('datetime')
    df.set_index('datetime', inplace=True)
    df['close'] = df['close'].astype(float)

    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    df['price_change'] = df['close'].pct_change()
    df.dropna(subset=['rsi', 'ema', 'price_change'], inplace=True)

    return df.tail(1)

def run_ai_once():
    global last_signal
    try:
        latest = get_latest_xau()
        X_live = latest[['rsi', 'ema', 'price_change']]
        prediction = model.predict(X_live)[0]
        signal = "📈 BUY" if prediction == 1 else "📉 SELL"

        utc_time = latest.index[0]
        thai_time = utc_time.tz_localize('UTC').astimezone(ZoneInfo("Asia/Bangkok"))
        timestamp = thai_time.strftime('%Y-%m-%d %H:%M')

        if signal != last_signal:
            msg = f"{signal} XAU/USD @ {timestamp}"
            send_telegram(msg)
            last_signal = signal
            return f"🔔 ส่ง Telegram: {msg}"
        else:
            return f"✅ ยังไม่มีสัญญาณใหม่ ({timestamp})"
    except Exception as e:
        return f"❌ ERROR: {e}"

@app.route('/')
def health():
    return Response('OK', status=200,mimetype='text/plain')

@app.route('/run-ai')
def run_ai():
    def task():
        result = run_ai_once()
        print(result)

    Thread(target=task).start()
    return jsonify({"status": "🔁 AI started on-demand."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
