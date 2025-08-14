import requests
import pandas as pd
import time
import ta
import joblib
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, Response
from threading import Thread

load_dotenv()

# 🔐 TOKEN และ CHAT_ID ของ Telegram Bot
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

# โหลดโมเดล AI
model = joblib.load("xau_model.pkl")

app = Flask(__name__)
last_signal = None

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
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

        try:
            prediction = model.predict(X_live)[0]
            can_predict = True
        except Exception:
            can_predict = False

        price = latest['close'].iloc[0]
        utc_time = latest.index[0]
        thai_time = utc_time.tz_localize('UTC').astimezone(ZoneInfo("Asia/Bangkok"))
        timestamp_str = thai_time.strftime('%d %b %Y เวลา %H:%M น')
        rsi = latest['rsi'].iloc[0]
        ema = latest['ema'].iloc[0]

        if not can_predict:
            msg = (
                f"ตอนนี้ วันที่ {timestamp_str}\n"
                f"XAUUSD TF H1 ราคาปิดที่ {price:,.2f}$\n"
                f"BOT ยังไม่สามารถทำนายจุดที่จะทำการซื้อขายได้"
            )
            send_telegram(msg)
            return msg

        signal = "BUY" if prediction == 1 else "SELL"

        if signal == "BUY":
            tp1 = price * 1.002
            tp2 = price * 1.004
            tp3 = price * 1.006
            sl = price * 0.998
            reason = f"RSI {rsi:.2f} > 50 และราคาปิดเหนือ EMA {ema:.2f}"
        else:
            tp1 = price * 0.998
            tp2 = price * 0.996
            tp3 = price * 0.994
            sl = price * 1.002
            reason = f"RSI {rsi:.2f} < 50 และราคาปิดต่ำกว่า EMA {ema:.2f}"

        msg = (
            f"ตอนนี้ วันที่ {timestamp_str}\n"
            f"XAUUSD TF H1 ราคาปิดที่ {price:,.2f}$\n"
            f"BOT สามารถทำนายจุดที่จะทำการซื้อขายได้\n"
            f"เหตุผล: {reason}\n"
            f"📌 ขอให้เข้า {signal} ที่ราคา {price:,.2f}$\n"
            f"🎯 TP1: {tp1:,.2f}\n"
            f"🎯 TP2: {tp2:,.2f}\n"
            f"🎯 TP3: {tp3:,.2f}\n"
            f"🛑 SL: {sl:,.2f}"
        )

        send_telegram(msg)
        return msg

    except Exception as e:
        return f"❌ ERROR: {e}"

@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    return Response("OK", status=200, headers={"Content-Type": "text/plain", "Cache-Control": "no-cache"})

@app.route('/test-telegram')
def test_telegram():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"✅ ทดสอบส่งข้อความจาก AI Bot @ {now}"
    status = send_telegram(message)
    return jsonify({"status": status, "message": message})

@app.route('/run-ai')
def run_ai():
    def task():
        result = run_ai_once()
        print(result)
    Thread(target=task).start()
    return jsonify({"status": "🔁 AI started on-demand."})

def hourly_task_exact():
    while True:
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)  # เริ่ม +5 วิ กันพลาด
        wait_seconds = (next_hour - now).total_seconds()
        time.sleep(wait_seconds)
        run_ai_once()

if __name__ == '__main__':
    Thread(target=hourly_task_exact, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
