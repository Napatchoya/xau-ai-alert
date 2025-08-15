import requests
import pandas as pd
import ta
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, Response
from threading import Thread

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

model = None  # โหลดโมเดลตอนใช้งานจริง
last_signal = None
last_sent_hour = None

app = Flask(__name__)

def load_model():
    global model
    if model is None:
        model = joblib.load("xau_model.pkl")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.status_code
    except requests.exceptions.RequestException as e:
        print(f"Telegram send error: {e}")
        return None

def get_latest_xau():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=50&apikey={API_KEY}"
    res = requests.get(url, timeout=10).json()
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
        load_model()
        latest = get_latest_xau()
        X_live = latest[['rsi', 'ema', 'price_change']]
        price = latest['close'].iloc[0]
        utc_time = latest.index[0]
        thai_time = utc_time.tz_localize('UTC').astimezone(ZoneInfo("Asia/Bangkok"))
        timestamp = thai_time.strftime('%Y-%m-%d %H:%M')

        if X_live.isnull().values.any():
            msg = (
                f"ตอนนี้ วันที่ {timestamp} \n"
                f"XAUUSD TF H1 ราคาปิดที่ {price:,.2f}$\n"
                f"BOT ยังไม่สามารถทำนายจุดที่จะทำการซื้อขายได้"
            )
            send_telegram(msg)
            return msg

        prediction = model.predict(X_live)[0]

        if prediction == 1:  # BUY
            signal = "📈 BUY"
            tp1 = price * 1.002
            tp2 = price * 1.004
            tp3 = price * 1.006
            sl = price * 0.998
            reason = "RSI ต่ำกว่า 30 และราคาปิดสูงกว่า EMA 10 บ่งชี้แนวโน้มขาขึ้น"
        else:  # SELL
            signal = "📉 SELL"
            tp1 = price * 0.998
            tp2 = price * 0.996
            tp3 = price * 0.994
            sl = price * 1.002
            reason = "RSI สูงกว่า 70 และราคาปิดต่ำกว่า EMA 10 บ่งชี้แนวโน้มขาลง"

        msg = (
            f"ตอนนี้ วันที่ {timestamp} \n"
            f"XAUUSD TF H1 ราคาปิดที่ {price:,.2f}$\n"
            f"BOT สามารถทำนายจุดที่จะทำการซื้อขายได้\n"
            f"เหตุผลเพราะ {reason}\n"
            f"ขอให้เข้า {signal} ที่ราคา {price:,.2f}$\n"
            f"🎯 TP1: {tp1:,.2f}$\n"
            f"🎯 TP2: {tp2:,.2f}$\n"
            f"🎯 TP3: {tp3:,.2f}$\n"
            f"🛑 SL: {sl:,.2f}$"
        )
        send_telegram(msg)
        last_signal = signal
        return msg

    except Exception as e:
        err_msg = f"❌ ERROR: {e}"
        print(err_msg)
        return err_msg

@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    return Response("OK", status=200, headers={"Content-Type": "text/plain"})

@app.route('/test-telegram')
def test_telegram():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"✅ ทดสอบส่งข้อความจาก AI Bot @ {now}"
    status = send_telegram(message)
    return jsonify({"status": status, "message": message})

@app.route('/run-ai')
def run_ai():
    global last_sent_hour
    now = datetime.now(ZoneInfo("Asia/Bangkok"))
    current_hour = now.hour

    if current_hour != last_sent_hour:
        last_sent_hour = current_hour
        Thread(target=lambda: print(run_ai_once())).start()
        return jsonify({"status": "✅ ส่งข้อความรอบต้นชั่วโมง", "time": now.strftime("%Y-%m-%d %H:%M")})
    else:
        return jsonify({"status": "⏳ รอรอบต้นชั่วโมงถัดไป", "time": now.strftime("%Y-%m-%d %H:%M")})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
