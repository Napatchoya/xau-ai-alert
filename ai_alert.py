import requests
import pandas as pd
import ta
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo  # ใช้ได้ใน Python 3.9+
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
last_sent_hour = None  # เก็บชั่วโมงล่าสุดที่ส่งแล้ว

# ฟังก์ชันส่งข้อความไป Telegram
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.status_code

# ดึงข้อมูล XAU/USD ล่าสุด
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

# รัน AI และส่งข้อความตามเงื่อนไข
def run_ai_once():
    global last_signal
    try:
        latest = get_latest_xau()
        X_live = latest[['rsi', 'ema', 'price_change']]
        price = latest['close'].iloc[0]
        utc_time = latest.index[0]
        thai_time = utc_time.tz_localize('UTC').astimezone(ZoneInfo("Asia/Bangkok"))
        timestamp = thai_time.strftime('%Y-%m-%d %H:%M')

        # ถ้าไม่มีข้อมูลเพียงพอ
        if X_live.isnull().values.any():
            msg = (
                f"ตอนนี้ วันที่ {timestamp} \n"
                f"XAUUSD TF H1 ราคาปิดที่ {price:,.2f}$\n"
                f"BOT ยังไม่สามารถทำนายจุดที่จะทำการซื้อขายได้"
            )
            send_telegram(msg)
            return msg

        # ทำนาย
        prediction = model.predict(X_live)[0]

        # คำนวณ TP/SL และเหตุผล
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
        return f"❌ ERROR: {e}"

# Health check
@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    return Response("OK", status=200, headers={
        "Content-Type": "text/plain",
        "Cache-Control": "no-cache"
    })

# ทดสอบส่งข้อความ Telegram
@app.route('/test-telegram')
def test_telegram():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"✅ ทดสอบส่งข้อความจาก AI Bot @ {now}"
    status = send_telegram(message)
    return jsonify({"status": status, "message": message})

# เรียก AI ทุกต้นชั่วโมง
@app.route('/run-ai')
def run_ai():
    global last_sent_hour
    now = datetime.now(ZoneInfo("Asia/Bangkok"))
    current_hour = now.hour

    if current_hour != last_sent_hour:
        last_sent_hour = current_hour
        def task():
            result = run_ai_once()
            print(result)
        Thread(target=task).start()
        return jsonify({"status": "✅ ส่งข้อความรอบต้นชั่วโมง", "time": now.strftime("%Y-%m-%d %H:%M")})
    else:
        return jsonify({"status": "⏳ รอรอบต้นชั่วโมงถัดไป", "time": now.strftime("%Y-%m-%d %H:%M")})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
