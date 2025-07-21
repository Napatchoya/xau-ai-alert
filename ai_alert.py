import requests
import pandas as pd
import time
import ta
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo  # ใช้ได้ใน Python 3.9+
from flask import Flask
from threading import Thread

load_dotenv()

# 🔐 ใส่ TOKEN และ CHAT_ID ของ Telegram Bot
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")     
  # เช่น 123456789 หรือ -987654321 (ถ้าเป็นกลุ่ม)

# โหลดโมเดล AI
model = joblib.load("xau_model.pkl")

# ✅ ฟังก์ชันส่งข้อความเข้า Telegram
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, data=data)
    return response.status_code

# ✅ ดึงข้อมูล XAU/USD แบบ real-time จาก TwelveData
def get_latest_xau():
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=50&apikey={API_KEY}"
    res = requests.get(url).json()

    df = pd.DataFrame(res['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])  # <--- ต้องแปลงก่อน

     # 🔍 ฟิลเตอร์ข้อมูลล่วงหน้าออก
    now_utc =      pd.Timestamp.utcnow().replace(tzinfo=None)  # <-- ทำให้ tz-naive
    df = df[df['datetime'] <= now_utc]  # ลบแท่งที่ล่วงหน้า

    df = df.sort_values('datetime')
    df.set_index('datetime', inplace=True)
    df['close'] = df['close'].astype(float)

    # เพิ่ม indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    df['price_change'] = df['close'].pct_change()
    df.dropna(subset=['rsi', 'ema', 'price_change'], inplace=True)

    return df.tail(1)


# ✅ Web Server สำหรับ UptimeRobot
app = Flask('')

@app.route('/')
def home():
    return "✅ Bot is running.", 200

def run_web():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run_web)
    t.start()

# ✅ เริ่มต้นเว็บเซิร์ฟเวอร์
keep_alive()

# ✅ วนลูปตรวจสอบสัญญาณใหม่และส่ง Telegram
last_signal = None
print("🔁 เริ่มทำงาน AI Real-time XAUUSD...")

while True:
    try:
        latest = get_latest_xau()
        X_live = latest[['rsi', 'ema', 'price_change']]
        print("🔎 ข้อมูลที่ใช้ทำนาย X_live:\n", X_live)
        print("🕒 ตอนนี้ (UTC):", pd.Timestamp.utcnow())
        print("📅 ข้อมูลล่าสุดจาก API:", latest.index[-1])
        prediction = model.predict(X_live)[0]
        print("✅ ผลลัพธ์ที่โมเดลทำนาย:", prediction)
        signal = "📈 BUY" if prediction == 1 else "📉 SELL"
        utc_time = latest.index[0]
        thai_time = utc_time.tz_localize('UTC').astimezone(ZoneInfo("Asia/Bangkok"))
        timestamp = thai_time.strftime('%Y-%m-%d %H:%M')

        if signal != last_signal:
            msg = f"{signal} XAU/USD @ {timestamp}"
            send_telegram(msg)
            print("🔔 ส่ง Telegram:", msg)
            last_signal = signal
        else:
            print(f"✅ ยังไม่มีสัญญาณใหม่ ({timestamp})")

    except Exception as e:
        print("❌ ERROR:", e)

    time.sleep(60 * 60)  # เช็คทุก 1 ชั่วโมง 
