
import requests
import pandas as pd
import time
import ta
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv

load dotenv()

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
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval=1h&outputsize=10&apikey={API_KEY}"
    res = requests.get(url).json()

    df = pd.DataFrame(res['values'])
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    df['close'] = df['close'].astype(float)

    # เพิ่ม indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    df['price_change'] = df['close'].pct_change()
    df.dropna(inplace=True)

    return df.tail(1)

# ✅ วนลูปตรวจสอบสัญญาณใหม่และส่ง Telegram
last_signal = None
print("🔁 เริ่มทำงาน AI Real-time XAUUSD...")

while True:
    try:
        latest = get_latest_xau()
        X_live = latest[['rsi', 'ema', 'price_change']]
        prediction = model.predict(X_live)[0]
        signal = "📈 BUY" if prediction == 1 else "📉 SELL"
        timestamp = latest.index[0].strftime('%Y-%m-%d %H:%M')

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
