# ai_alert.py
import os
import time
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from zoneinfo import ZoneInfo  # Python 3.9+
from flask import Flask, jsonify, Response
from threading import Thread

load_dotenv()

# 🔐 ENV
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

# โหลดโมเดล
model = joblib.load("xau_model.pkl")

app = Flask(__name__)
last_signal = None
last_sent_hour = None  # ส่งแค่ต้นชั่วโมง

FEATURES = ["rsi", "ema", "price_change"]

# ====================== Utilities ======================
def send_telegram(message: str) -> int:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    res = requests.post(url, data=data, timeout=20)
    return res.status_code

def get_latest_xau() -> pd.DataFrame:
    # Historical H1
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD&interval=1h&outputsize=50&apikey={API_KEY}"
    )
    res = requests.get(url, timeout=30).json()
    df = pd.DataFrame(res["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df.set_index("datetime", inplace=True)
    df["close"] = df["close"].astype(float)

    # ====== ดึงราคาปัจจุบันแบบ Real-time ======
    price_url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
    real_res = requests.get(price_url, timeout=15).json()
    real_price = float(real_res["price"])

    # ====== ใช้ real-time price แทนแท่งล่าสุด ======
    last_idx = df.index[-1]
    df.at[last_idx, "close"] = real_price

    # ====== คำนวณ Indicators ======
    import ta
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df["price_change"] = df["close"].pct_change()

    df.dropna(subset=FEATURES, inplace=True)
    return df.tail(1)

def format_th_time(ts_utc: pd.Timestamp) -> str:
    thai_time = ts_utc.tz_localize("UTC").astimezone(ZoneInfo("Asia/Bangkok"))
    return thai_time.strftime("%Y-%m-%d %H:%M")

# ================== Model-based Explanation ==================
def explain_prediction(model, x_vec: np.ndarray, price: float, ema_val: float, rsi_val: float, pred_label: int):
    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([x_vec])[0]
            confidence = float(np.max(proba))
            pred_by_proba = int(np.argmax(proba))
            if hasattr(model, "classes_"):
                cls_idx = np.where(model.classes_ == pred_label)[0]
                if cls_idx.size > 0:
                    confidence = float(proba[cls_idx[0]])
    except Exception:
        pass

    lines = []

    if hasattr(model, "coef_"):
        try:
            coefs = np.ravel(model.coef_)
            contributions = coefs * x_vec
            target_sign = 1 if pred_label == 1 else -1
            support_scores = contributions * target_sign
            top_idx = np.argsort(np.abs(support_scores))[::-1][:2]
            dir_word = "BUY" if pred_label == 1 else "SELL"

            lines.append("เหตุผล (จากน้ำหนักโมเดลเชิงเส้น):")
            for idx in top_idx:
                fname = FEATURES[idx]
                w = coefs[idx]
                val = x_vec[idx]
                effect = "หนุน" if support_scores[idx] >= 0 else "ต้าน"
                lines.append(f"• {fname}: ค่า {val:.5f} × weight {w:.4f} ⇒ {effect} {dir_word}")
        except Exception:
            pass

    elif hasattr(model, "feature_importances_"):
        try:
            imps = np.array(model.feature_importances_)
            order = np.argsort(imps)[::-1]
            top_idx = order[:2]
            imp_txt = ", ".join([f"{FEATURES[i]}={imps[i]:.2f}" for i in top_idx])

            dir_word = "BUY" if pred_label == 1 else "SELL"
            ctx = []
            ctx.append(f"RSI={rsi_val:.2f} ({'ต่ำ/oversold' if rsi_val<30 else 'สูง/overbought' if rsi_val>70 else 'โซนกลาง'})")
            ctx.append(f"ราคา {'เหนือ' if price>ema_val else 'ใต้'} EMA10 ({ema_val:.2f})")
            ctx.append(f"ราคาเปลี่ยนแปลงชั่วโมงล่าสุด={x_vec[2]:.5f}")

            lines.append("เหตุผล (จากความสำคัญฟีเจอร์ของโมเดล):")
            lines.append(f"• ฟีเจอร์เด่น: {imp_txt}")
            lines.append(f"• บริบทตอนนี้: {', '.join(ctx)} → หนุน {dir_word}")
        except Exception:
            pass

    if not lines:
        dir_word = "BUY" if pred_label == 1 else "SELL"
        rule_hint = []
        if rsi_val < 30:
            rule_hint.append("RSI < 30 (oversold)")
        elif rsi_val > 70:
            rule_hint.append("RSI > 70 (overbought)")
        rule_hint.append("ราคาเหนือ EMA10" if price > ema_val else "ราคาใต้ EMA10")
        lines.append("เหตุผล (fallback): " + ", ".join(rule_hint) + f" → {dir_word}")

    if confidence is not None:
        lines.append(f"ความมั่นใจของโมเดล ≈ {confidence*100:.1f}%")

    return "\n".join(lines), confidence

def calc_targets(pred_label: int, price: float):
    if pred_label == 1:  # BUY
        tp1 = price * 1.002
        tp2 = price * 1.004
        tp3 = price * 1.006
        sl  = price * 0.998
        signal = "📈 BUY"
    else:  # SELL
        tp1 = price * 0.998
        tp2 = price * 0.996
        tp3 = price * 0.994
        sl  = price * 1.002
        signal = "📉 SELL"
    return signal, (tp1, tp2, tp3, sl)

# ================== Core Prediction ==================
def run_ai_once():
    global last_signal
    try:
        latest = get_latest_xau()
        X_live = latest[FEATURES]

        if X_live.empty or X_live.isnull().values.any():
            price = latest["close"].iloc[0]
            ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
            msg = (
                f"ตอนนี้ วันที่ {ts_txt}\n"
                f"XAUUSD TF H1 ราคาปัจจุบัน {price:,.2f}$\n"
                f"BOT ยังไม่สามารถทำนายได้"
            )
            send_telegram(msg)
            return msg

        x = X_live.iloc[0].values.astype(float)
        price = latest["close"].iloc[0]
        ema_val = float(latest["ema"].iloc[0])
        rsi_val = float(latest["rsi"].iloc[0])

        pred = int(model.predict([x])[0])
        reason_text, _ = explain_prediction(model, x, price, ema_val, rsi_val, pred)

        signal, (tp1, tp2, tp3, sl) = calc_targets(pred, price)
        ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")

        msg = (
            f"ตอนนี้ วันที่ {ts_txt}\n"
            f"XAUUSD TF H1 ราคาปัจจุบัน {price:,.2f}$\n"
            f"BOT ทำนาย: {signal}\n"
            f"{reason_text}\n"
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

# ================== Routes ==================
@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    return Response("OK", status=200, headers={
        "Content-Type": "text/plain",
        "Cache-Control": "no-cache"
    })

@app.route('/test-telegram')
def test_telegram():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"✅ ทดสอบส่งข้อความจาก AI Bot @ {now}"
    status = send_telegram(message)
    return jsonify({"status": status, "message": message})

@app.route('/run-ai')
def run_ai():
    global last_sent_hour
    now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
    current_hour = now_th.hour

    if current_hour != last_sent_hour:
        last_sent_hour = current_hour
        def task():
            result = run_ai_once()
            print(result)
        Thread(target=task, daemon=True).start()
        return jsonify({"status": "✅ ส่งข้อความรอบต้นชั่วโมง", "time": now_th.strftime("%Y-%m-%d %H:%M")})
    else:
        return jsonify({"status": "⏳ รอรอบต้นชั่วโมงถัดไป", "time": now_th.strftime("%Y-%m-%d %H:%M")})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
