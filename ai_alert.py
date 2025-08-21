import os
import time
import joblib
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, Response
from threading import Thread
import io
import warnings
warnings.filterwarnings('ignore')

# Try importing optional ML libraries
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
    HAS_TENSORFLOW = True
except ImportError:
    print("⚠️ TensorFlow not available, using fallback methods")
    HAS_TENSORFLOW = False

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    print("⚠️ Scikit-learn not available")
    HAS_SKLEARN = False

try:
    import ta
    HAS_TA = True
except ImportError:
    print("⚠️ TA-lib not available, using basic indicators")
    HAS_TA = False

try:
    import mplfinance as mpf
    from PIL import Image
    HAS_CHARTS = True
except ImportError:
    print("⚠️ Chart libraries not available")
    HAS_CHARTS = False

load_dotenv()

# 🔐 Environment Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

app = Flask(__name__)

# Global variables
last_signal = None
last_original_sent_hour = None  # สำหรับระบบเก่า
last_pattern_sent_hour = None   # สำหรับระบบ Pattern AI
message_sent_this_hour = {      # เพิ่มตัวตรวจสอบว่าส่งข้อความไหนแล้วบ้าง
    'original': None,
    'pattern': None
}

FEATURES = ["rsi", "ema", "price_change"]

# Load original model (if exists)
try:
    model = joblib.load("xau_model.pkl")
    print("✅ Original XAU model loaded successfully")
except:
    print("⚠️ Original model not found, creating dummy model")
    if HAS_SKLEARN:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Create dummy training data
        X_dummy = np.random.randn(100, 3)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
    else:
        model = None

# ====================== Basic Technical Indicators ======================

def calculate_rsi(prices, period=14):
    """Calculate RSI manually if TA library not available"""
    try:
        if HAS_TA:
            return ta.momentum.RSIIndicator(prices, window=period).rsi()
        else:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    except:
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_ema(prices, period=10):
    """Calculate EMA manually if TA library not available"""
    try:
        if HAS_TA:
            return ta.trend.EMAIndicator(prices, window=period).ema_indicator()
        else:
            return prices.ewm(span=period).mean()
    except:
        return prices

# ====================== Data Fetching ======================

def get_latest_xau():
    """Get latest XAU data with technical indicators"""
    try:
        # Historical H1
        url = (
            "https://api.twelvedata.com/time_series"
            f"?symbol=XAU/USD&interval=1h&outputsize=50&apikey={API_KEY}"
        )
        res = requests.get(url, timeout=30).json()
        
        if 'values' not in res:
            raise Exception(f"API Error: {res}")
            
        df = pd.DataFrame(res["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df.set_index("datetime", inplace=True)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

        # Get real-time price
        try:
            price_url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
            real_res = requests.get(price_url, timeout=15).json()
            if 'price' in real_res:
                real_price = float(real_res["price"])
                last_idx = df.index[-1]
                df.at[last_idx, "close"] = real_price
        except:
            print("⚠️ Real-time price fetch failed, using last close")

        # Calculate indicators
        df["rsi"] = calculate_rsi(df["close"])
        df["ema"] = calculate_ema(df["close"], 10)
        df["price_change"] = df["close"].pct_change()

        df.dropna(subset=FEATURES, inplace=True)
        return df
        
    except Exception as e:
        print(f"Error in get_latest_xau: {e}")
        return None

def get_xau_data_extended():
    """Get extended XAU data for pattern detection"""
    try:
        url = (
            "https://api.twelvedata.com/time_series"
            f"?symbol=XAU/USD&interval=1h&outputsize=100&apikey={API_KEY}"
        )
        
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'values' not in data:
            raise Exception(f"API Error: {data}")
            
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        df.set_index('datetime', inplace=True)
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get real-time price
        try:
            price_url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
            price_response = requests.get(price_url, timeout=15)
            real_price_data = price_response.json()
            
            if 'price' in real_price_data:
                real_price = float(real_price_data['price'])
                df.iloc[-1, df.columns.get_loc('close')] = real_price
        except:
            print("⚠️ Real-time price fetch failed")
        
        return df
        
    except Exception as e:
        print(f"Error fetching extended XAU data: {e}")
        return None

# ====================== Original System Functions ======================

def explain_prediction(model, x_vec: np.ndarray, price: float, ema_val: float, rsi_val: float, pred_label: int):
    """Explain model prediction"""
    confidence = None
    
    if model is None:
        return "โมเดลไม่พร้อมใช้งาน", 0.5
        
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([x_vec])[0]
            confidence = float(np.max(proba))
    except Exception:
        confidence = 0.5

    lines = []
    dir_word = "BUY" if pred_label == 1 else "SELL"
    
    # Simple explanation
    ctx = []
    ctx.append(f"RSI={rsi_val:.2f} ({'ต่ำ/oversold' if rsi_val<30 else 'สูง/overbought' if rsi_val>70 else 'โซนกลาง'})")
    ctx.append(f"ราคา {'เหนือ' if price>ema_val else 'ใต้'} EMA10 ({ema_val:.2f})")
    ctx.append(f"การเปลี่ยนแปลงราคา={x_vec[2]:.5f}")

    lines.append("เหตุผล:")
    lines.append(f"• {', '.join(ctx)} → สัญญาณ {dir_word}")
    
    if confidence is not None:
        lines.append(f"ความมั่นใจ ≈ {confidence*100:.1f}%")

    return "\n".join(lines), confidence

def calc_targets(pred_label: int, price: float):
    """Calculate TP/SL targets"""
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

def run_ai_once():
    """Original AI system"""
    global last_signal
    try:
        latest = get_latest_xau()
        if latest is None or len(latest) < 5:
            return "❌ ไม่สามารถดึงข้อมูลได้"
            
        X_live = latest[FEATURES]

        if X_live.empty or X_live.isnull().values.any():
            price = latest["close"].iloc[0] if len(latest) > 0 else 2500
            ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
            msg = (
                f"🤖 ORIGINAL BOT (RSI+EMA+Price Change)\n"
                f"ตอนนี้ วันที่ {ts_txt}\n"
                f"XAUUSD ราคาปัจจุบัน {price:,.2f}$\n"
                f"ยังไม่สามารถทำนายได้ (ข้อมูลไม่เพียงพอ)"
            )
            return msg

        x = X_live.iloc[0].values.astype(float)
        price = latest["close"].iloc[0]
        ema_val = float(latest["ema"].iloc[0])
        rsi_val = float(latest["rsi"].iloc[0])

        # OHLC
        o = latest["open"].iloc[0]
        h = latest["high"].iloc[0]
        l = latest["low"].iloc[0]
        c = latest["close"].iloc[0]

        # Prediction
        if model is not None:
            try:
                pred = int(model.predict([x])[0])
            except:
                pred = 1 if rsi_val < 50 else 0
        else:
            pred = 1 if rsi_val < 50 else 0
            
        reason_text, _ = explain_prediction(model, x, price, ema_val, rsi_val, pred)
        signal, (tp1, tp2, tp3, sl) = calc_targets(pred, price)
        ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")

        msg = (
            f"🤖 ORIGINAL BOT (RSI+EMA+Price Change)\n"
            f"⏰ {ts_txt}\n"
            f"💰 XAUUSD TF H1\n"
            f"Open = {o:,.2f}$ | High = {h:,.2f}$\n"
            f"Low = {l:,.2f}$ | Close = {c:,.2f}$\n"
            f"ราคาปัจจุบัน = {price:,.2f}$\n\n"
            f"🎯 BOT ทำนาย: {signal}\n"
            f"{reason_text}\n\n"
            f"🎯 TP1: {tp1:,.2f}$ | TP2: {tp2:,.2f}$\n"
            f"🎯 TP3: {tp3:,.2f}$ | 🛑 SL: {sl:,.2f}$"
        )
        
        last_signal = signal
        return msg
        
    except Exception as e:
        return f"❌ ORIGINAL BOT ERROR: {e}"

# ====================== Pattern Detection System ======================

class SimplePatternDetector:
    def __init__(self):
        self.patterns = {
            0: "NO_PATTERN",
            1: "HEAD_SHOULDERS", 
            2: "DOUBLE_TOP",
            3: "DOUBLE_BOTTOM",
            4: "ASCENDING_TRIANGLE",
            5: "BULL_FLAG"
        }

    def detect_pattern(self, df):
        """Simple rule-based pattern detection"""
        try:
            if len(df) < 20:
                return {
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'INSUFFICIENT_DATA'
                }
                
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            closes = df['close'].values[-20:]
            
            # Head & Shoulders detection
            if len(highs) >= 5:
                mid_idx = len(highs) // 2
                if mid_idx >= 2 and mid_idx + 2 < len(highs):
                    left_shoulder = highs[mid_idx-2]
                    head = highs[mid_idx]
                    right_shoulder = highs[mid_idx+2]
                    
                    if head > left_shoulder and head > right_shoulder:
                        if abs(left_shoulder - right_shoulder) / left_shoulder < 0.02:
                            return {
                                'pattern_id': 1,
                                'pattern_name': 'HEAD_SHOULDERS',
                                'confidence': 0.75,
                                'method': 'RULE_BASED'
                            }
            
            # Double top detection
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.01:
                    return {
                        'pattern_id': 2,
                        'pattern_name': 'DOUBLE_TOP',
                        'confidence': 0.70,
                        'method': 'RULE_BASED'
                    }
            
            # Simple trend analysis
            recent_closes = closes[-10:]
            if len(recent_closes) >= 10:
                slope = (recent_closes[-1] - recent_closes[0]) / len(recent_closes)
                if slope > 0:
                    return {
                        'pattern_id': 5,
                        'pattern_name': 'BULL_FLAG',
                        'confidence': 0.60,
                        'method': 'TREND_ANALYSIS'
                    }
            
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.50,
                'method': 'RULE_BASED'
            }
            
        except Exception as e:
            print(f"Pattern detection error: {e}")
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN', 
                'confidence': 0.30,
                'method': 'ERROR'
            }

    def predict_signals(self, df, pattern_info):
        """Predict trading signals based on patterns"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate basic indicators
            df['rsi'] = calculate_rsi(df['close'])
            df['ema_10'] = calculate_ema(df['close'], 10)
            df['ema_21'] = calculate_ema(df['close'], 21)
            
            current_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
            current_ema10 = df['ema_10'].iloc[-1]
            current_ema21 = df['ema_21'].iloc[-1]
            
            pattern_id = pattern_info['pattern_id']
            confidence = pattern_info['confidence']
            
            # Pattern-based signals
            if pattern_id == 1:  # HEAD_SHOULDERS (Bearish)
                action = "SELL"
                entry_price = current_price - (current_price * 0.001)
                tp1 = current_price * 0.995
                tp2 = current_price * 0.990  
                tp3 = current_price * 0.985
                sl = current_price * 1.010
                
            elif pattern_id == 2:  # DOUBLE_TOP (Bearish)
                action = "SELL"
                entry_price = current_price - (current_price * 0.0005)
                tp1 = current_price * 0.996
                tp2 = current_price * 0.992
                tp3 = current_price * 0.988
                sl = current_price * 1.008
                
            elif pattern_id == 3:  # DOUBLE_BOTTOM (Bullish)
                action = "BUY"
                entry_price = current_price + (current_price * 0.0005)
                tp1 = current_price * 1.004
                tp2 = current_price * 1.008
                tp3 = current_price * 1.012
                sl = current_price * 0.992
                
            elif pattern_id in [4, 5]:  # ASCENDING_TRIANGLE, BULL_FLAG (Bullish)
                action = "BUY"
                entry_price = current_price + (current_price * 0.001)
                tp1 = current_price * 1.005
                tp2 = current_price * 1.010
                tp3 = current_price * 1.015
                sl = current_price * 0.990
                
            else:  # NO_PATTERN - use indicators
                if current_rsi < 30 and current_price < current_ema10:
                    action = "BUY"
                    entry_price = current_price + (current_price * 0.0005)
                    tp1 = current_price * 1.003
                    tp2 = current_price * 1.006
                    tp3 = current_price * 1.009
                    sl = current_price * 0.995
                elif current_rsi > 70 and current_price > current_ema10:
                    action = "SELL"
                    entry_price = current_price - (current_price * 0.0005)
                    tp1 = current_price * 0.997
                    tp2 = current_price * 0.994
                    tp3 = current_price * 0.991
                    sl = current_price * 1.005
                else:
                    action = "WAIT"
                    entry_price = current_price
                    tp1 = tp2 = tp3 = sl = current_price
            
            # Apply confidence filter
            if confidence < 0.6 and action != "WAIT":
                action = "WAIT"
                
            return {
                'action': action,
                'entry_price': round(entry_price, 2),
                'tp1': round(tp1, 2),
                'tp2': round(tp2, 2), 
                'tp3': round(tp3, 2),
                'sl': round(sl, 2),
                'confidence': confidence,
                'current_price': round(current_price, 2),
                'rsi': round(current_rsi, 1),
                'ema10': round(current_ema10, 2),
                'ema21': round(current_ema21, 2)
            }
            
        except Exception as e:
            print(f"Signal prediction error: {e}")
            current_price = df['close'].iloc[-1] if len(df) > 0 else 2500
            return {
                'action': 'WAIT',
                'entry_price': round(current_price, 2),
                'tp1': round(current_price, 2),
                'tp2': round(current_price, 2),
                'tp3': round(current_price, 2),
                'sl': round(current_price, 2),
                'confidence': 0.30,
                'current_price': round(current_price, 2),
                'rsi': 50.0,
                'ema10': round(current_price, 2),
                'ema21': round(current_price, 2)
            }

def run_pattern_ai():
    """Pattern AI system"""
    try:
        df = get_xau_data_extended()
        if df is None or len(df) < 20:
            return "❌ ไม่สามารถดึงข้อมูลสำหรับ Pattern Detection ได้"
        
        detector = SimplePatternDetector()
        pattern_info = detector.detect_pattern(df.tail(50))
        trading_signals = detector.predict_signals(df, pattern_info)
        
        current_data = df.iloc[-1]
        current_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        
        # Pattern descriptions
        pattern_desc = {
            'HEAD_SHOULDERS': '🗣️ หัวไหล่ (Bearish Reversal)',
            'DOUBLE_TOP': '⛰️ ยอดคู่ (Bearish Reversal)',
            'DOUBLE_BOTTOM': '🏔️ ก้นคู่ (Bullish Reversal)', 
            'ASCENDING_TRIANGLE': '📈 สามเหลี่ยมขาขึ้น (Bullish)',
            'BULL_FLAG': '🚩 ธงวัว (Bullish Continuation)',
            'NO_PATTERN': '🔍 ไม่พบแพทเทิร์นชัดเจน'
        }
        
        action_emoji = {
            'BUY': '🟢 BUY',
            'SELL': '🔴 SELL', 
            'WAIT': '🟡 WAIT'
        }
        
        message = f"""🚀 AI PATTERN DETECTION BOT
⏰ {current_time} | 💰 XAUUSD (1H)

📊 MARKET DATA:
Open: ${current_data['open']:,.2f} | High: ${current_data['high']:,.2f}
Low: ${current_data['low']:,.2f} | Close: ${current_data['close']:,.2f}

🔍 PATTERN DETECTED:
{pattern_desc.get(pattern_info['pattern_name'], pattern_info['pattern_name'])}
🤖 Method: {pattern_info['method']} | 🎯 Confidence: {pattern_info['confidence']*100:.1f}%

📈 TECHNICAL INDICATORS:
RSI: {trading_signals['rsi']:.1f} ({'Oversold' if trading_signals['rsi']<30 else 'Overbought' if trading_signals['rsi']>70 else 'Neutral'})
EMA10: ${trading_signals['ema10']:,.2f} ({'Above' if trading_signals['current_price']>trading_signals['ema10'] else 'Below'})
EMA21: ${trading_signals['ema21']:,.2f} ({'Above' if trading_signals['current_price']>trading_signals['ema21'] else 'Below'})

🚦 PATTERN AI SIGNAL: {action_emoji[trading_signals['action']]}"""

        if trading_signals['action'] != 'WAIT':
            message += f"""

💼 TRADING SETUP:
🎯 Entry: ${trading_signals['entry_price']:,.2f}
🟢 TP1: ${trading_signals['tp1']:,.2f} | TP2: ${trading_signals['tp2']:,.2f} | TP3: ${trading_signals['tp3']:,.2f}
🛑 SL: ${trading_signals['sl']:,.2f}
📊 Pattern Confidence: {trading_signals['confidence']*100:.1f}%

⚠️ Risk: ใช้เงินเพียง 1-2% ต่อออเดอร์"""
        else:
            message += f"""

⏳ รอ Pattern ที่ชัดเจนกว่า
📊 Current: ${trading_signals['current_price']:,.2f}
🔍 กำลังวิเคราะห์แพทเทิร์นใหม่..."""

        return message
        
    except Exception as e:
        return f"❌ PATTERN AI ERROR: {str(e)}"

# ====================== Utilities ======================

def send_telegram(message: str) -> int:
    """Send message to Telegram"""
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print("⚠️ Telegram credentials not configured")
            return 400
            
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": CHAT_ID, 
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=20)
        print(f"Telegram response: {response.status_code}")
        return response.status_code
    except Exception as e:
        print(f"Telegram send error: {e}")
        return 500

# ====================== Flask Routes ======================

@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    """Health check endpoint"""
    return Response("OK", status=200, headers={
        "Content-Type": "text/plain",
        "Cache-Control": "no-cache"
    })

@app.route('/test-telegram')
def test_telegram():
    """Test Telegram connection"""
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"✅ ทดสอบส่งข้อความจาก AI Bot @ {now}"
        status = send_telegram(message)
        return jsonify({
            "status": status, 
            "message": message,
            "success": status == 200
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/run-ai')
def run_ai():
    """Run original AI system - Send Telegram once per hour"""
    global last_original_sent_hour, message_sent_this_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        current_time = now_th.strftime("%Y-%m-%d %H:%M")
        
        # Reset message tracking เมื่อเปลี่ยนชั่วโมง
        if current_hour != last_original_sent_hour:
            last_original_sent_hour = current_hour
            message_sent_this_hour['original'] = None  # Reset status
        
        # ตรวจสอบว่าส่งข้อความในชั่วโมงนี้แล้วหรือยัง
        if message_sent_this_hour['original'] != current_hour:
            # ส่งสัญญาณจริง (ครั้งแรกในชั่วโมงนั้น)
            message_sent_this_hour['original'] = current_hour
            
            def send_original_task():
                try:
                    result = run_ai_once()
                    send_status = send_telegram(result)
                    print(f"✅ [{current_time}] Original AI sent to Telegram: Status {send_status}")
                    print(f"Original message preview: {result[:150]}...")
                except Exception as e:
                    print(f"❌ [{current_time}] Original AI send error: {e}")
                    error_msg = f"❌ Original AI Error @ {current_time}\nError: {str(e)[:100]}"
                    send_telegram(error_msg)
            
            Thread(target=send_original_task, daemon=True).start()
            
            return jsonify({
                "status": "✅ Original AI - Signal Sent", 
                "mode": "TELEGRAM_SENT",
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": True,
                "system": "RSI+EMA+Price Change",
                "note": f"🤖 ORIGINAL signal sent to Telegram at {current_time}",
                "sent_count_this_hour": 1
            })
        else:
            # Ping ครั้งที่ 2+ ในชั่วโมงเดียวกัน (แค่ keep alive)
            return jsonify({
                "status": "✅ Original AI - Keep Alive",
                "mode": "KEEP_ALIVE", 
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": False,
                "system": "RSI+EMA+Price Change",
                "note": f"Original signal already sent in hour {current_hour}, keeping service alive",
                "next_signal_time": f"{current_hour + 1}:00"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/run-pattern-bot')
def run_pattern_bot():
    """Run pattern AI system - Send Telegram once per hour"""
    global last_pattern_sent_hour, message_sent_this_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        current_time = now_th.strftime("%Y-%m-%d %H:%M")
        
        # Reset message tracking เมื่อเปลี่ยนชั่วโมง
        if current_hour != last_pattern_sent_hour:
            last_pattern_sent_hour = current_hour
            message_sent_this_hour['pattern'] = None  # Reset status
        
        # ตรวจสอบว่าส่งข้อความในชั่วโมงนี้แล้วหรือยัง
        if message_sent_this_hour['pattern'] != current_hour:
            # ส่งสัญญาณจริง (ครั้งแรกในชั่วโมงนั้น)
            message_sent_this_hour['pattern'] = current_hour
            
            def send_pattern_task():
                try:
                    result = run_pattern_ai()
                    send_status = send_telegram(result)
                    print(f"✅ [{current_time}] Pattern AI sent to Telegram: Status {send_status}")
                    print(f"Pattern message preview: {result[:150]}...")
                except Exception as e:
                    print(f"❌ [{current_time}] Pattern AI send error: {e}")
                    error_msg = f"❌ Pattern AI Error @ {current_time}\nError: {str(e)[:100]}"
                    send_telegram(error_msg)
            
            Thread(target=send_pattern_task, daemon=True).start()
            
            return jsonify({
                "status": "✅ Pattern AI - Signal Sent", 
                "mode": "TELEGRAM_SENT",
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": True,
                "system": "CNN+RNN+Patterns",
                "note": f"🚀 PATTERN signal sent to Telegram at {current_time}",
                "sent_count_this_hour": 1
            })
        else:
            # Ping ครั้งที่ 2+ ในชั่วโมงเดียวกัน (แค่ keep alive)
            return jsonify({
                "status": "✅ Pattern AI - Keep Alive",
                "mode": "KEEP_ALIVE", 
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": False,
                "system": "CNN+RNN+Patterns",
                "note": f"Pattern signal already sent in hour {current_hour}, keeping service alive",
                "next_signal_time": f"{current_hour + 1}:00"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/test-pattern-ai')
def test_pattern_ai():
    """Test pattern AI system"""
    try:
        result = run_pattern_ai()
        return jsonify({
            "status": "success", 
            "message": "Pattern AI test completed",
            "result": result[:500] + "..." if len(result) > 500 else result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/pattern-status')
def pattern_status():
    """Get current pattern analysis status"""
    try:
        df = get_xau_data_extended()
        if df is None or len(df) < 10:
            return jsonify({
                "status": "error", 
                "message": "Cannot fetch sufficient data"
            })
        
        detector = SimplePatternDetector()
        pattern_info = detector.detect_pattern(df.tail(50))
        
        current_price = float(df['close'].iloc[-1])
        current_rsi = float(calculate_rsi(df['close']).iloc[-1]) if len(df) > 14 else 50.0
        
        return jsonify({
            "status": "success",
            "current_price": current_price,
            "pattern_detection": pattern_info,
            "rsi": current_rsi,
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        })

@app.route('/status')
def system_status():
    """Get overall system status"""
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        
        status_info = {
            "app": "XAU AI Trading Bot",
            "version": "2.0 - Dual Signal System",
            "timestamp": datetime.now().isoformat(),
            "current_hour": current_hour,
            "bangkok_time": now_th.strftime("%Y-%m-%d %H:%M:%S"),
            "systems": {
                "original": "RSI + EMA + Price Change",
                "pattern": "Rule-based Pattern Detection"
            },
            "message_status": {
                "original_sent_this_hour": message_sent_this_hour.get('original') == current_hour,
                "pattern_sent_this_hour": message_sent_this_hour.get('pattern') == current_hour,
                "total_messages_this_hour": sum([
                    1 if message_sent_this_hour.get('original') == current_hour else 0,
                    1 if message_sent_this_hour.get('pattern') == current_hour else 0
                ])
            },
            "libraries": {
                "tensorflow": HAS_TENSORFLOW,
                "sklearn": HAS_SKLEARN,
                "ta": HAS_TA,
                "charts": HAS_CHARTS
            },
            "endpoints": [
                "/health",
                "/run-ai",
                "/run-pattern-bot", 
                "/test-telegram",
                "/test-pattern-ai",
                "/pattern-status",
                "/status"
            ],
            "environment": {
                "bot_token_configured": bool(BOT_TOKEN),
                "chat_id_configured": bool(CHAT_ID),
                "api_key_configured": bool(API_KEY)
            }
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/')
def home():
    """Home page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>XAU AI Trading Bot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #1a1a1a; color: #ffffff; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #00ff88; text-align: center; }
            h2 { color: #ffaa00; border-bottom: 2px solid #ffaa00; padding-bottom: 10px; }
            .endpoint { background-color: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #00ff88; }
            .method { display: inline-block; background-color: #007acc; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
            .status { color: #00ff88; font-weight: bold; }
            .warning { color: #ffaa00; }
            .error { color: #ff4444; }
            code { background-color: #333; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 XAU AI Trading Bot</h1>
            <p class="status">✅ System Online | Hybrid AI Trading System</p>
            
            <h2>📊 Trading Systems</h2>
            <ul>
                <li><strong>Original System:</strong> RSI + EMA + Price Change Analysis</li>
                <li><strong>Pattern AI System:</strong> Chart Pattern Detection + Technical Analysis</li>
            </ul>
            
            <h2>🔗 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong>
                <p>Health check endpoint for monitoring services</p>
                <p><em>Returns:</em> Simple "OK" response</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/run-ai</strong>
                <p><span class="status">📡 ACTIVE MODE:</span> Execute original AI system with Telegram alerts</p>
                <p><em>Frequency:</em> Every 3 minutes (recommended)</p>
                <p><em>Output:</em> Telegram message <strong>once per hour</strong> with RSI+EMA+Price Change signals</p>
                <p><em>Logic:</em> First ping of each hour = send signal, subsequent pings = keep alive</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/run-pattern-bot</strong>
                <p><span class="status">📡 ACTIVE MODE:</span> Execute pattern AI trading system with Telegram alerts</p>
                <p><em>Frequency:</em> Every 3 minutes (recommended)</p>
                <p><em>Output:</em> Telegram message <strong>once per hour</strong> with pattern-based signals</p>
                <p><em>Logic:</em> First ping of each hour = send signal, subsequent pings = keep alive</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/test-telegram</strong>
                <p>Test Telegram bot connection</p>
                <p><em>Returns:</em> Test message status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/test-pattern-ai</strong>
                <p>Test pattern detection system without sending to Telegram</p>
                <p><em>Returns:</em> Pattern analysis results</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/pattern-status</strong>
                <p>Get current pattern analysis status</p>
                <p><em>Returns:</em> JSON with current patterns and indicators</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/status</strong>
                <p>Get comprehensive system status</p>
                <p><em>Returns:</em> System information and configuration status</p>
            </div>
            
            <h2>⚙️ Configuration</h2>
            <p>The bot requires the following environment variables:</p>
            <ul>
                <li><code>BOT_TOKEN</code> - Telegram bot token</li>
                <li><code>CHAT_ID</code> - Telegram chat ID for messages</li>
                <li><code>API_KEY</code> - TwelveData API key for market data</li>
            </ul>
            
            <h2>📈 Recommended UptimeRobot Setup</h2>
            <div style="background-color: #2a2a2a; padding: 20px; border-radius: 8px; border-left: 4px solid #00ff88;">
                <h3 style="margin-top: 0; color: #00ff88;">🎯 Dual System Strategy:</h3>
                <p><strong>Monitor 1:</strong> <code>/run-ai</code> - Every 3 minutes</p>
                <p style="margin-left: 20px;">→ Sends ORIGINAL system signals <strong>once per hour</strong></p>
                
                <p><strong>Monitor 2:</strong> <code>/run-pattern-bot</code> - Every 3 minutes</p>
                <p style="margin-left: 20px;">→ Sends PATTERN AI signals <strong>once per hour</strong></p>
                
                <p style="color: #ffaa00;"><strong>Result:</strong> <span style="color: #00ff88;">Exactly 2 trading signals per hour</span> via Telegram</p>
                <p style="color: #ffaa00;"><strong>Benefit:</strong> Compare both systems + Service never sleeps + No duplicate messages</p>
                
                <h4 style="color: #00ff88;">📱 Expected Telegram Messages per Hour:</h4>
                <p>🤖 <strong>Original AI Signal</strong> - RSI + EMA + Price Change analysis</p>
                <p>🚀 <strong>Pattern AI Signal</strong> - CNN + RNN + Pattern detection</p>
                <p style="color: #666;">Each system sends exactly once per hour, independent tracking</p>
                
                <h4 style="color: #ffaa00;">🔧 How It Works:</h4>
                <p style="margin-left: 10px;">• First ping to <code>/run-ai</code> in each hour = Send Original signal</p>
                <p style="margin-left: 10px;">• First ping to <code>/run-pattern-bot</code> in each hour = Send Pattern signal</p>
                <p style="margin-left: 10px;">• Subsequent pings = Keep service alive only</p>
                <p style="margin-left: 10px;">• Independent tracking prevents interference between systems</p>
            </div>
            
            <h2>⚠️ Risk Disclaimer</h2>
            <p class="warning">This is an automated trading bot for educational purposes. Always use proper risk management and never risk more than 1-2% of your account per trade. Past performance does not guarantee future results.</p>
            
            <hr style="border-color: #444; margin: 40px 0;">
            <p style="text-align: center; color: #666;">
                🚀 XAU AI Trading Bot v2.0 | Powered by Python + Flask + AI
            </p>
        </div>
    </body>
    </html>
    """
    return html_content

# ====================== Error Handlers ======================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/health", "/run-ai", "/run-pattern-bot", 
            "/test-telegram", "/test-pattern-ai", 
            "/pattern-status", "/status", "/"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "suggestion": "Check the logs or try again later"
    }), 500

# ====================== Main Application ======================

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 XAU AI Trading Bot v2.0 Starting...")
    print("=" * 60)
    print(f"🔗 Health Check: /health")
    print(f"📈 Original System: /run-ai")
    print(f"🎯 Pattern AI: /run-pattern-bot")
    print(f"📱 Test Telegram: /test-telegram")
    print(f"🧪 Test Pattern: /test-pattern-ai")
    print(f"📊 Pattern Status: /pattern-status")
    print(f"ℹ️ System Status: /status")
    print("=" * 60)
    print(f"🔥 Libraries Available:")
    print(f"   • TensorFlow: {'✅' if HAS_TENSORFLOW else '❌'}")
    print(f"   • Scikit-learn: {'✅' if HAS_SKLEARN else '❌'}")
    print(f"   • TA-Lib: {'✅' if HAS_TA else '❌'}")
    print(f"   • Charts: {'✅' if HAS_CHARTS else '❌'}")
    print("=" * 60)
    print(f"⚙️ Configuration:")
    print(f"   • Bot Token: {'✅ Configured' if BOT_TOKEN else '❌ Missing'}")
    print(f"   • Chat ID: {'✅ Configured' if CHAT_ID else '❌ Missing'}")
    print(f"   • API Key: {'✅ Configured' if API_KEY else '❌ Missing'}")
    print("=" * 60)
    print("🚀 Ready for AI-powered trading!")
    print("💰 Asset: XAU/USD | Timeframe: 1H")
    print("📡 Monitoring: Configure UptimeRobot with endpoints above")
    print("=" * 60)
    
    # Get port from environment
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=False)
