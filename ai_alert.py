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
last_sent_hour = None

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
    """Run original AI system (RSI+EMA+Price Change)"""
    global last_sent_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour

        if current_hour != last_sent_hour:
            last_sent_hour = current_hour
            
            def task():
                try:
                    result = run_ai_once()
                    send_telegram(result)
                    print(f"Original AI Result: {result[:200]}...")
                except Exception as e:
                    print(f"Original AI Task Error: {e}")
            
            Thread(target=task, daemon=True).start()
            
            return jsonify({
                "status": "✅ Original AI Bot executed",
                "time": now_th.strftime("%Y-%m-%d %H:%M"),
                "system": "RSI+EMA+Price Change"
            })
        else:
            return jsonify({
                "status": "⏳ รอรอบต้นชั่วโมงถัดไป",
                "time": now_th.strftime("%Y-%m-%d %H:%M"),
                "next_run": f"{current_hour + 1}:00"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/run-pattern-bot')
def run_pattern_bot():
    """Run pattern AI system (CNN+RNN+Patterns)"""
    global last_sent_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        
        if current_hour != last_sent_hour:
            last_sent_hour = current_hour
            
            def task():
                try:
                    result = run_pattern_ai()
                    send_telegram(result)
                    print(f"Pattern AI Result: {result[:200]}...")
                except Exception as e:
                    print(f"Pattern AI Task Error: {e}")
            
            Thread(target=task, daemon=True).start()
            
            return jsonify({
                "status": "✅ Pattern AI Bot executed", 
                "time": now_th.strftime("%Y-%m-%d %H:%M"),
                "system": "CNN+RNN+Patterns"
            })
        else:
            return jsonify({
                "status": "⏳ รอรอบต้นชั่วโมงถัดไป",
                "time": now_th.strftime("%Y-%m-%d %H:%M"),
                "next_run": f"{current_hour + 1}:00"
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
            "result": result[:
