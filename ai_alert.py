import os
import time
import joblib
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, Response
from threading import Thread
import io
import base64
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import ta

load_dotenv()

# 🔐 Environment Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

# Load original model (existing system)
try:
    model = joblib.load("xau_model.pkl")
    print("✅ Original XAU model loaded successfully")
except:
    print("⚠️ Original model not found, creating dummy model")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)

app = Flask(__name__)

# Global variables
last_signal = None
last_sent_hour = None
scaler = MinMaxScaler()

FEATURES = ["rsi", "ema", "price_change"]

# ====================== Original System (RSI, EMA, Price Change) ======================

def get_latest_xau():
    """Original function - Get latest XAU data with technical indicators"""
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
        price_url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
        real_res = requests.get(price_url, timeout=15).json()
        
        if 'price' in real_res:
            real_price = float(real_res["price"])
            # Update last close with real-time price
            last_idx = df.index[-1]
            df.at[last_idx, "close"] = real_price

        # Calculate original indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["ema"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
        df["price_change"] = df["close"].pct_change()

        df.dropna(subset=FEATURES, inplace=True)
        return df
        
    except Exception as e:
        print(f"Error in get_latest_xau: {e}")
        return None

def explain_prediction(model, x_vec: np.ndarray, price: float, ema_val: float, rsi_val: float, pred_label: int):
    """Original function - Explain model prediction"""
    confidence = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([x_vec])[0]
            confidence = float(np.max(proba))
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
    """Original function - Calculate TP/SL targets"""
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
    """Original AI function - RSI, EMA, Price Change based"""
    global last_signal
    try:
        latest = get_latest_xau()
        if latest is None:
            return "❌ ไม่สามารถดึงข้อมูลได้"
            
        X_live = latest[FEATURES]

        if X_live.empty or X_live.isnull().values.any():
            price = latest["close"].iloc[0]
            ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
            msg = (
                f"ตอนนี้ วันที่ {ts_txt}\n"
                f"XAUUSD TF H1 ราคาปัจจุบัน {price:,.2f}$\n"
                f"BOT ยังไม่สามารถทำนายได้"
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

        pred = int(model.predict([x])[0])
        reason_text, _ = explain_prediction(model, x, price, ema_val, rsi_val, pred)

        signal, (tp1, tp2, tp3, sl) = calc_targets(pred, price)
        ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")

        msg = (
            f"🤖 ORIGINAL AI BOT (RSI+EMA+Price Change)\n"
            f"ตอนนี้ วันที่ {ts_txt}\n"
            f"XAUUSD TF H1\n"
            f"Open = {o:,.2f}$\n"
            f"High = {h:,.2f}$\n"
            f"Low = {l:,.2f}$\n"
            f"Close = {c:,.2f}$\n"
            f"ราคาปัจจุบัน = {price:,.2f}$\n\n"
            f"Original BOT ทำนาย: {signal}\n"
            f"{reason_text}\n"
            f"🎯 TP1: {tp1:,.2f}$\n"
            f"🎯 TP2: {tp2:,.2f}$\n"
            f"🎯 TP3: {tp3:,.2f}$\n"
            f"🛑 SL: {sl:,.2f}$"
        )
        
        last_signal = signal
        return msg
        
    except Exception as e:
        return f"❌ ORIGINAL BOT ERROR: {e}"

# ====================== New AI Pattern Detection System ======================

class PatternDetector:
    def __init__(self):
        self.patterns = {
            0: "NO_PATTERN",
            1: "HEAD_SHOULDERS", 
            2: "DOUBLE_TOP",
            3: "DOUBLE_BOTTOM",
            4: "ASCENDING_TRIANGLE",
            5: "DESCENDING_TRIANGLE",
            6: "RISING_WEDGE",
            7: "FALLING_WEDGE",
            8: "BULL_FLAG",
            9: "BEAR_FLAG"
        }
        
        # Load pretrained models (simulate loading - replace with actual models)
        self.cnn_model = self._create_dummy_cnn()
        self.rnn_model = self._create_dummy_rnn()
        self.price_predictor = self._create_dummy_predictor()
        
    def _create_dummy_cnn(self):
        """Create a dummy CNN model - replace with actual pretrained model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.patterns), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            return model
        except Exception as e:
            print(f"CNN model creation error: {e}")
            return None
    
    def _create_dummy_rnn(self):
        """Create a dummy RNN model - replace with actual pretrained model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(50, 4)),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(self.patterns), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            return model
        except Exception as e:
            print(f"RNN model creation error: {e}")
            return None
    
    def _create_dummy_predictor(self):
        """Create price prediction model"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(50, 6)),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.Dense(4)  # [entry_price, tp1, tp2, sl]
            ])
            model.compile(optimizer='adam', loss='mse')
            return model
        except Exception as e:
            print(f"Price predictor creation error: {e}")
            return None

    def predict_trading_signals(self, df, pattern_info):
        """Predict entry, TP, SL using pattern and technical indicators"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            df['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            
            # Pattern-based trading logic
            pattern_id = pattern_info['pattern_id']
            confidence = pattern_info['confidence']
            
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
                
            elif pattern_id in [4, 8]:  # ASCENDING_TRIANGLE, BULL_FLAG (Bullish)
                action = "BUY"
                entry_price = current_price + (current_price * 0.001)
                tp1 = current_price * 1.005
                tp2 = current_price * 1.010
                tp3 = current_price * 1.015
                sl = current_price * 0.990
                
            elif pattern_id in [5, 9]:  # DESCENDING_TRIANGLE, BEAR_FLAG (Bearish)
                action = "SELL"
                entry_price = current_price - (current_price * 0.001)
                tp1 = current_price * 0.995
                tp2 = current_price * 0.990
                tp3 = current_price * 0.985
                sl = current_price * 1.010
                
            else:  # NO_PATTERN or others - use indicators
                rsi_current = df['rsi'].iloc[-1] if not df['rsi'].isna().iloc[-1] else 50
                ema_10 = df['ema_10'].iloc[-1]
                
                if rsi_current < 30 and current_price < ema_10:
                    action = "BUY"
                    entry_price = current_price + (current_price * 0.0005)
                    tp1 = current_price * 1.003
                    tp2 = current_price * 1.006
                    tp3 = current_price * 1.009
                    sl = current_price * 0.995
                elif rsi_current > 70 and current_price > ema_10:
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
            
            # Apply confidence adjustment
            if confidence < 0.6:
                action = "WAIT"
                
            return {
                'action': action,
                'entry_price': round(entry_price, 2),
                'tp1': round(tp1, 2),
                'tp2': round(tp2, 2), 
                'tp3': round(tp3, 2),
                'sl': round(sl, 2),
                'confidence': confidence,
                'current_price': round(current_price, 2)
            }
            
        except Exception as e:
            print(f"Trading signal prediction error: {e}")
            current_price = df['close'].iloc[-1]
            return {
                'action': 'WAIT',
                'entry_price': round(current_price, 2),
                'tp1': round(current_price, 2),
                'tp2': round(current_price, 2),
                'tp3': round(current_price, 2),
                'sl': round(current_price, 2),
                'confidence': 0.30,
                'current_price': round(current_price, 2)
            }

    def create_candlestick_image(self, df, save_path=None):
        """Convert OHLC data to candlestick chart image for CNN"""
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create simple candlestick representation
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            colors = ['green' if c >= o else 'red' for c, o in zip(closes, opens)]
            
            for i in range(len(df)):
                # Draw the high-low line
                ax.plot([i, i], [lows[i], highs[i]], color='white', linewidth=1)
                # Draw the body
                body_height = abs(closes[i] - opens[i])
                bottom = min(opens[i], closes[i])
                ax.add_patch(plt.Rectangle((i-0.3, bottom), 0.6, body_height, 
                                         color=colors[i], alpha=0.8))
            
            ax.set_xlim(-1, len(df))
            ax.set_ylim(df['low'].min() * 0.999, df['high'].max() * 1.001)
            ax.axis('off')
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            buf.seek(0)
            
            # Convert to PIL Image and resize
            img = Image.open(buf)
            img = img.resize((64, 64))
            plt.close()
            
            if save_path:
                img.save(save_path)
                
            return np.array(img) / 255.0  # Normalize
            
        except Exception as e:
            print(f"Error creating candlestick image: {e}")
            return np.zeros((64, 64, 3))

    def detect_pattern_cnn(self, df):
        """Detect pattern using CNN on candlestick image"""
        try:
            if self.cnn_model is None:
                return self._fallback_pattern_detection(df)
                
            img_array = self.create_candlestick_image(df)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict using CNN
            prediction = self.cnn_model.predict(img_array, verbose=0)
            pattern_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            return {
                'pattern_id': int(pattern_id),
                'pattern_name': self.patterns[pattern_id],
                'confidence': float(confidence),
                'method': 'CNN'
            }
        except Exception as e:
            print(f"CNN pattern detection error: {e}")
            return self._fallback_pattern_detection(df)

    def detect_pattern_rnn(self, df):
        """Detect pattern using RNN on OHLC sequence"""
        try:
            if self.rnn_model is None:
                return self._fallback_pattern_detection(df)
                
            # Prepare sequence data
            ohlc_data = df[['open', 'high', 'low', 'close']].values[-50:]
            
            if len(ohlc_data) < 50:
                return self._fallback_pattern_detection(df)
                
            ohlc_scaled = scaler.fit_transform(ohlc_data)
            ohlc_scaled = np.expand_dims(ohlc_scaled, axis=0)
            
            # Predict using RNN
            prediction = self.rnn_model.predict(ohlc_scaled, verbose=0)
            pattern_id = np.argmax(prediction)
            confidence = np.max(prediction)
            
            return {
                'pattern_id': int(pattern_id),
                'pattern_name': self.patterns[pattern_id], 
                'confidence': float(confidence),
                'method': 'RNN'
            }
        except Exception as e:
            print(f"RNN pattern detection error: {e}")
            return self._fallback_pattern_detection(df)

    def _fallback_pattern_detection(self, df):
        """Rule-based pattern detection as fallback"""
        try:
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            closes = df['close'].values[-20:]
            
            if len(highs) < 5:
                return {
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'RULE_BASED'
                }
            
            # Simple Head & Shoulders detection
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
                        'pattern_id': 8,
                        'pattern_name': 'BULL_FLAG',
                        'confidence': 0.60,
                        'method': 'RULE_BASED'
                    }
                elif slope < 0:
                    return {
                        'pattern_id': 9,
                        'pattern_name': 'BEAR_FLAG',
                        'confidence': 0.60,
                        'method': 'RULE_BASED'
                    }
            
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.50,
                'method': 'RULE_BASED'
            }
            
        except Exception as e:
            print(f"Fallback pattern detection error: {e}")
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN', 
                'confidence': 0.30,
                'method': 'ERROR'
            }

def get_xau_data_extended():
    """Get extended XAU/USD data for pattern detection"""
    try:
        # Get historical data (100 bars for better pattern detection)
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
        
        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get real-time price
        price_url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
        price_response = requests.get(price_url, timeout=15)
        real_price_data = price_response.json()
        
        if 'price' in real_price_data:
            real_price = float(real_price_data['price'])
            # Update last close with real-time price
            df.iloc[-1, df.columns.get_loc('close')] = real_price
        
        return df
        
    except Exception as e:
        print(f"Error fetching XAU data: {e}")
        return None

def run_pattern_ai():
    """New AI Pattern Detection System"""
    try:
        # Get extended data for pattern detection
        df = get_xau_data_extended()
        if df is None or len(df) < 50:
            return "❌ ไม่สามารถดึงข้อมูลสำหรับ Pattern Detection ได้"
        
        # Initialize pattern detector
        detector = PatternDetector()
        
        # Detect patterns using both CNN and RNN
        cnn_pattern = detector.detect_pattern_cnn(df.tail(50))
        rnn_pattern = detector.detect_pattern_rnn(df.tail(50))
        
        # Choose the pattern with higher confidence
        if cnn_pattern['confidence'] > rnn_pattern['confidence']:
            selected_pattern = cnn_pattern
        else:
            selected_pattern = rnn_pattern
        
        # Get current market data
        current_data = df.iloc[-1]
        current_price = current_data['close']
        current_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        
        # Calculate technical indicators for additional context
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        
        current_rsi = df['rsi'].iloc[-1]
        current_ema10 = df['ema_10'].iloc[-1]
        current_ema21 = df['ema_21'].iloc[-1]
        
        # Pattern-based trading signals
        pattern_signals = calculate_pattern_signals(selected_pattern, current_price, 
                                                  current_rsi, current_ema10, current_ema21)
        
        # Create message
        pattern_emoji = {
            'HEAD_SHOULDERS': '🗣️',
            'DOUBLE_TOP': '⛰️⛰️', 
            'DOUBLE_BOTTOM': '🏔️🏔️',
            'ASCENDING_TRIANGLE': '📈▲',
            'DESCENDING_TRIANGLE': '📉▼',
            'RISING_WEDGE': '📈🔺',
            'FALLING_WEDGE': '📉🔻',
            'BULL_FLAG': '🚩📈',
            'BEAR_FLAG': '🚩📉',
            'NO_PATTERN': '🔍'
        }
        
        pattern_desc = {
            'HEAD_SHOULDERS': 'หัวไหล่ (Bearish Reversal)',
            'DOUBLE_TOP': 'ยอดคู่ (Bearish Reversal)',
            'DOUBLE_BOTTOM': 'ก้นคู่ (Bullish Reversal)', 
            'ASCENDING_TRIANGLE': 'สามเหลี่ยมขาขึ้น (Bullish)',
            'DESCENDING_TRIANGLE': 'สามเหลี่ยมขาลง (Bearish)',
            'RISING_WEDGE': 'เวดจ์ขาขึ้น (Bearish)',
            'FALLING_WEDGE': 'เวดจ์ขาลง (Bullish)',
            'BULL_FLAG': 'ธงวัว (Bullish Continuation)',
            'BEAR_FLAG': 'ธงหมี (Bearish Continuation)',
            'NO_PATTERN': 'ไม่พบแพทเทิร์นชัดเจน'
        }
        
        action_emoji = {
            'BUY': '🟢 BUY',
            'SELL': '🔴 SELL', 
            'WAIT': '🟡 WAIT'
        }
        
        message = f"""🚀 AI PATTERN DETECTION BOT (CNN + RNN)
⏰ {current_time}
💰 XAUUSD (1H Timeframe)

📊 MARKET DATA:
Open: ${current_data['open']:,.2f}
High: ${current_data['high']:,.2f}  
Low: ${current_data['low']:,.2f}
Close: ${current_data['close']:,.2f}

🔍 PATTERN DETECTED:
{pattern_emoji.get(selected_pattern['pattern_name'], '🔍')} {pattern_desc.get(selected_pattern['pattern_name'], selected_pattern['pattern_name'])}
🤖 Detection: {selected_pattern['method']}
🎯 Confidence: {selected_pattern['confidence']*100:.1f}%

📈 TECHNICAL CONTEXT:
RSI: {current_rsi:.1f} ({'Oversold' if current_rsi<30 else 'Overbought' if current_rsi>70 else 'Neutral'})
EMA10: ${current_ema10:.2f} ({'Above' if current_price>current_ema10 else 'Below'})
EMA21: ${current_ema21:.2f} ({'Above' if current_price>current_ema21 else 'Below'})

🚦 PATTERN AI SIGNAL: {action_emoji[pattern_signals['action']]}"""

        if pattern_signals['action'] != 'WAIT':
            message += f"""
💼 PATTERN TRADING SETUP:
🎯 Entry: ${pattern_signals['entry_price']:,.2f}
🟢 TP1: ${pattern_signals['tp1']:,.2f}
🟢 TP2: ${pattern_signals['tp2']:,.2f}  
🟢 TP3: ${pattern_signals['tp3']:,.2f}
🛑 Stop Loss: ${pattern_signals['sl']:,.2f}
📊 Pattern Confidence: {pattern_signals['confidence']*100:.1f}%

⚠️ Risk Management: ใช้เงินเพียง 1-2% ต่อออเดอร์"""
        else:
            message += f"""
⏳ รอ Pattern ที่ชัดเจนกว่า
📊 Current Price: ${current_price:,.2f}
🔍 กำลังวิเคราะห์แพทเทิร์นใหม่..."""

        return message
        
    except Exception as e:
        return f"❌ PATTERN AI ERROR: {str(e)}"

def calculate_pattern_signals(pattern_info, current_price, rsi, ema10, ema21):
    """Calculate trading signals based on detected patterns"""
    pattern_id = pattern_info['pattern_id']
    confidence = pattern_info['confidence']
    
    # Pattern-based trading logic
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
        
    elif pattern_id in [4, 8]:  # ASCENDING_TRIANGLE, BULL_FLAG (Bullish)
        action = "BUY"
        entry_price = current_price + (current_price * 0.001)
        tp1 = current_price * 1.005
        tp2 = current_price * 1.010
        tp3 = current_price * 1.015
        sl = current_price * 0.990
        
    elif pattern_id in [5, 9]:  # DESCENDING_TRIANGLE, BEAR_FLAG (Bearish)
        action = "SELL"
        entry_price = current_price - (current_price * 0.001)
        tp1 = current_price * 0.995
        tp2 = current_price * 0.990
        tp3 = current_price * 0.985
        sl = current_price * 1.010
        
    else:  # NO_PATTERN or others - use indicators
        if rsi < 30 and current_price < ema10:
            action = "BUY"
            entry_price = current_price + (current_price * 0.0005)
            tp1 = current_price * 1.003
            tp2 = current_price * 1.006
            tp3 = current_price * 1.009
            sl = current_price * 0.995
        elif rsi > 70 and current_price > ema10:
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
    
    # Apply confidence adjustment
    if confidence < 0.6:
        action = "WAIT"
        
    return {
        'action': action,
        'entry_price': round(entry_price, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2), 
        'tp3': round(tp3, 2),
        'sl': round(sl, 2),
        'confidence': confidence,
        'current_price': round(current_price, 2)
    }

# ====================== Utilities ======================

def send_telegram(message: str) -> int:
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": CHAT_ID, 
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data, timeout=20)
        return response.status_code
    except Exception as e:
        print(f"Telegram send error: {e}")
        return 500

# ====================== Flask Routes ======================

@app.route('/health', methods=['GET', 'HEAD'])
def health_check():
    """Health check endpoint for monitoring services"""
    return Response("OK", status=200, headers={
        "Content-Type": "text/plain",
        "Cache-Control": "no-cache"
    })
