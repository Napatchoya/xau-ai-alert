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
from flask import Flask, jsonify, Response, request
from threading import Thread
import io
import warnings
warnings.filterwarnings('ignore')
import matplotlib.patches as patches

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

# Environment Variables
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

def get_shared_xau_data():
    """Get shared XAU data for both systems - ใช้ข้อมูลชุดเดียวกัน"""
    try:
        # Get historical data (100 bars เพื่อให้ indicators ถูกต้อง)
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
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get real-time price (เรียกครั้งเดียว)
        try:
            price_url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={API_KEY}"
            price_response = requests.get(price_url, timeout=15)
            real_price_data = price_response.json()
            
            if 'price' in real_price_data:
                real_price = float(real_price_data['price'])
                # Update last close with real-time price
                df.iloc[-1, df.columns.get_loc('close')] = real_price
                print(f"Data updated: Real-time price ${real_price:,.2f}")
        except Exception as e:
            print(f"⚠️ Real-time price fetch failed: {e}")
        
        # Calculate shared indicators
        df["rsi"] = calculate_rsi(df["close"])
        df["ema"] = calculate_ema(df["close"], 10)
        df["ema_21"] = calculate_ema(df["close"], 21)
        df["price_change"] = df["close"].pct_change()
        
        return df
        
    except Exception as e:
        print(f"Error fetching shared XAU data: {e}")
        return None

# ====================== Chart Generation Functions ======================

def create_candlestick_chart(df, trading_signals, pattern_info):
    """Create candlestick chart with pattern lines and trading levels"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import matplotlib.dates as mdates
        
        # Use last 50 candles for better visibility
        chart_df = df.tail(50).copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Set dark theme
        fig.patch.set_facecolor('#1a1a1a')
        ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('#1a1a1a')
        
        # Main candlestick chart
        for i, (idx, row) in enumerate(chart_df.iterrows()):
            color = '#00ff88' if row['close'] >= row['open'] else '#ff4444'
            
            # Draw candle body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            ax1.add_patch(patches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, edgecolor=color, alpha=0.8
            ))
            
            # Draw wicks
            ax1.plot([i, i], [row['low'], row['high']], 
                    color=color, linewidth=1, alpha=0.7)
        
        # Plot EMAs
        if 'ema' in chart_df.columns:
            ax1.plot(range(len(chart_df)), chart_df['ema'].values, 
                    color='#ffaa00', linewidth=2, label='EMA 10', alpha=0.8)
        if 'ema_21' in chart_df.columns:
            ax1.plot(range(len(chart_df)), chart_df['ema_21'].values, 
                    color='#ff6600', linewidth=2, label='EMA 21', alpha=0.8)
        
        # Add trading levels
        current_price = trading_signals['current_price']
        entry_price = trading_signals['entry_price']
        tp1, tp2, tp3 = trading_signals['tp1'], trading_signals['tp2'], trading_signals['tp3']
        sl = trading_signals['sl']
        
        # Draw horizontal lines for trading levels
        x_range = range(len(chart_df))
        
        # Entry line
        ax1.axhline(y=entry_price, color='#ffffff', linestyle='--', 
                   linewidth=2, alpha=0.9, label=f'Entry: ${entry_price}')
        
        # TP lines
        ax1.axhline(y=tp1, color='#00ff88', linestyle='-', 
                   linewidth=1.5, alpha=0.7, label=f'TP1: ${tp1}')
        ax1.axhline(y=tp2, color='#00dd66', linestyle='-', 
                   linewidth=1.5, alpha=0.7, label=f'TP2: ${tp2}')
        ax1.axhline(y=tp3, color='#00bb44', linestyle='-', 
                   linewidth=1.5, alpha=0.7, label=f'TP3: ${tp3}')
        
        # SL line
        ax1.axhline(y=sl, color='#ff4444', linestyle='-', 
                   linewidth=2, alpha=0.8, label=f'SL: ${sl}')
        
        # Add pattern detection lines
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        if pattern_name != 'NO_PATTERN':
            draw_pattern_lines(ax1, chart_df, pattern_name)
        
        # Add support/resistance levels
        draw_support_resistance(ax1, chart_df)
        
        # Style main chart
        ax1.set_title(f'XAU/USD - Pattern: {pattern_name} | Signal: {trading_signals["action"]}', 
                     color='#ffffff', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', color='#ffffff', fontsize=12)
        ax1.tick_params(colors='#ffffff')
        ax1.grid(True, alpha=0.3, color='#444444')
        ax1.legend(loc='upper left', facecolor='#2a2a2a', edgecolor='#444444', 
                  labelcolor='#ffffff')
        
        # RSI subplot
        if 'rsi' in chart_df.columns:
            rsi_values = chart_df['rsi'].dropna()
            ax2.plot(range(len(rsi_values)), rsi_values.values, 
                    color='#00aaff', linewidth=2, label='RSI')
            ax2.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(y=50, color='#888888', linestyle='-', alpha=0.5)
            
            ax2.set_ylabel('RSI', color='#ffffff', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.tick_params(colors='#ffffff')
            ax2.grid(True, alpha=0.3, color='#444444')
            ax2.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='#444444', 
                      labelcolor='#ffffff')
        
        # Format x-axis
        ax1.set_xlim(-1, len(chart_df))
        ax2.set_xlim(-1, len(chart_df))
        
        # Add timestamp
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        fig.text(0.02, 0.02, f"Generated: {timestamp} (Bangkok)", 
                color='#888888', fontsize=10)
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#1a1a1a', 
                   edgecolor='none', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Chart creation error: {e}")
        return None

def draw_pattern_lines(ax, df, pattern_name):
    """Draw pattern-specific lines on chart"""
    try:
        if pattern_name == 'HEAD_SHOULDERS':
            # Find peaks for head and shoulders
            highs = df['high'].values
            if len(highs) >= 20:
                # Simplified head and shoulders pattern
                mid_point = len(highs) // 2
                left_shoulder = np.argmax(highs[max(0, mid_point-10):mid_point])
                head = np.argmax(highs[mid_point-5:mid_point+5]) + mid_point - 5
                right_shoulder = np.argmax(highs[mid_point:mid_point+10]) + mid_point
                
                # Draw neckline
                if left_shoulder < head < right_shoulder:
                    neckline_y = (df['low'].iloc[left_shoulder] + df['low'].iloc[right_shoulder]) / 2
                    ax.axhline(y=neckline_y, color='#ff00ff', linestyle=':', 
                              linewidth=2, alpha=0.8, label='Neckline')
                    
                    # Mark the pattern points
                    ax.scatter([left_shoulder, head, right_shoulder], 
                              [highs[left_shoulder], highs[head], highs[right_shoulder]], 
                              color='#ff00ff', s=60, alpha=0.8, marker='^')
        
        elif pattern_name == 'DOUBLE_TOP':
            # Find two highest peaks
            highs = df['high'].values
            peaks = []
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                # Take last two peaks
                peak1, peak2 = peaks[-2], peaks[-1]
                ax.scatter([peak1[0], peak2[0]], [peak1[1], peak2[1]], 
                          color='#ff6600', s=60, alpha=0.8, marker='v')
                
                # Draw resistance line
                ax.plot([peak1[0], peak2[0]], [peak1[1], peak2[1]], 
                       color='#ff6600', linestyle='--', linewidth=2, alpha=0.8, label='Resistance')
        
        elif pattern_name == 'ASCENDING_TRIANGLE':
            # Draw ascending triangle pattern
            highs = df['high'].values
            lows = df['low'].values
            
            # Find resistance level (horizontal)
            resistance = np.max(highs[-20:])
            ax.axhline(y=resistance, color='#00ffff', linestyle='-', 
                      linewidth=2, alpha=0.8, label='Resistance')
            
            # Find ascending support line
            recent_lows = [(i, lows[i]) for i in range(len(lows)-20, len(lows)) 
                          if i > 0 and lows[i] < lows[i-1]]
            if len(recent_lows) >= 2:
                x_vals = [p[0] for p in recent_lows[-2:]]
                y_vals = [p[1] for p in recent_lows[-2:]]
                ax.plot(x_vals, y_vals, color='#00ffff', linestyle='-', 
                       linewidth=2, alpha=0.8, label='Support')
        
    except Exception as e:
        print(f"Pattern line drawing error: {e}")

def draw_support_resistance(ax, df):
    """Draw support and resistance levels"""
    try:
        # Calculate pivot points
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Find recent swing highs and lows
        swing_highs = []
        swing_lows = []
        
        lookback = min(20, len(df))
        
        for i in range(2, lookback-2):
            # Swing high
            if (highs[-(i+1)] > highs[-(i+2)] and highs[-(i+1)] > highs[-i] and
                highs[-(i+1)] > highs[-(i+3)] and highs[-(i+1)] > highs[-(i-1)]):
                swing_highs.append(highs[-(i+1)])
            
            # Swing low
            if (lows[-(i+1)] < lows[-(i+2)] and lows[-(i+1)] < lows[-i] and
                lows[-(i+1)] < lows[-(i+3)] and lows[-(i+1)] < lows[-(i-1)]):
                swing_lows.append(lows[-(i+1)])
        
        # Draw support levels (from swing lows)
        for level in swing_lows[-3:]:  # Show last 3 support levels
            ax.axhline(y=level, color='#00ff88', linestyle=':', 
                      linewidth=1, alpha=0.6)
        
        # Draw resistance levels (from swing highs)
        for level in swing_highs[-3:]:  # Show last 3 resistance levels
            ax.axhline(y=level, color='#ff4444', linestyle=':', 
                      linewidth=1, alpha=0.6)
        
        # Add labels
        if swing_lows:
            ax.text(0.02, 0.02, f'Support: ${swing_lows[-1]:.2f}', 
                   transform=ax.transAxes, color='#00ff88', fontsize=10)
        if swing_highs:
            ax.text(0.02, 0.95, f'Resistance: ${swing_highs[-1]:.2f}', 
                   transform=ax.transAxes, color='#ff4444', fontsize=10)
            
    except Exception as e:
        print(f"Support/Resistance drawing error: {e}")

def get_pattern_description(pattern_name):
    """Get detailed pattern description with BOT prediction - COMPLETE VERSION"""
    bot_signal = get_pattern_signal(pattern_name)
    
    base_descriptions = {
        'HEAD_SHOULDERS': f"""📊 HEAD & SHOULDERS PATTERN:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• รูปแบบกลับตัวแบบ Bearish (ลดลง)
• ประกอบด้วย 3 จุดสูง: ไหล่ซ้าย - หัว - ไหล่ขวา
• หัวสูงกว่าไหล่ทั้งสองข้าง
• เส้น Neckline เป็นแนวรับสำคัญ

📈 สัญญาณ:
• เมื่อราคาทะลุ Neckline ลงมา = สัญญาณ SELL
• Target = ระยะทางจากหัวถึง Neckline
• ปริมาณการซื้อขายลดลงที่ไหล่ขวา

⚠️ ความเสี่ยง: รอให้ทะลุ Neckline ก่อนเข้า SELL""",

        'DOUBLE_TOP': f"""📊 DOUBLE TOP PATTERN:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• รูปแบบกลับตัวแบบ Bearish (ลดลง)
• มี 2 จุดสูงใกล้เคียงกัน
• ระหว่างจุดสูงมี Valley (หุบเขา)
• แนวรับที่ Valley = Support สำคัญ

📈 สัญญาณ:
• เมื่อราคาทะลุ Support ที่ Valley = สัญญาณ SELL  
• Target = ระยะทางจาก Peak ถึง Valley
• ปริมาณการซื้อขายลดลงที่ Top ที่ 2

⚠️ ความเสี่ยง: False breakout เกิดได้ง่าย""",

        'DOUBLE_BOTTOM': f"""📊 DOUBLE BOTTOM PATTERN:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• รูปแบบกลับตัวแบบ Bullish (เพิ่มขึ้น)
• มี 2 จุดต่ำใกล้เคียงกัน
• ระหว่างจุดต่ำมี Peak (ยอดเขา)
• แนวต้านที่ Peak = Resistance สำคัญ

📈 สัญญาณ:
• เมื่อราคาทะลุ Resistance ที่ Peak = สัญญาณ BUY
• Target = ระยะทางจาก Bottom ถึง Peak  
• ปริมาณการซื้อขายเพิ่มขึ้นตอน Breakout

⚠️ ความเสี่ยง: ต้องรอการยืนยันการทะลุ""",

        'ASCENDING_TRIANGLE': f"""📊 ASCENDING TRIANGLE:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• รูปแบบ Continuation แบบ Bullish
• แนวต้านแนวนอน (Horizontal Resistance)
• แนวรับทะยานขึ้น (Ascending Support)  
• ปริมาณการซื้อขายค่อยๆ ลดลง

📈 สัญญาณ:
• เมื่อราคาทะลุ Resistance = สัญญาณ BUY
• Target = ความสูงของรูปสามเหลี่ยม
• Stop Loss ใต้แนวรับล่าสุด

⚠️ ความเสี่ยง: อาจ False Breakout ได้""",

        'BULL_FLAG': f"""📊 BULL FLAG PATTERN:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• รูปแบบ Continuation แบบ Bullish
• เกิดหลังการขึ้นแรง (Flagpole)
• ช่วง Consolidation รูปสี่เหลี่ยม
• ปริมาณการซื้อขายลดลงในช่วง Flag

📈 สัญญาณ:  
• เมื่อราคาทะลุ Flag ขึ้นไป = สัญญาณ BUY
• Target = ความยาวของ Flagpole + Breakout Point
• Entry หลังจาก Breakout พร้อม Volume

⚠️ ความเสี่ยง: ระยะเวลา Flag ไม่ควรเกิน 3 สัปดาห์""",

        'HAMMER': f"""📊 HAMMER CANDLESTICK:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• ตัวแท่งเล็ก เงาล่างยาว เงาบนสั้น
• รูปร่างเหมือนค้อน
• เกิดหลัง Downtrend
• สัญญาณ Bullish Reversal

📈 สัญญาณ:
• การกลับตัวจากแนวรับ
• แท่งถัดไปขึ้น = ยืนยัน BUY
• Stop Loss ใต้ Low ของ Hammer

⚠️ ความเสี่ยง: ต้องมีการยืนยัน""",

        'SHOOTING_STAR': f"""📊 SHOOTING STAR CANDLESTICK:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• ตัวแท่งเล็ก เงาบนยาว เงาล่างสั้น
• รูปร่างเหมือนดาวตก
• เกิดหลัง Uptrend
• สัญญาณ Bearish Reversal

📈 สัญญาณ:
• การกลับตัวจากแนวต้าน
• แท่งถัดไปลง = ยืนยัน SELL
• Stop Loss เหนือ High ของ Shooting Star

⚠️ ความเสี่ยง: ต้องมีการยืนยัน""",

        'DOJI': f"""📊 DOJI CANDLESTICK:
🎯BOTทำนาย: {bot_signal}

🔍 คุณสมบัติ:
• แท่งเทียนตัวเล็ก (เปิด-ปิดใกล้กัน)
• เงาบนและล่างยาว
• แสดงความไม่แน่ใจของตลาด
• สัญญาณ Indecision

📈 สัญญาณ:
• ที่จุดสูง = อาจกลับตัวลง
• ที่จุดต่ำ = อาจกลับตัวขึ้น  
• ต้องดูแท่งถัดไปเป็นการยืนยัน

⚠️ ความเสี่ยง: ไม่ใช่สัญญาณแน่นอน""",

        'NO_PATTERN': f"""📊 NO CLEAR PATTERN:
🎯BOTทำนาย: {bot_signal}

🔍 สถานการณ์ปัจจุบัน:
• ไม่พบแพทเทิร์นชัดเจน
• ตลาดอาจอยู่ในช่วง Sideways
• รอการก่อตัวของแพทเทิร์นใหม่

📈 คำแนะนำ:
• รอจังหวะที่ชัดเจนกว่า
• เฝ้าดูแนวรับแนวต้านสำคัญ
• ใช้ Technical Indicators ประกอบ

⚠️ ควรระมัดระวัง: ตลาด Sideways เสี่ยง Whipsaw"""
    }

    if pattern_name in base_descriptions:
        return base_descriptions[pattern_name]
    else:
        # เรียกใช้ extended patterns
        extended_descriptions = get_extended_pattern_descriptions_with_signals()
        return extended_descriptions.get(pattern_name, f"ไม่มีข้อมูลแพทเทิร์นนี้\n🎯BOTทำนาย: {bot_signal}")
    
# ====================== Extended Pattern Descriptions ======================
extended_patterns = [
    'DESCENDING_TRIANGLE', 'SYMMETRICAL_TRIANGLE', 'BEAR_FLAG', 'WEDGE_RISING', 
    'WEDGE_FALLING', 'CUP_AND_HANDLE', 'RECTANGLE', 'DOJI', 'HAMMER', 
    'SHOOTING_STAR', 'ENGULFING_BULLISH', 'ENGULFING_BEARISH', 'MORNING_STAR',
    'EVENING_STAR', 'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS', 'MARUBOZU',
    'PENNANT', 'INVERSE_HEAD_SHOULDERS', 'DIAMOND', 'HANGING_MAN', 
    'INVERTED_HAMMER', 'SPINNING_TOP', 'PIERCING_LINE', 'DARK_CLOUD_COVER',
    'HARAMI_BULLISH', 'HARAMI_BEARISH', 'TWEEZER_TOP', 'TWEEZER_BOTTOM'
]

def get_extended_pattern_descriptions_with_signals():
    """Extended pattern descriptions with BOT predictions - COMPLETE ALL PATTERNS"""
    extended_descriptions = {
        'DESCENDING_TRIANGLE': f"""📊 DESCENDING TRIANGLE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('DESCENDING_TRIANGLE')}

🔍 คุณสมบัติ:
• รูปแบบ Continuation แบบ Bearish
• แนวรับแนวนอน (Horizontal Support)
• แนวต้านลาดลง (Descending Resistance)
• ปริมาณการซื้อขายค่อยๆ ลดลง

📈 สัญญาณ:
• เมื่อราคาทะลุ Support = สัญญาณ SELL
• Target = ความสูงของรูปสามเหลี่ยม
• Stop Loss เหนือแนวต้านล่าสุด

⚠️ ความเสี่ยง: อาจ False Breakout ได้""",

        'BEAR_FLAG': f"""📊 BEAR FLAG PATTERN:
🎯BOTทำนาย: {get_pattern_signal('BEAR_FLAG')}

🔍 คุณสมบัติ:
• รูปแบบ Continuation แบบ Bearish
• เกิดหลังการลดลงแรง (Flagpole)
• ช่วง Consolidation รูปธงเล็กๆ
• ปริมาณการซื้อขายลดลงในช่วง Flag

📈 สัญญาณ:
• เมื่อราคาทะลุ Flag ลงไป = สัญญาณ SELL
• Target = ความยาวของ Flagpole + Breakout Point
• Entry หลังจาก Breakdown พร้อม Volume

⚠️ ความเสี่ยง: ระยะเวลา Flag ไม่ควรเกิน 3 สัปดาห์""",

        'WEDGE_RISING': f"""📊 RISING WEDGE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('WEDGE_RISING')}

🔍 คุณสมบัติ:
• รูปแบบ Reversal แบบ Bearish
• ทั้งแนวรับและแนวต้านลาดขึ้น
• แนวรับขึ้นชันกว่าแนวต้าน (บีบตัว)
• ปริมาณการซื้อขายลดลง

📈 สัญญาณ:
• เมื่อราคาทะลุแนวรับ = สัญญาณ SELL
• Target = ความสูงของ Wedge
• มักเกิดหลังจาก Uptrend ที่แรง

⚠️ ความเสี่ยง: สัญญาณ Reversal ที่แรง""",

        'WEDGE_FALLING': f"""📊 FALLING WEDGE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('WEDGE_FALLING')}

🔍 คุณสมบัติ:
• รูปแบบ Reversal แบบ Bullish
• ทั้งแนวรับและแนวต้านลาดลง
• แนวต้านลงชันกว่าแนวรับ (บีบตัว)
• ปริมาณการซื้อขายลดลง

📈 สัญญาณ:
• เมื่อราคาทะลุแนวต้าน = สัญญาณ BUY
• Target = ความสูงของ Wedge
• มักเกิดหลังจาก Downtrend ที่แรง

⚠️ ความเสี่ยง: ต้องรอการยืนยันการทะลุ""",

        'CUP_AND_HANDLE': f"""📊 CUP AND HANDLE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('CUP_AND_HANDLE')}

🔍 คุณสมบัติ:
• รูปแบบ Continuation แบบ Bullish
• Cup รูปตัว U + Handle รูปธงเล็ก
• ใช้เวลาสร้างนาน (สัปดาห์-เดือน)
• ปริมาณการซื้อขายลดลงใน Handle

📈 สัญญาณ:
• เมื่อราคาทะลุ Handle = สัญญาณ BUY
• Target = ความลึกของ Cup + Breakout
• Entry พร้อม Volume Confirmation

⚠️ ความเสี่ยง: Pattern ใหญ่ ต้องใช้เวลานาน""",

        'INVERSE_HEAD_SHOULDERS': f"""📊 INVERSE HEAD & SHOULDERS:
🎯BOTทำนาย: {get_pattern_signal('INVERSE_HEAD_SHOULDERS')}

🔍 คุณสมบัติ:
• รูปแบบกลับตัวแบบ Bullish
• 3 จุดต่ำ: ไหล่ซ้าย-หัว-ไหล่ขวา
• หัวต่ำกว่าไหล่ทั้งสองข้าง
• Neckline เป็นแนวต้านสำคัญ

📈 สัญญาณ:
• ทะลุ Neckline ขึ้น = สัญญาณ BUY
• Target = ระยะทางจากหัวถึง Neckline
• Volume เพิ่มขึ้นตอน Breakout

⚠️ ความเสี่ยง: รอการยืนยันการทะลุ""",

        'ENGULFING_BULLISH': f"""📊 BULLISH ENGULFING:
🎯BOTทำนาย: {get_pattern_signal('ENGULFING_BULLISH')}

🔍 คุณสมบัติ:
• แท่งดำ + แท่งขาวใหญ่
• แท่งขาวครอบแท่งดำ
• เกิดหลัง Downtrend
• สัญญาณ Bullish Reversal แรง

📈 สัญญาณ:
• เปลี่ยนแปลงโมเมนตัมชัดเจน
• Entry หลัง Engulfing Candle
• Target ตามแนวต้านถัดไป

⚠️ ความเสี่ยง: Volume ต้องสูงด้วย""",

        'ENGULFING_BEARISH': f"""📊 BEARISH ENGULFING:
🎯BOTทำนาย: {get_pattern_signal('ENGULFING_BEARISH')}

🔍 คุณสมบัติ:
• แท่งขาว + แท่งดำใหญ่
• แท่งดำครอบแท่งขาว
• เกิดหลัง Uptrend
• สัญญาณ Bearish Reversal แรง

📈 สัญญาณ:
• เปลี่ยนแปลงโมเมนตัมชัดเจน
• Entry หลัง Engulfing Candle
• Target ตามแนวรับถัดไป

⚠️ ความเสี่ยง: Volume ต้องสูงด้วย""",

        'MORNING_STAR': f"""📊 MORNING STAR:
🎯BOTทำนาย: {get_pattern_signal('MORNING_STAR')}

🔍 คุณสมบัติ:
• 3 แท่ง: ดำ-เล็ก-ขาว
• แท่งกลางเป็น Doji หรือ Spinning Top
• เกิดหลัง Downtrend
• สัญญาณ Bullish Reversal แรง

📈 สัญญาณ:
• ดาวรุ่งนำทางขาขึ้น
• Confirmation แรงกว่า Hammer
• Target ตามแนวต้านหลัก

⚠️ ความเสี่ยง: ต้องเกิดที่แนวรับสำคัญ""",

        'EVENING_STAR': f"""📊 EVENING STAR:
🎯BOTทำนาย: {get_pattern_signal('EVENING_STAR')}

🔍 คุณสมบัติ:
• 3 แท่ง: ขาว-เล็ก-ดำ
• แท่งกลางเป็น Doji หรือ Spinning Top  
• เกิดหลัง Uptrend
• สัญญาณ Bearish Reversal แรง

📈 สัญญาณ:
• ดาวค่ำประกาศขาลง
• Confirmation แรงกว่า Shooting Star
• Target ตามแนวรับหลัก

⚠️ ความเสี่ยง: ต้องเกิดที่แนวต้านสำคัญ""",

        'THREE_WHITE_SOLDIERS': f"""📊 THREE WHITE SOLDIERS:
🎯BOTทำนาย: {get_pattern_signal('THREE_WHITE_SOLDIERS')}

🔍 คุณสมบัติ:
• 3 แท่งขาวติดกัน
• แต่ละแท่งปิดสูงขึ้น
• เงาสั้น แสดงความแรง
• สัญญาณ Bullish Continuation

📈 สัญญาณ:
• โมเมนตัมขาขึ้นแรงมาก
• มักเกิดหลัง Consolidation
• Target ขึ้นตาม Momentum

⚠️ ความเสี่ยง: อาจ Overbought ระยะสั้น""",

        'THREE_BLACK_CROWS': f"""📊 THREE BLACK CROWS:
🎯BOTทำนาย: {get_pattern_signal('THREE_BLACK_CROWS')}

🔍 คุณสมบัติ:
• 3 แท่งดำติดกัน
• แต่ละแท่งปิดต่ำลง
• เงาสั้น แสดงความแรง
• สัญญาณ Bearish Continuation

📈 สัญญาณ:
• โมเมนตัมขาลงแรงมาก
• มักเกิดหลัง Consolidation
• Target ลงตาม Momentum

⚠️ ความเสี่ยง: อาจ Oversold ระยะสั้น""",

        'PIERCING_LINE': f"""📊 PIERCING LINE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('PIERCING_LINE')}

🔍 คุณสมบัติ:
• แท่งดำ + แท่งขาวทะลุ
• แท่งขาวเปิดใต้ Low แท่งดำ
• แท่งขาวปิดเหนือ Midpoint แท่งดำ
• สัญญาณกลับตัวแบบ Bullish

📈 สัญญาณ:
• การเข้ามาซื้อที่แรง
• Entry หลัง Piercing Candle
• Target ตามแนวต้านถัดไป

⚠️ ความเสี่ยง: ต้องมี Volume สนับสนุน""",

        'DARK_CLOUD_COVER': f"""📊 DARK CLOUD COVER:
🎯BOTทำนาย: {get_pattern_signal('DARK_CLOUD_COVER')}

🔍 คุณสมบัติ:
• แท่งขาว + แท่งดำครอบ
• แท่งดำเปิดเหนือ High แท่งขาว
• แท่งดำปิดใต้ Midpoint แท่งขาว
• สัญญาณกลับตัวแบบ Bearish

📈 สัญญาณ:
• การเข้ามาขายที่แรง
• Entry หลัง Dark Cloud Candle
• Target ตามแนวรับถัดไป

⚠️ ความเสี่ยง: ต้องมี Volume ยืนยัน""",

        'HARAMI_BULLISH': f"""📊 BULLISH HARAMI:
🎯BOTทำนาย: {get_pattern_signal('HARAMI_BULLISH')}

🔍 คุณสมบัติ:
• แท่งดำใหญ่ + แท่งขาวเล็กข้างใน
• แท่งลูกอยู่ในช่วงของแท่งแม่
• เกิดหลัง Downtrend
• สัญญาณ Indecision นำไปสู่ Reversal

📈 สัญญาณ:
• ความเบื่อหน่ายในการขาย
• รอแท่งถัดไปยืนยัน
• Stop Loss ใต้ Low ของ Mother Candle

⚠️ ความเสี่ยง: สัญญาณอ่อนกว่า Engulfing""",

        'HARAMI_BEARISH': f"""📊 BEARISH HARAMI:
🎯BOTทำนาย: {get_pattern_signal('HARAMI_BEARISH')}

🔍 คุณสมบัติ:
• แท่งขาวใหญ่ + แท่งดำเล็กข้างใน
• แท่งลูกอยู่ในช่วงของแท่งแม่
• เกิดหลัง Uptrend
• สัญญาณ Indecision นำไปสู่ Reversal

📈 สัญญาณ:
• ความเบื่อหน่ายในการซื้อ
• รอแท่งถัดไปยืนยัน
• Stop Loss เหนือ High ของ Mother Candle

⚠️ ความเสี่ยง: สัญญาณอ่อนกว่า Engulfing""",

        'TWEEZER_TOP': f"""📊 TWEEZER TOP PATTERN:
🎯BOTทำนาย: {get_pattern_signal('TWEEZER_TOP')}

🔍 คุณสมบัติ:
• 2 แท่งที่มี High ใกล้เคียงกัน
• เกิดที่แนวต้านสำคัญ
• แสดงการปฏิเสธระดับราคา
• สัญญาณกลับตัวแบบ Bearish

📈 สัญญาณ:
• ไม่สามารถทะลุแนวต้านได้
• Entry หลังแท่งที่ 2 ปิด
• Target ตามแนวรับถัดไป

⚠️ ความเสี่ยง: ต้องเกิดที่แนวต้านสำคัญ""",

        'TWEEZER_BOTTOM': f"""📊 TWEEZER BOTTOM PATTERN:
🎯BOTทำนาย: {get_pattern_signal('TWEEZER_BOTTOM')}

🔍 คุณสมบัติ:
• 2 แท่งที่มี Low ใกล้เคียงกัน
• เกิดที่แนวรับสำคัญ
• แสดงการสนับสนุนระดับราคา
• สัญญาณกลับตัวแบบ Bullish

📈 สัญญาณ:
• ไม่สามารถทะลุแนวรับได้
• Entry หลังแท่งที่ 2 ปิด
• Target ตามแนวต้านถัดไป

⚠️ ความเสี่ยง: ต้องเกิดที่แนวรับสำคัญ""",

        'INVERTED_HAMMER': f"""📊 INVERTED HAMMER:
🎯BOTทำนาย: {get_pattern_signal('INVERTED_HAMMER')}

🔍 คุณสมบัติ:
• ตัวเล็ก เงาบนยาว เงาล่างสั้น
• เกิดหลัง Downtrend
• สัญญาณกลับตัวแบบ Bullish
• คล้าย Shooting Star แต่บริบทต่างกัน

📈 สัญญาณ:
• การกลับตัวจากแนวรับ
• แท่งถัดไปขึ้น = ยืนยัน BUY
• Target ตามแนวต้านถัดไป

⚠️ ความเสี่ยง: ต้องรอการยืนยัน""",

        'MARUBOZU': f"""📊 MARUBOZU CANDLESTICK:
🎯BOTทำนาย: {get_pattern_signal('MARUBOZU')}

🔍 คุณสมบัติ:
• แท่งเทียนตัวใหญ่ ไม่มีเงาบนและล่าง
• เปิด = ต่ำสุด, ปิด = สูงสุด (Bullish Marubozu)
• หรือ เปิด = สูงสุด, ปิด = ต่ำสุด (Bearish Marubozu)
• แสดงความแรงของโมเมนตัมที่ชัดเจน

📈 สัญญาณ:
• White Marubozu = สัญญาณ BUY ที่แรงมาก
• Black Marubozu = สัญญาณ SELL ที่แรงมาก
• ไม่มีการลังเลในการซื้อ/ขาย
• โมเมนตัมต่อเนื่องแนวโน้มสูง

⚠️ ความเสี่ยง: อาจเกิด Reversal หลังโมเมนตัมแรง""",

        'HANGING_MAN': f"""📊 HANGING MAN CANDLESTICK:
🎯BOTทำนาย: {get_pattern_signal('HANGING_MAN')}

🔍 คุณสมบัติ:
• คล้าย Hammer แต่ในบริบท Bearish
• ตัวเล็ก เงาล่างยาว เงาบนสั้น
• เกิดหลัง Uptrend
• สัญญาณเตือน Reversal

📈 สัญญาณ:
• เตือนการกลับตัวจากแนวต้าน
• ต้องรอแท่งถัดไปยืนยัน
• Stop Loss เหนือ High ของแท่ง

⚠️ ความเสี่ยง: ต้องมี Volume ยืนยัน""",

        'SPINNING_TOP': f"""📊 SPINNING TOP CANDLESTICK:
🎯BOTทำนาย: {get_pattern_signal('SPINNING_TOP')}

🔍 คุณสมบัติ:
• ตัวแท่งเล็ก เงาบนล่างยาว
• ความไม่แน่ใจของตลาด
• อาจเกิดได้ทั้งขาขึ้นและขาลง
• สัญญาณ Consolidation

📈 สัญญาณ:
• รอแท่งถัดไปกำหนดทิศทาง
• ถ้าอยู่ในเทรนด์ = อาจพัก
• ที่จุดสำคัญ = อาจกลับตัว

⚠️ ความเสี่ยง: ไม่ใช่สัญญาณที่ชัดเจน""",

        'RECTANGLE': f"""📊 RECTANGLE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('RECTANGLE')}

🔍 คุณสมบัติ:
• รูปแบบ Continuation (Trading Range)
• แนวต้าน-แนวรับแนวนอนชัดเจน
• ราคาเด้งไปมาระหว่างแนว
• ปริมาณการซื้อขายปกติ

📈 สัญญาณ:
• ทะลุขึ้น = BUY, ทะลุลง = SELL
• Target = ความสูงของ Rectangle
• เทรดได้ทั้ง Range และ Breakout

⚠️ ความเสี่ยง: False Breakout เกิดง่าย""",

        'SYMMETRICAL_TRIANGLE': f"""📊 SYMMETRICAL TRIANGLE PATTERN:
🎯BOTทำนาย: {get_pattern_signal('SYMMETRICAL_TRIANGLE')}

🔍 คุณสมบัติ:
• รูปแบบ Continuation (ตามทิศทางเดิม)
• แนวต้านลาดลง + แนวรับลาดขึ้น
• ราคาบีบตัวเข้าหาจุดยอด
• ปริมาณการซื้อขายลดลง

📈 สัญญาณ:
• ทิศทางทะลุขึ้นกับเทรนด์หลัก
• Entry หลัง Breakout พร้อม Volume
• Target = ความสูงของรูปสามเหลี่ยม

⚠️ ความเสี่ยง: ต้องรอ Breakout ที่ชัดเจน""",

        'DIAMOND': f"""📊 DIAMOND PATTERN:
🎯BOTทำนาย: {get_pattern_signal('DIAMOND')}

🔍 คุณสมบัติ:
• ความผันผวนขยายตัวแล้วหดตัว
• รูปเพชร (ขยาย-หด)
• เกิดที่จุดสูงหรือต่ำสำคัญ
• Volume ลดลงตอนปลาย

📈 สัญญาณ:
• Breakout บ่งบอกทิศทางใหม่
• มักเป็นสัญญาณ Reversal
• Target ตามขนาดของเพชร

⚠️ ความเสี่ยง: Pattern ที่หายาก แต่แม่นยำสูง""",

        'PENNANT': f"""📊 PENNANT PATTERN:
🎯BOTทำนาย: {get_pattern_signal('PENNANT')}

🔍 คุณสมบัติ:
• เกิดหลังการเคลื่อนไหวแรง (Flagpole)
• รูปสามเหลี่ยมเล็กที่บีบตัว
• เส้นแนวโน้มบนและล่างมาบรรจบกัน
• ปริมาณการซื้อขายลดลง

📈 สัญญาณ:
• Breakout ตามทิศทางเดิมของ Flagpole
• Target = ความยาวของ Flagpole
• การทะลุพร้อม Volume สูง

⚠️ ความเสี่ยง: False breakout ในตลาด Sideways"""

    }
    return extended_descriptions  

def send_multiple_patterns_message(all_patterns, shared_df):
    """ส่งข้อความรูปแบบใหม่แยกตาม pattern categories - Updated Version"""
    try:
        current_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        current_data = shared_df.iloc[-1]
        
        # สร้าง detector instance เพียงครั้งเดียว
        detector = AdvancedPatternDetector()
        trading_signals = detector.predict_signals(shared_df)
        
        # จำแนกประเภท patterns
        reversal_patterns = []
        continuation_patterns = []
        bearish_patterns = []
        bullish_patterns = []
        neutral_patterns = []
        
        # กำหนด pattern categories
        reversal_list = [
            'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'DOJI', 'HAMMER', 
            'SHOOTING_STAR', 'WEDGE_RISING', 'WEDGE_FALLING', 'INVERSE_HEAD_SHOULDERS',
            'ENGULFING_BULLISH', 'ENGULFING_BEARISH', 'MORNING_STAR', 'EVENING_STAR',
            'PIERCING_LINE', 'DARK_CLOUD_COVER', 'HANGING_MAN', 'INVERTED_HAMMER'
        ]
        
        continuation_list = [
            'BULL_FLAG', 'BEAR_FLAG', 'ASCENDING_TRIANGLE', 'DESCENDING_TRIANGLE',
            'CUP_AND_HANDLE', 'PENNANT', 'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS'
        ]
        
        bearish_list = [
            'HEAD_SHOULDERS', 'DOUBLE_TOP', 'BEAR_FLAG', 'DESCENDING_TRIANGLE',
            'WEDGE_RISING', 'SHOOTING_STAR', 'EVENING_STAR', 'ENGULFING_BEARISH',
            'THREE_BLACK_CROWS', 'DARK_CLOUD_COVER', 'HANGING_MAN', 'TWEEZER_TOP'
        ]
        
        bullish_list = [
            'DOUBLE_BOTTOM', 'BULL_FLAG', 'ASCENDING_TRIANGLE', 'WEDGE_FALLING',
            'CUP_AND_HANDLE', 'INVERSE_HEAD_SHOULDERS', 'HAMMER', 'MORNING_STAR',
            'ENGULFING_BULLISH', 'THREE_WHITE_SOLDIERS', 'PIERCING_LINE', 'TWEEZER_BOTTOM',
            'INVERTED_HAMMER'
        ]
        
        neutral_list = [
            'DOJI', 'SPINNING_TOP', 'RECTANGLE', 'SYMMETRICAL_TRIANGLE',
            'DIAMOND', 'HARAMI_BULLISH', 'HARAMI_BEARISH', 'MARUBOZU'
        ]
        
        # แยก patterns ตามประเภท
        for pattern in all_patterns:
            pattern_name = pattern['pattern_name']
            
            if pattern_name in reversal_list:
                reversal_patterns.append(pattern)
            if pattern_name in continuation_list:
                continuation_patterns.append(pattern)
            if pattern_name in bearish_list:
                bearish_patterns.append(pattern)
            if pattern_name in bullish_list:
                bullish_patterns.append(pattern)
            if pattern_name in neutral_list:
                neutral_patterns.append(pattern)
        
        # 1) ข้อความแรก - สรุปภาพรวม
        total_patterns = len([p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN'])
        
        message_1 = f"""🔍 MULTIPLE PATTERNS ANALYSIS REPORT

⏰ {current_time} | 💰 XAU/USD (1H)
💾 SHARED DATA SOURCE

📊 PATTERN SUMMARY:
รวมพบ {total_patterns} patterns คุณภาพสูง

🔄 Reversal Patterns: {len(reversal_patterns)}
➡️ Continuation Patterns: {len(continuation_patterns)}
🔴 Bearish Patterns: {len(bearish_patterns)}
🟢 Bullish Patterns: {len(bullish_patterns)}
🟡 Neutral Patterns: {len(neutral_patterns)}

💰 CURRENT MARKET DATA:
Open: ${current_data['open']:,.2f} | High: ${current_data['high']:,.2f}
Low: ${current_data['low']:,.2f} | Close: ${current_data['close']:,.2f}
ราคาปัจจุบัน: ${trading_signals['current_price']:,.2f}"""
        
        send_telegram(message_1)
        time.sleep(3)
        
        # 2) ข้อความ Reversal Patterns
        if reversal_patterns:
            message_reversal = f"""🔄 REVERSAL PATTERNS DETECTED

📊 {len(reversal_patterns)} pattern(s) บ่งบอกการกลับทิศทาง:

"""
            for i, pattern in enumerate(reversal_patterns[:3], 1):
                signal = "🔴 SELL" if pattern['pattern_name'] in bearish_list else "🟢 BUY"
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                
                message_reversal += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: {signal} | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
            
            message_reversal += """⚠️ REVERSAL IMPLICATIONS:
• ตลาดอาจเปลี่ยนทิศทางหลัก
• รอการยืนยันก่อนเข้าเทรด
• ใช้ Stop Loss เข้มงวด
• เฝ้าดู Volume สนับสนุน"""
            
            send_telegram(message_reversal)
            time.sleep(3)
        
        # 3) ข้อความ Continuation Patterns
        if continuation_patterns:
            message_continuation = f"""➡️ CONTINUATION PATTERNS DETECTED

📊 {len(continuation_patterns)} pattern(s) สนับสนุนทิศทางเดิม:

"""
            for i, pattern in enumerate(continuation_patterns[:3], 1):
                signal = "🔴 SELL" if pattern['pattern_name'] in bearish_list else "🟢 BUY"
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                
                message_continuation += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: {signal} | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
            
            message_continuation += """📈 CONTINUATION IMPLICATIONS:
• เทรนด์ปัจจุบันมีแนวโน้มดำเนินต่อ
• เข้าตามทิศทางหลักของตลาด
• Target ตามขนาดของ Pattern
• Breakout พร้อม Volume"""
            
            send_telegram(message_continuation)
            time.sleep(3)
        
        # 4) ข้อความ Bullish vs Bearish Analysis
        bullish_count = len(bullish_patterns)
        bearish_count = len(bearish_patterns)
        
        if bullish_count > 0 or bearish_count > 0:
            message_direction = f"""📊 DIRECTIONAL ANALYSIS

🟢 BULLISH PATTERNS: {bullish_count}
🔴 BEARISH PATTERNS: {bearish_count}

"""
            
            if bullish_count > bearish_count:
                dominant_bias = "🟢 BULLISH BIAS"
                market_sentiment = "ตลาดมีแนวโน้มขาขึ้น"
            elif bearish_count > bullish_count:
                dominant_bias = "🔴 BEARISH BIAS"
                market_sentiment = "ตลาดมีแนวโน้มขาลง"
            else:
                dominant_bias = "🟡 NEUTRAL BIAS"
                market_sentiment = "ตลาดไม่มีทิศทางชัดเจน"
            
            message_direction += f"""🎯 MARKET BIAS: {dominant_bias}
📈 Sentiment: {market_sentiment}

"""
            
            # แสดง Top 2 Bullish patterns
            if bullish_patterns:
                message_direction += "🟢 TOP BULLISH PATTERNS:\n"
                for i, pattern in enumerate(bullish_patterns[:2], 1):
                    message_direction += f"{i}. {pattern['pattern_name'].replace('_', ' ')} ({pattern['confidence']*100:.1f}%)\n"
            
            # แสดง Top 2 Bearish patterns
            if bearish_patterns:
                message_direction += "\n🔴 TOP BEARISH PATTERNS:\n"
                for i, pattern in enumerate(bearish_patterns[:2], 1):
                    message_direction += f"{i}. {pattern['pattern_name'].replace('_', ' ')} ({pattern['confidence']*100:.1f}%)\n"
            
            message_direction += f"""
💡 TRADING RECOMMENDATION:
• Focus on {dominant_bias.split()[1]} setups
• Wait for confirmation candles
• Use appropriate Risk/Reward ratio
• Monitor key support/resistance levels"""
            
            send_telegram(message_direction)
            time.sleep(3)
        
        # 5) ข้อความ Original Bot Analysis
        original_action = trading_signals.get('action', 'WAIT')
        action_text = "🔴 SELL" if original_action == 'SELL' else "🟢 BUY" if original_action == 'BUY' else "⏸️ WAIT"
        
        message_original = f"""🤖 ORIGINAL BOT CONFIRMATION

💻 SYSTEM: RSI + EMA + Price Change
🎯 BOT Signal: {action_text}

💹 TECHNICAL INDICATORS:
RSI: {trading_signals['rsi']:.1f} ({'Oversold' if trading_signals['rsi']<30 else 'Overbought' if trading_signals['rsi']>70 else 'Neutral'})
EMA10: ${trading_signals['ema10']:,.2f} ({'Above' if trading_signals['current_price']>trading_signals['ema10'] else 'Below'})
EMA21: ${trading_signals['ema21']:,.2f} ({'Above' if trading_signals['current_price']>trading_signals['ema21'] else 'Below'})"""
        
        if original_action != 'WAIT':
            message_original += f"""

💼 SUGGESTED TRADING SETUP:
🎯 Entry: ${trading_signals['entry_price']:,.2f}
🟢 TP1: ${trading_signals['tp1']:,.2f} | TP2: ${trading_signals['tp2']:,.2f} | TP3: ${trading_signals['tp3']:,.2f}
🔴 SL: ${trading_signals['sl']:,.2f}
💯 Confidence: {trading_signals['confidence']*100:.1f}%"""
        
        send_telegram(message_original)
        time.sleep(4)
        
        # 6) ข้อความ Final Recommendations + Chart
        main_pattern = all_patterns[0]
        chart_buffer = create_candlestick_chart(shared_df, trading_signals, main_pattern)
        
        final_message = f"""📋 FINAL ANALYSIS & RECOMMENDATIONS

🎯 DOMINANT PATTERN: {main_pattern['pattern_name'].replace('_', ' ')}
💯 Highest Confidence: {main_pattern['confidence']*100:.1f}%

🔍 PATTERN CONFLICTS:
"""
        
        if bullish_count > 0 and bearish_count > 0:
            final_message += f"""⚠️ MIXED SIGNALS DETECTED!
🟢 Bullish: {bullish_count} patterns
🔴 Bearish: {bearish_count} patterns

🚨 CONFLICT RESOLUTION STRATEGY:
• รอ Breakout ที่ชัดเจน
• ใช้ขนาด Position ที่เล็กลง
• Set Tight Stop Loss
• เฝ้าดู Volume Confirmation"""
        else:
            final_message += "✅ No major conflicts detected"
        
        final_message += f"""

💡 COMPREHENSIVE TRADING PLAN:
1. Primary Signal: {action_text} (Original Bot)
2. Pattern Bias: {dominant_bias if 'dominant_bias' in locals() else 'Analyzing...'}
3. Risk Level: {'HIGH' if bullish_count > 0 and bearish_count > 0 else 'MEDIUM'}
4. Position Size: {'0.5-1%' if bullish_count > 0 and bearish_count > 0 else '1-2%'} of account

⚠️ RISK MANAGEMENT:
• Never risk more than 2% per trade
• Use proper Stop Loss always
• Wait for confirmation if patterns conflict
• Monitor market news and events

📊 Chart Analysis attached below ⬇️"""
        
        send_telegram_with_chart(final_message, chart_buffer)
        time.sleep(3)
        
        # 7) ข้อความสรุปสุดท้าย
        summary_message = f"""📚 PATTERN ANALYSIS COMPLETE ✅

🔢 SUMMARY STATISTICS:
• Total Quality Patterns: {total_patterns}
• Reversal Signals: {len(reversal_patterns)}
• Continuation Signals: {len(continuation_patterns)}
• Bullish Bias: {bullish_count} patterns
• Bearish Bias: {bearish_count} patterns
• Neutral/Wait: {len(neutral_patterns)} patterns

🎯 KEY TAKEAWAYS:
• Main Direction: {dominant_bias if 'dominant_bias' in locals() else 'Mixed'}
• Confidence Level: {main_pattern['confidence']*100:.1f}% (Highest)
• Market Phase: {'Trending' if len(continuation_patterns) > len(reversal_patterns) else 'Turning' if len(reversal_patterns) > 0 else 'Consolidating'}

💼 NEXT STEPS:
1. Monitor price action for confirmation
2. Watch for volume spikes on breakouts
3. Set alerts at key support/resistance levels
4. Prepare trading plan for next hour

⏰ Next Analysis: {current_hour + 1 if 'current_hour' in locals() else 'Next'}:00 Bangkok Time

🚀 Happy Trading! Stay disciplined and manage risk properly."""
        
        send_telegram(summary_message)
        print(f"✅ Comprehensive multiple patterns analysis sent successfully")
        return 200
        
    except Exception as e:
        print(f"Send multiple patterns error: {e}")
        return 500  

def create_pattern_theory_diagram(pattern_name):
    """Create theoretical diagram explaining pattern characteristics"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Set dark theme
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        if pattern_name == 'HEAD_SHOULDERS':
            create_head_shoulders_diagram(ax)
            title = "📊 HEAD & SHOULDERS PATTERN - ทฤษฎีและหลักการ"
            
        elif pattern_name == 'DOUBLE_TOP':
            create_double_top_diagram(ax)
            title = "📊 DOUBLE TOP PATTERN - ทฤษฎีและหลักการ"
            
        elif pattern_name == 'DOUBLE_BOTTOM':
            create_double_bottom_diagram(ax)
            title = "📊 DOUBLE BOTTOM PATTERN - ทฤษฎีและหลักการ"
            
        elif pattern_name == 'ASCENDING_TRIANGLE':
            create_ascending_triangle_diagram(ax)
            title = "📊 ASCENDING TRIANGLE - ทฤษฎีและหลักการ"
            
        elif pattern_name == 'BULL_FLAG':
            create_bull_flag_diagram(ax)
            title = "📊 BULL FLAG PATTERN - ทฤษฎีและหลักการ"
            
        else:
            create_generic_pattern_diagram(ax)
            title = "📊 CHART PATTERN ANALYSIS - ทฤษฎีทั่วไป"

        if pattern_name in extended_patterns: 
            create_extended_theory_diagrams(ax, pattern_name)
            title = f"📊 {pattern_name} PATTERN - ทฤษฎีและหลักการ"
        elif pattern_name == 'HEAD_SHOULDERS':
            create_head_shoulders_diagram(ax)
            title = "📊 HEAD & SHOULDERS PATTERN - ทฤษฎีและหลักการ"
        # ... existing patterns ...
        else:
            create_generic_pattern_diagram(ax)
            title = "📊 CHART PATTERN ANALYSIS - ทฤษฎีทั่วไป"
        
        # Style the chart
        ax.set_title(title, color='#ffffff', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time (เวลา)', color='#ffffff', fontsize=12)
        ax.set_ylabel('Price (ราคา)', color='#ffffff', fontsize=12)
        ax.tick_params(colors='#ffffff')
        ax.grid(True, alpha=0.3, color='#444444')
        
        # Remove axes numbers for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        ax.legend(loc='upper left', facecolor='#2a2a2a', edgecolor='#444444', 
                 labelcolor='#ffffff', fontsize=10)
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#1a1a1a', 
                   edgecolor='none', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Pattern theory diagram error: {e}")
        return None

def create_head_shoulders_diagram(ax):
    """Create Head & Shoulders theoretical diagram"""
    # Price points for the pattern
    x = np.linspace(0, 10, 100)
    
    # Create idealized head and shoulders pattern
    left_shoulder = 2 + 0.5 * np.sin((x - 2) * 2) * np.exp(-((x - 2) / 1.5)**2)
    head = 3 + 1.2 * np.sin((x - 5) * 2) * np.exp(-((x - 5) / 1.0)**2)
    right_shoulder = 2 + 0.5 * np.sin((x - 8) * 2) * np.exp(-((x - 8) / 1.5)**2)
    
    price_line = 2 + left_shoulder + head + right_shoulder
    
    # Plot the main pattern
    ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
    
    # Mark key points
    shoulder_points_x = [2, 5, 8]
    shoulder_points_y = [price_line[20], price_line[50], price_line[80]]
    
    ax.scatter(shoulder_points_x, shoulder_points_y, color='#ff4444', s=100, 
              marker='^', label='Key Points', zorder=5)
    
    # Draw neckline
    neckline_y = (shoulder_points_y[0] + shoulder_points_y[2]) / 2
    ax.axhline(y=neckline_y, color='#ff00ff', linestyle='--', 
              linewidth=2, alpha=0.8, label='Neckline')
    
    # Add annotations
    ax.annotate('Left Shoulder\n(ไหล่ซ้าย)', xy=(2, shoulder_points_y[0]), 
               xytext=(1, shoulder_points_y[0] + 0.8), 
               arrowprops=dict(arrowstyle='->', color='#ffaa00'),
               color='#ffaa00', fontsize=10, ha='center')
    
    ax.annotate('Head\n(หัว)', xy=(5, shoulder_points_y[1]), 
               xytext=(5, shoulder_points_y[1] + 1.0), 
               arrowprops=dict(arrowstyle='->', color='#ffaa00'),
               color='#ffaa00', fontsize=10, ha='center')
    
    ax.annotate('Right Shoulder\n(ไหล่ขวา)', xy=(8, shoulder_points_y[2]), 
               xytext=(9, shoulder_points_y[2] + 0.8), 
               arrowprops=dict(arrowstyle='->', color='#ffaa00'),
               color='#ffaa00', fontsize=10, ha='center')
    
    # Add breakout arrow
    ax.annotate('Breakout Target\n(เป้าหมายทะลุ)', xy=(6, neckline_y - 0.5), 
               xytext=(7.5, neckline_y - 1.5), 
               arrowprops=dict(arrowstyle='->', color='#ff4444'),
               color='#ff4444', fontsize=10, ha='center')

def create_double_top_diagram(ax):
    """Create Double Top theoretical diagram"""
    x = np.linspace(0, 10, 100)
    
    # Create double top pattern
    peak1 = 1.5 * np.exp(-((x - 3) / 0.8)**2)
    valley = -0.5 * np.exp(-((x - 5) / 0.6)**2)
    peak2 = 1.5 * np.exp(-((x - 7) / 0.8)**2)
    
    price_line = 3 + peak1 + valley + peak2
    
    ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
    
    # Mark peaks and valley
    peak_points_x = [3, 7]
    peak_points_y = [price_line[30], price_line[70]]
    valley_point_x = 5
    valley_point_y = price_line[50]
    
    ax.scatter(peak_points_x, peak_points_y, color='#ff4444', s=100, 
              marker='v', label='Double Top', zorder=5)
    ax.scatter([valley_point_x], [valley_point_y], color='#00ff88', s=100, 
              marker='^', label='Support', zorder=5)
    
    # Draw support line
    ax.axhline(y=valley_point_y, color='#00ff88', linestyle='--', 
              linewidth=2, alpha=0.8, label='Support Level')
    
    # Annotations
    ax.annotate('Top 1', xy=(3, peak_points_y[0]), 
               xytext=(2.5, peak_points_y[0] + 0.5), 
               arrowprops=dict(arrowstyle='->', color='#ffaa00'),
               color='#ffaa00', fontsize=10, ha='center')
    
    ax.annotate('Top 2', xy=(7, peak_points_y[1]), 
               xytext=(7.5, peak_points_y[1] + 0.5), 
               arrowprops=dict(arrowstyle='->', color='#ffaa00'),
               color='#ffaa00', fontsize=10, ha='center')

def create_double_bottom_diagram(ax):
    """Create Double Bottom theoretical diagram"""
    x = np.linspace(0, 10, 100)
    
    # Create double bottom pattern (inverted double top)
    bottom1 = -1.5 * np.exp(-((x - 3) / 0.8)**2)
    peak = 0.5 * np.exp(-((x - 5) / 0.6)**2)
    bottom2 = -1.5 * np.exp(-((x - 7) / 0.8)**2)
    
    price_line = 3 + bottom1 + peak + bottom2
    
    ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
    
    # Mark bottoms and peak
    bottom_points_x = [3, 7]
    bottom_points_y = [price_line[30], price_line[70]]
    peak_point_x = 5
    peak_point_y = price_line[50]
    
    ax.scatter(bottom_points_x, bottom_points_y, color='#00ff88', s=100, 
              marker='^', label='Double Bottom', zorder=5)
    ax.scatter([peak_point_x], [peak_point_y], color='#ff4444', s=100, 
              marker='v', label='Resistance', zorder=5)
    
    # Draw resistance line
    ax.axhline(y=peak_point_y, color='#ff4444', linestyle='--', 
              linewidth=2, alpha=0.8, label='Resistance Level')

def create_ascending_triangle_diagram(ax):
    """Create Ascending Triangle theoretical diagram"""
    x = np.linspace(0, 10, 100)
    
    # Create ascending triangle
    resistance_level = 4
    ascending_support = 2 + 0.2 * x
    
    # Price action within triangle
    price_oscillation = 0.3 * np.sin(x * 2) * (10 - x) / 10
    price_line = ascending_support + price_oscillation
    
    # Ensure price stays below resistance
    price_line = np.minimum(price_line, resistance_level - 0.1)
    
    ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
    
    # Draw triangle lines
    ax.axhline(y=resistance_level, color='#ff4444', linestyle='-', 
              linewidth=2, alpha=0.8, label='Horizontal Resistance')
    ax.plot(x, ascending_support, color='#00ff88', linestyle='-', 
           linewidth=2, alpha=0.8, label='Ascending Support')
    
    # Add breakout arrow
    ax.annotate('Breakout Point\n(จุดทะลุ)', xy=(8.5, resistance_level + 0.2), 
               xytext=(9.5, resistance_level + 0.8), 
               arrowprops=dict(arrowstyle='->', color='#ffaa00'),
               color='#ffaa00', fontsize=10, ha='center')

def create_bull_flag_diagram(ax):
    """Create Bull Flag theoretical diagram"""
    x = np.linspace(0, 10, 100)
    
    # Flagpole (strong uptrend)
    flagpole_x = x[x <= 4]
    flagpole_y = 1 + 2 * (flagpole_x / 4)
    
    # Flag (consolidation)
    flag_x = x[(x > 4) & (x <= 7)]
    flag_y = 3 - 0.1 * (flag_x - 4) + 0.1 * np.sin((flag_x - 4) * 3)
    
    # Breakout
    breakout_x = x[x > 7]
    breakout_y = 2.7 + 1.5 * ((breakout_x - 7) / 3)
    
    # Plot segments
    ax.plot(flagpole_x, flagpole_y, color='#00ff88', linewidth=4, 
           label='Flagpole (Strong Trend)')
    ax.plot(flag_x, flag_y, color='#ffaa00', linewidth=3, 
           label='Flag (Consolidation)')
    ax.plot(breakout_x, breakout_y, color='#00ff88', linewidth=4, 
           label='Breakout Continuation')
    
    # Mark flag boundaries
    flag_top = np.max(flag_y) + 0.1
    flag_bottom = np.min(flag_y) - 0.1
    
    ax.axhline(y=flag_top, xmin=0.4, xmax=0.7, color='#ff4444', 
              linestyle='--', alpha=0.7, label='Flag Boundaries')
    ax.axhline(y=flag_bottom, xmin=0.4, xmax=0.7, color='#ff4444', 
              linestyle='--', alpha=0.7)

def create_generic_pattern_diagram(ax):
    """Create generic pattern analysis diagram"""
    x = np.linspace(0, 10, 100)
    
    # Generic price movement with support/resistance
    price_line = 3 + 0.5 * np.sin(x) + 0.2 * np.sin(2 * x) + 0.1 * x
    
    ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
    
    # Add generic support/resistance
    ax.axhline(y=3.5, color='#00ff88', linestyle='--', alpha=0.7, label='Support')
    ax.axhline(y=4.5, color='#ff4444', linestyle='--', alpha=0.7, label='Resistance')
    
    ax.text(5, 2, 'รอการก่อตัวของแพทเทิร์น\nWaiting for Pattern Formation', 
           ha='center', va='center', color='#ffffff', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

def create_extended_theory_diagrams(ax, pattern_name):
    """Create theory diagrams for new patterns"""
    x = np.linspace(0, 10, 100)
    
    if pattern_name == 'DESCENDING_TRIANGLE':
        # Descending triangle with horizontal support and descending resistance
        support_level = 2.5
        resistance_line = 4.5 - 0.2 * x
        
        # Price action within triangle
        price_oscillation = 0.2 * np.sin(x * 3) * (10 - x) / 10
        price_line = support_level + 0.3 + price_oscillation
        price_line = np.minimum(price_line, resistance_line - 0.05)
        price_line = np.maximum(price_line, support_level + 0.05)
        
        ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
        ax.axhline(y=support_level, color='#00ff88', linestyle='-', 
                  linewidth=2, alpha=0.8, label='Horizontal Support')
        ax.plot(x, resistance_line, color='#ff4444', linestyle='-', 
               linewidth=2, alpha=0.8, label='Descending Resistance')
        
        # Breakout arrow
        ax.annotate('Bearish Breakout\n(ทะลุลง)', xy=(8, support_level - 0.3), 
                   xytext=(9, support_level - 0.8), 
                   arrowprops=dict(arrowstyle='->', color='#ff4444'),
                   color='#ff4444', fontsize=10, ha='center')

    elif pattern_name == 'SYMMETRICAL_TRIANGLE':
        # Symmetrical triangle - converging lines
        resistance_line = 4 - 0.15 * x
        support_line = 2.5 + 0.1 * x
        
        # Price oscillating within converging lines
        price_oscillation = 0.1 * np.sin(x * 4) * (10 - x) / 10
        price_line = (resistance_line + support_line) / 2 + price_oscillation
        
        ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
        ax.plot(x, resistance_line, color='#ff4444', linestyle='-', 
               linewidth=2, alpha=0.8, label='Descending Resistance')
        ax.plot(x, support_line, color='#00ff88', linestyle='-', 
               linewidth=2, alpha=0.8, label='Ascending Support')
        
        # Convergence point
        ax.scatter([10], [resistance_line[-1]], color='#ffaa00', s=100, 
                  marker='*', label='Apex', zorder=5)
        
        # Breakout possibilities
        ax.annotate('Breakout Direction\n(ตามเทรนด์)', xy=(8.5, 3.2), 
                   xytext=(9.5, 3.8), 
                   arrowprops=dict(arrowstyle='->', color='#ffaa00'),
                   color='#ffaa00', fontsize=10, ha='center')

    elif pattern_name == 'BEAR_FLAG':
        # Strong downtrend (flagpole)
        flagpole_x = x[x <= 3]
        flagpole_y = 4 - 1.5 * (flagpole_x / 3)
        
        # Flag (consolidation)
        flag_x = x[(x > 3) & (x <= 7)]
        flag_y = 2.5 + 0.1 * (flag_x - 3) + 0.05 * np.sin((flag_x - 3) * 4)
        
        # Breakdown
        breakdown_x = x[x > 7]
        breakdown_y = 2.7 - 1.2 * ((breakdown_x - 7) / 3)
        
        ax.plot(flagpole_x, flagpole_y, color='#ff4444', linewidth=4, 
               label='Flagpole (Strong Downtrend)')
        ax.plot(flag_x, flag_y, color='#ffaa00', linewidth=3, 
               label='Flag (Consolidation)')
        ax.plot(breakdown_x, breakdown_y, color='#ff4444', linewidth=4, 
               label='Breakdown Continuation')
        
        # Mark flag boundaries
        flag_top = np.max(flag_y) + 0.1
        flag_bottom = np.min(flag_y) - 0.1
        
        ax.axhline(y=flag_top, xmin=0.3, xmax=0.7, color='#888888', 
                  linestyle='--', alpha=0.7)
        ax.axhline(y=flag_bottom, xmin=0.3, xmax=0.7, color='#888888', 
                  linestyle='--', alpha=0.7)

    elif pattern_name == 'WEDGE_RISING':
        # Rising wedge - both lines ascending but converging
        support_line = 1.5 + 0.15 * x
        resistance_line = 2 + 0.1 * x
        
        # Price oscillating within wedge
        price_oscillation = 0.1 * np.sin(x * 4) * (10 - x) / 10
        price_line = (support_line + resistance_line) / 2 + price_oscillation
        
        ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
        ax.plot(x, support_line, color='#00ff88', linestyle='-', 
               linewidth=2, alpha=0.8, label='Rising Support')
        ax.plot(x, resistance_line, color='#ff4444', linestyle='-', 
               linewidth=2, alpha=0.8, label='Rising Resistance')
        
        # Bearish breakout
        ax.annotate('Bearish Breakdown\n(ทะลุลง)', xy=(8, support_line[80] - 0.2), 
                   xytext=(9, support_line[80] - 0.6), 
                   arrowprops=dict(arrowstyle='->', color='#ff4444'),
                   color='#ff4444', fontsize=10, ha='center')

    elif pattern_name == 'WEDGE_FALLING':
        # Falling wedge - both lines descending but converging
        resistance_line = 4 - 0.1 * x
        support_line = 3.5 - 0.15 * x
        
        # Price oscillating within wedge
        price_oscillation = 0.1 * np.sin(x * 4) * (10 - x) / 10
        price_line = (resistance_line + support_line) / 2 + price_oscillation
        
        ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
        ax.plot(x, resistance_line, color='#ff4444', linestyle='-', 
               linewidth=2, alpha=0.8, label='Falling Resistance')
        ax.plot(x, support_line, color='#00ff88', linestyle='-', 
               linewidth=2, alpha=0.8, label='Falling Support')
        
        # Bullish breakout
        ax.annotate('Bullish Breakout\n(ทะลุขึ้น)', xy=(8, resistance_line[80] + 0.2), 
                   xytext=(9, resistance_line[80] + 0.6), 
                   arrowprops=dict(arrowstyle='->', color='#00ff88'),
                   color='#00ff88', fontsize=10, ha='center')

    elif pattern_name == 'CUP_AND_HANDLE':
        # Cup formation
        cup_x = x[x <= 7]
        cup_y = 3 - 0.8 * ((cup_x - 3.5)**2 / 12)
        
        # Handle
        handle_x = x[x > 7]
        handle_y = 2.8 - 0.1 * (handle_x - 7) + 0.05 * np.sin((handle_x - 7) * 6)
        
        ax.plot(cup_x, cup_y, color='#00ff88', linewidth=3, label='Cup Formation')
        ax.plot(handle_x, handle_y, color='#ffaa00', linewidth=3, label='Handle')
        
        # Rim line
        rim_level = 3
        ax.axhline(y=rim_level, xmin=0, xmax=0.7, color='#ff4444', 
                  linestyle='--', alpha=0.7, label='Cup Rim')
        
        # Breakout point
        ax.annotate('Breakout Point\n(จุดทะลุ)', xy=(8.5, rim_level + 0.1), 
                   xytext=(9.5, rim_level + 0.5), 
                   arrowprops=dict(arrowstyle='->', color='#00ff88'),
                   color='#00ff88', fontsize=10, ha='center')

    elif pattern_name == 'RECTANGLE':
        # Trading range with horizontal support and resistance
        resistance_level = 4
        support_level = 2.5
        
        # Price oscillating between levels
        price_oscillation = 0.3 * np.sin(x * 2) + 0.2 * np.sin(x * 5)
        price_line = 3.25 + price_oscillation
        price_line = np.minimum(price_line, resistance_level - 0.1)
        price_line = np.maximum(price_line, support_level + 0.1)
        
        ax.plot(x, price_line, color='#00ff88', linewidth=3, label='Price Action')
        ax.axhline(y=resistance_level, color='#ff4444', linestyle='-', 
                  linewidth=2, alpha=0.8, label='Resistance')
        ax.axhline(y=support_level, color='#00ff88', linestyle='-', 
                  linewidth=2, alpha=0.8, label='Support')
        
        # Trading range
        ax.fill_between(x, support_level, resistance_level, alpha=0.1, color='#ffaa00')
        ax.text(5, 3.25, 'Trading Range\nระหว่างแนวรับ-แนวต้าน', 
               ha='center', va='center', color='#ffffff', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'DOJI':
        # Single doji candlestick illustration
        ax.axhline(y=3, xmin=0.4, xmax=0.6, color='#ffffff', linewidth=6, 
                  label='Open/Close Same Level')
        ax.plot([5, 5], [2, 4], color='#ffffff', linewidth=3, 
               label='Upper/Lower Shadow')
        ax.scatter([5], [3], color='#ffaa00', s=200, marker='o', 
                  label='Doji Point', zorder=5)
        
        # Add explanation
        ax.text(5, 1.5, 'Doji Candlestick\nเปิด = ปิด (เงาบน-ล่างยาว)\nความไม่แน่ใจของตลาด', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'HAMMER':
        # Hammer candlestick
        ax.add_patch(patches.Rectangle((4.8, 3), 0.4, 0.2, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.plot([5, 5], [2, 3], color='#00ff88', linewidth=3, 
               label='Long Lower Shadow')
        ax.plot([5, 5], [3.2, 3.4], color='#00ff88', linewidth=2, 
               label='Short Upper Shadow')
        
        ax.text(5, 1.5, 'Hammer Pattern\nตัวเล็ก + เงาล่างยาว\nสัญญาณกลับตัวขาขึ้น', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'SHOOTING_STAR':
        # Shooting star candlestick
        ax.add_patch(patches.Rectangle((4.8, 2.8), 0.4, 0.2, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        ax.plot([5, 5], [3, 4.5], color='#ff4444', linewidth=3, 
               label='Long Upper Shadow')
        ax.plot([5, 5], [2.6, 2.8], color='#ff4444', linewidth=2, 
               label='Short Lower Shadow')
        
        ax.text(5, 1.8, 'Shooting Star\nตัวเล็ก + เงาบนยาว\nสัญญาณกลับตัวขาลง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'ENGULFING_BULLISH':
        # Two candlesticks - bearish then bullish engulfing
        ax.add_patch(patches.Rectangle((4, 2.8), 0.3, -0.6, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.7, 1.8), 0.3, 1.4, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        
        # Wicks
        ax.plot([4.15, 4.15], [2.9, 3.1], color='#ff4444', linewidth=2)
        ax.plot([4.15, 4.15], [2.1, 2.2], color='#ff4444', linewidth=2)
        ax.plot([4.85, 4.85], [3.3, 3.4], color='#00ff88', linewidth=2)
        ax.plot([4.85, 4.85], [1.7, 1.8], color='#00ff88', linewidth=2)
        
        ax.text(4.4, 1.2, 'Bullish Engulfing\nแท่งขาวใหญ่กลืนแท่งดำ\nสัญญาณกลับตัวแรง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'MORNING_STAR':
        # Three candlesticks pattern
        ax.add_patch(patches.Rectangle((3, 3), 0.25, -0.8, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.2, 2.3), 0.25, -0.1, 
                                     facecolor='#ffaa00', edgecolor='#ffaa00', alpha=0.8))
        ax.add_patch(patches.Rectangle((5.4, 2.4), 0.25, 0.9, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        
        # Add wicks
        ax.plot([3.125, 3.125], [3.1, 3.3], color='#ff4444', linewidth=2)
        ax.plot([3.125, 3.125], [2.1, 2.2], color='#ff4444', linewidth=2)
        ax.plot([4.325, 4.325], [2.4, 2.5], color='#ffaa00', linewidth=2)
        ax.plot([4.325, 4.325], [2.1, 2.2], color='#ffaa00', linewidth=2)
        ax.plot([5.525, 5.525], [3.4, 3.5], color='#00ff88', linewidth=2)
        ax.plot([5.525, 5.525], [2.3, 2.4], color='#00ff88', linewidth=2)
        
        ax.text(4.2, 1.5, 'Morning Star\nดำ-เล็ก-ขาว\nดาวรุ่งนำทางขาขึ้น', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'THREE_WHITE_SOLDIERS':
        # Three consecutive white candles
        for i, x_pos in enumerate([3.5, 4.5, 5.5]):
            height = 0.8 + i * 0.1  # Increasing heights
            bottom = 2.2 + i * 0.3   # Rising bottoms
            ax.add_patch(patches.Rectangle((x_pos-0.15, bottom), 0.3, height, 
                                         facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
            # Small wicks
            ax.plot([x_pos, x_pos], [bottom + height, bottom + height + 0.1], 
                   color='#00ff88', linewidth=2)
            ax.plot([x_pos, x_pos], [bottom - 0.1, bottom], color='#00ff88', linewidth=2)
        
        ax.text(4.5, 1.5, 'Three White Soldiers\nสามแท่งขาวขึ้นเรื่อยๆ\nโมเมนตัมขาขึ้นแรง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'MARUBOZU':
        # Bullish Marubozu
        ax.add_patch(patches.Rectangle((4.5, 2), 1, 1.5, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.9))
        ax.text(5, 1.2, 'MARUBOZU\nไม่มีเงาบน-ล่าง\nโมเมนตัมแรงและชัดเจน', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'PENNANT':
        # Flagpole
        flagpole_x = x[x <= 3]
        flagpole_y = 2 + 1.5 * (flagpole_x / 3)
        # Pennant (small triangle)
        pennant_x = x[(x > 3) & (x <= 7)]
        resistance_line = 3.5 - 0.1 * (pennant_x - 3)
        support_line = 3.3 + 0.05 * (pennant_x - 3)
        
        ax.plot(flagpole_x, flagpole_y, color='#00ff88', linewidth=4, label='Strong Move')
        ax.plot(pennant_x, resistance_line, color='#ff4444', linestyle='-', linewidth=2)
        ax.plot(pennant_x, support_line, color='#00ff88', linestyle='-', linewidth=2)
        ax.text(5, 1.8, 'PENNANT\nธงเล็ก หลังการเคลื่อนไหวแรง\nBreakout ตามทิศทางเดิม', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'HANGING_MAN':
        # Hanging man candlestick
        ax.add_patch(patches.Rectangle((4.8, 3.5), 0.4, 0.2, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        ax.plot([5, 5], [2, 3.5], color='#ff4444', linewidth=3, label='Long Lower Shadow')
        ax.plot([5, 5], [3.7, 3.8], color='#ff4444', linewidth=2, label='Short Upper Shadow')
        ax.text(5, 1.5, 'HANGING MAN\nคล้าย Hammer แต่บริบท Bearish\nเตือนการกลับตัว', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'INVERTED_HAMMER':
        ax.add_patch(patches.Rectangle((4.8, 2.8), 0.4, 0.2, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.plot([5, 5], [3, 4.5], color='#00ff88', linewidth=3)
        ax.plot([5, 5], [2.6, 2.8], color='#00ff88', linewidth=2)
        ax.text(5, 1.8, 'INVERTED HAMMER\nเงาบนยาว หลัง Downtrend\nสัญญาณกลับตัวขาขึ้น', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'SPINNING_TOP':
        ax.add_patch(patches.Rectangle((4.85, 2.9), 0.3, 0.2, 
                                     facecolor='#ffaa00', edgecolor='#ffaa00', alpha=0.8))
        ax.plot([5, 5], [3.1, 3.8], color='#ffaa00', linewidth=2)
        ax.plot([5, 5], [2.2, 2.9], color='#ffaa00', linewidth=2)
        ax.text(5, 1.5, 'SPINNING TOP\nตัวเล็ก เงาบน-ล่างยาว\nความไม่แน่ใจ', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'PIERCING_LINE':
        ax.add_patch(patches.Rectangle((3.8, 3), 0.3, -0.6, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.7, 2.2), 0.3, 1.2, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.text(4.25, 1.5, 'PIERCING LINE\nแท่งขาวทะลุผ่าน Midpoint\nสัญญาณกลับตัวขาขึ้น', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'HARAMI_BULLISH':
        # Mother candle (large black)
        ax.add_patch(patches.Rectangle((3.5, 3.5), 0.4, -1.0, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        # Baby candle (small white inside)
        ax.add_patch(patches.Rectangle((4.3, 3.1), 0.3, 0.4, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.text(3.9, 1.8, 'BULLISH HARAMI\nแท่งลูกขาวในแท่งแม่ดำ\nความเบื่อหน่ายในการขาย', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'ENGULFING_BEARISH':
        # Two candlesticks - bullish then bearish engulfing
        ax.add_patch(patches.Rectangle((3.8, 2.8), 0.3, 0.6, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.7, 1.8), 0.3, -1.4, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        
        # Wicks
        ax.plot([3.95, 3.95], [3.5, 3.6], color='#00ff88', linewidth=2)
        ax.plot([3.95, 3.95], [2.7, 2.8], color='#00ff88', linewidth=2)
        ax.plot([4.85, 4.85], [1.9, 2.0], color='#ff4444', linewidth=2)
        ax.plot([4.85, 4.85], [1.7, 1.8], color='#ff4444', linewidth=2)
        
        ax.text(4.4, 4.2, 'Bearish Engulfing\nแท่งดำใหญ่กลืนแท่งขาว\nสัญญาณกลับตัวขาลง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'EVENING_STAR':
        # Three candlesticks pattern
        ax.add_patch(patches.Rectangle((3, 2.4), 0.25, 0.9, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.2, 3.2), 0.25, 0.1, 
                                     facecolor='#ffaa00', edgecolor='#ffaa00', alpha=0.8))
        ax.add_patch(patches.Rectangle((5.4, 3), 0.25, -0.8, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        
        # Add wicks
        ax.plot([3.125, 3.125], [3.4, 3.5], color='#00ff88', linewidth=2)
        ax.plot([3.125, 3.125], [2.3, 2.4], color='#00ff88', linewidth=2)
        ax.plot([4.325, 4.325], [3.4, 3.5], color='#ffaa00', linewidth=2)
        ax.plot([4.325, 4.325], [3.1, 3.2], color='#ffaa00', linewidth=2)
        ax.plot([5.525, 5.525], [3.1, 3.2], color='#ff4444', linewidth=2)
        ax.plot([5.525, 5.525], [2.1, 2.2], color='#ff4444', linewidth=2)
        
        ax.text(4.2, 1.5, 'Evening Star\nขาว-เล็ก-ดำ\nดาวค่ำประกาศขาลง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'THREE_BLACK_CROWS':
        # Three consecutive black candles
        for i, x_pos in enumerate([3.5, 4.5, 5.5]):
            height = 0.8 + i * 0.1  # Increasing heights (more negative)
            bottom = 3.2 - i * 0.3   # Falling bottoms
            ax.add_patch(patches.Rectangle((x_pos-0.15, bottom), 0.3, -height, 
                                         facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
            # Small wicks
            ax.plot([x_pos, x_pos], [bottom, bottom + 0.1], 
                   color='#ff4444', linewidth=2)
            ax.plot([x_pos, x_pos], [bottom - height - 0.1, bottom - height], 
                   color='#ff4444', linewidth=2)
        
        ax.text(4.5, 1.2, 'Three Black Crows\nสามแท่งดำลงเรื่อยๆ\nโมเมนตัมขาลงแรง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'INVERSE_HEAD_SHOULDERS':
        # Inverse head and shoulders pattern
        x_pattern = np.linspace(0, 10, 100)
        
        # Create inverted head and shoulders
        left_shoulder = -0.5 * np.sin((x_pattern - 2) * 2) * np.exp(-((x_pattern - 2) / 1.5)**2)
        head = -1.2 * np.sin((x_pattern - 5) * 2) * np.exp(-((x_pattern - 5) / 1.0)**2)
        right_shoulder = -0.5 * np.sin((x_pattern - 8) * 2) * np.exp(-((x_pattern - 8) / 1.5)**2)
        
        price_line = 3 + left_shoulder + head + right_shoulder
        
        ax.plot(x_pattern, price_line, color='#00ff88', linewidth=3, label='Price Action')
        
        # Mark key points
        shoulder_points_x = [2, 5, 8]
        shoulder_points_y = [price_line[20], price_line[50], price_line[80]]
        
        ax.scatter(shoulder_points_x, shoulder_points_y, color='#00ff88', s=100, 
                  marker='v', label='Key Points', zorder=5)
        
        # Draw neckline
        neckline_y = (shoulder_points_y[0] + shoulder_points_y[2]) / 2
        ax.axhline(y=neckline_y, color='#ff00ff', linestyle='--', 
                  linewidth=2, alpha=0.8, label='Neckline')
        
        ax.text(5, 4.5, 'Inverse Head & Shoulders\nไหล่-หัว-ไหล่ กลับหัว\nสัญญาณกลับตัวขาขึ้น', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'DIAMOND':
        # Diamond pattern - expanding then contracting volatility
        x_diamond = np.linspace(0, 10, 50)
        
        # Create diamond shape
        expanding_phase = x_diamond[x_diamond <= 5]
        contracting_phase = x_diamond[x_diamond > 5]
        
        # Expanding volatility
        expand_upper = 3 + 0.3 * expanding_phase
        expand_lower = 3 - 0.3 * expanding_phase
        
        # Contracting volatility  
        contract_upper = 4.5 - 0.3 * (contracting_phase - 5)
        contract_lower = 1.5 + 0.3 * (contracting_phase - 5)
        
        # Plot diamond outline
        ax.plot(expanding_phase, expand_upper, color='#ff00ff', linewidth=2, alpha=0.8)
        ax.plot(expanding_phase, expand_lower, color='#ff00ff', linewidth=2, alpha=0.8)
        ax.plot(contracting_phase, contract_upper, color='#ff00ff', linewidth=2, alpha=0.8)
        ax.plot(contracting_phase, contract_lower, color='#ff00ff', linewidth=2, alpha=0.8)
        
        # Price action inside diamond
        diamond_price = 3 + 0.2 * np.sin(x_diamond * 3) * (5 - abs(x_diamond - 5)) / 5
        ax.plot(x_diamond, diamond_price, color='#00ff88', linewidth=2, label='Price')
        
        ax.text(5, 0.8, 'Diamond Pattern\nความผันผวนขยาย-หด\nเกิดที่จุดกลับตัวสำคัญ', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'DARK_CLOUD_COVER':
        # White candle followed by black candle covering it
        ax.add_patch(patches.Rectangle((3.8, 2.5), 0.3, 0.8, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.7, 3.5), 0.3, -1.2, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        
        # Wicks
        ax.plot([3.95, 3.95], [3.4, 3.5], color='#00ff88', linewidth=2)
        ax.plot([3.95, 3.95], [2.4, 2.5], color='#00ff88', linewidth=2)
        ax.plot([4.85, 4.85], [3.6, 3.7], color='#ff4444', linewidth=2)
        ax.plot([4.85, 4.85], [2.2, 2.3], color='#ff4444', linewidth=2)
        
        # Show midpoint line
        midpoint = (2.5 + 3.3) / 2
        ax.axhline(y=midpoint, xmin=0.35, xmax=0.52, color='#ffaa00', 
                  linestyle=':', linewidth=2, alpha=0.8, label='Midpoint')
        
        ax.text(4.25, 1.5, 'Dark Cloud Cover\nแท่งดำครอบแท่งขาว\nปิดใต้ Midpoint', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'HARAMI_BEARISH':
        # Mother candle (large white)
        ax.add_patch(patches.Rectangle((3.5, 2.5), 0.4, 1.0, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        # Baby candle (small black inside)
        ax.add_patch(patches.Rectangle((4.3, 3.2), 0.3, -0.4, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        
        # Show containment
        ax.plot([3.5, 3.9, 3.9, 3.5, 3.5], [2.5, 2.5, 3.5, 3.5, 2.5], 
               color='#ffaa00', linestyle=':', linewidth=1, alpha=0.6)
        
        ax.text(3.9, 1.8, 'Bearish Harami\nแท่งลูกดำในแท่งแม่ขาว\nความเบื่อหน่ายในการซื้อ', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'TWEEZER_TOP':
        # Two candles with similar highs
        ax.add_patch(patches.Rectangle((3.8, 2.8), 0.3, 0.9, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.7, 3.2), 0.3, -0.5, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        
        # Same high level wicks
        same_high = 4.0
        ax.plot([3.95, 3.95], [3.7, same_high], color='#00ff88', linewidth=3)
        ax.plot([4.85, 4.85], [3.2, same_high], color='#ff4444', linewidth=3)
        ax.plot([3.95, 3.95], [2.7, 2.8], color='#00ff88', linewidth=2)
        ax.plot([4.85, 4.85], [2.6, 2.7], color='#ff4444', linewidth=2)
        
        # Resistance line at same high
        ax.axhline(y=same_high, xmin=0.35, xmax=0.52, color='#ff4444', 
                  linestyle='-', linewidth=3, alpha=0.8, label='Resistance Level')
        
        ax.text(4.25, 1.5, 'Tweezer Top\nสอง High เท่ากัน\nแนวต้านแข็งแกร่ง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    elif pattern_name == 'TWEEZER_BOTTOM':
        # Two candles with similar lows
        ax.add_patch(patches.Rectangle((3.8, 3.2), 0.3, -0.9, 
                                     facecolor='#ff4444', edgecolor='#ff4444', alpha=0.8))
        ax.add_patch(patches.Rectangle((4.7, 2.8), 0.3, 0.5, 
                                     facecolor='#00ff88', edgecolor='#00ff88', alpha=0.8))
        
        # Same low level wicks
        same_low = 2.0
        ax.plot([3.95, 3.95], [2.3, same_low], color='#ff4444', linewidth=3)
        ax.plot([4.85, 4.85], [2.8, same_low], color='#00ff88', linewidth=3)
        ax.plot([3.95, 3.95], [3.3, 3.4], color='#ff4444', linewidth=2)
        ax.plot([4.85, 4.85], [3.4, 3.5], color='#00ff88', linewidth=2)
        
        # Support line at same low
        ax.axhline(y=same_low, xmin=0.35, xmax=0.52, color='#00ff88', 
                  linestyle='-', linewidth=3, alpha=0.8, label='Support Level')
        
        ax.text(4.25, 4.2, 'Tweezer Bottom\nสอง Low เท่ากัน\nแนวรับแข็งแกร่ง', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

    else:
        # Generic pattern
        ax.text(5, 3, f'{pattern_name}\nแพทเทิร์นนี้กำลังพัฒนา\nรายละเอียดเพิ่มเติมเร็วๆ นี้', 
               ha='center', va='center', color='#ffffff', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))

def send_pattern_theory_explanation(pattern_name, pattern_description):
    """Send pattern theory diagram and detailed explanation with BOT prediction"""
    try:
        # Create theory diagram
        theory_diagram = create_pattern_theory_diagram(pattern_name)
        bot_signal = get_pattern_signal(pattern_name)
        
        if theory_diagram:
            # Prepare enhanced theory explanation message with BOT prediction
            theory_message = f"""📚 PATTERN THEORY MASTER CLASS

🎯 {pattern_name} - ทฤษฎีและหลักการ
🎯BOTทำนาย: {bot_signal}

{pattern_description}

💡 หลักการสำคัญ:
• ศึกษาโครงสร้างแพทเทิร์นก่อนเทรดจริง
• รอการยืนยันก่อนเข้าออเดอร์
• ใช้ Risk Management ที่เข้มงวด
• เฝ้าดูปริมาณการซื้อขาย (Volume)

🤖 BOT Analysis:
• Pattern Signal: {bot_signal}
• Confidence Level: ขึ้นกับ Market Context
• Risk Warning: ใช้เงินไม่เกิน 1-2% ต่อออเดอร์

📖 แหล่งอ้างอิง: Technical Analysis Theory + AI Pattern Recognition"""
            
            # Send diagram with enhanced theory explanation
            send_status = send_telegram_with_chart(theory_message, theory_diagram)
            print(f"Pattern theory diagram with BOT prediction sent: Status {send_status}")
            return send_status
        else:
            # Fallback: send text-only theory with BOT prediction
            theory_text = f"""📚 PATTERN THEORY: {pattern_name}
🎯BOTทำนาย: {bot_signal}

{pattern_description}"""
            return send_telegram(theory_text)
            
    except Exception as e:
        print(f"Pattern theory explanation error: {e}")
        return 500    

def send_basic_pattern_info(pattern_name, confidence, method):
    """Send basic pattern info with BOT prediction"""
    try:
        bot_signal = get_pattern_signal(pattern_name)
        
        # สร้างข้อมูลพื้นฐาน
        basic_description = f"""📊 {pattern_name.replace('_', ' ')} PATTERN
🎯BOTทำนาย: {bot_signal}

🎯 Confidence Level: {confidence*100:.1f}%
🔧 Detection Method: {method}

🔍 Pattern Type: {get_pattern_type(pattern_name)}
📈 Market Context: {get_pattern_context(pattern_name)}

🤖 AI Recommendation:
• Signal: {bot_signal}
• Entry Timing: รอการยืนยัน
• Risk Management: 1-2% per trade

📚 รายละเอียดเพิ่มเติม:
• ศึกษาจากแหล่งอ้างอิง Technical Analysis
• สังเกตพฤติกรรมราคาในอนาคต
• บันทึกผลลัพธ์เพื่อการเรียนรู้

⚠️ คำแนะนำ: ใช้ร่วมกับ indicators อื่นๆ"""

        # ส่งข้อความ
        send_status = send_telegram(basic_description)
        print(f"Basic pattern info with BOT prediction sent for {pattern_name}: Status {send_status}")
        return send_status
        
    except Exception as e:
        print(f"Basic pattern info error: {e}")
        return 500

def get_pattern_type(pattern_name):
    """Get pattern type classification"""
    reversal_patterns = ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'DOJI', 'HAMMER', 'SHOOTING_STAR']
    continuation_patterns = ['BULL_FLAG', 'BEAR_FLAG', 'ASCENDING_TRIANGLE', 'RECTANGLE']
    
    if pattern_name in reversal_patterns:
        return 'Reversal Pattern (แพทเทิร์นกลับตัว)'
    elif pattern_name in continuation_patterns:
        return 'Continuation Pattern (แพทเทิร์นต่อเนื่อง)'
    else:
        return 'Chart Pattern (แพทเทิร์นกราฟ)'

def get_pattern_context(pattern_name):
    """Get pattern market context"""
    bullish_patterns = ['DOUBLE_BOTTOM', 'HAMMER', 'BULL_FLAG', 'ASCENDING_TRIANGLE']
    bearish_patterns = ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'SHOOTING_STAR', 'BEAR_FLAG']
    
    if pattern_name in bullish_patterns:
        return 'Bullish Context (บริบทขาขึ้น)'
    elif pattern_name in bearish_patterns:
        return 'Bearish Context (บริบทขาลง)'
    else:
        return 'Neutral Context (รอการยืนยัน)'

def get_pattern_signal(pattern_name):
    """Get BOT prediction signal for each pattern"""
    bullish_patterns = [
        'DOUBLE_BOTTOM', 'HAMMER', 'BULL_FLAG', 'ASCENDING_TRIANGLE', 
        'WEDGE_FALLING', 'CUP_AND_HANDLE', 'INVERSE_HEAD_SHOULDERS',
        'ENGULFING_BULLISH', 'MORNING_STAR', 'THREE_WHITE_SOLDIERS',
        'PIERCING_LINE', 'HARAMI_BULLISH', 'TWEEZER_BOTTOM',
        'INVERTED_HAMMER', 'MARUBOZU'  # Bullish Marubozu
    ]
    
    bearish_patterns = [
        'HEAD_SHOULDERS', 'DOUBLE_TOP', 'SHOOTING_STAR', 'BEAR_FLAG',
        'DESCENDING_TRIANGLE', 'WEDGE_RISING', 'ENGULFING_BEARISH',
        'EVENING_STAR', 'THREE_BLACK_CROWS', 'DARK_CLOUD_COVER',
        'HARAMI_BEARISH', 'TWEEZER_TOP', 'HANGING_MAN'
    ]
    
    neutral_patterns = [
        'DOJI', 'SPINNING_TOP', 'RECTANGLE', 'SYMMETRICAL_TRIANGLE',
        'DIAMOND', 'PENNANT'
    ]
    
    if pattern_name in bullish_patterns:
        return "🟢 BUY"
    elif pattern_name in bearish_patterns:
        return "🔴 SELL"
    elif pattern_name in neutral_patterns:
        return "🟡 WAIT/BREAKOUT"
    else:
        return "⏸️ WAIT"

def send_telegram_with_chart(message_text, chart_buffer):
    """Send message with chart image to Telegram"""
    try:
        if not BOT_TOKEN or not CHAT_ID:
            print("⚠️ Telegram credentials not configured")
            return 400
            
        if chart_buffer is None:
            # Send text only if chart failed
            return send_telegram(message_text)
        
        # Send photo with caption
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        
        files = {'photo': ('chart.png', chart_buffer, 'image/png')}
        data = {
            'chat_id': CHAT_ID,
            'caption': message_text,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        print(f"Telegram chart response: {response.status_code}")
        
        # Close buffer
        chart_buffer.close()
        
        return response.status_code
        
    except Exception as e:
        print(f"Telegram chart send error: {e}")
        # Fallback to text-only message
        return send_telegram(message_text)

def run_pattern_ai_shared_with_chart(shared_df):
    """Enhanced Pattern AI system with chart generation"""
    try:
        if shared_df is None or len(shared_df) < 20:
            return "❌ ไม่สามารถดึงข้อมูลสำหรับ Pattern Detection ได้", None, None, None
        
        detector = AdvancedPatternDetector()
        all_patterns = detector.detect_all_patterns(shared_df.tail(50))
        pattern_info = all_patterns[0]  # ใช้ pattern แรกสำหรับการแสดงผลหลัก
        trading_signals = detector.predict_signals(shared_df)
        
        current_data = shared_df.iloc[-1]
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
        
        # Create chart
        chart_buffer = create_candlestick_chart(shared_df, trading_signals, pattern_info)
        
        # Get pattern description
        pattern_description = get_pattern_description(pattern_info['pattern_name'])
        
        message = """🚀 AI PATTERN DETECTION BOT
⏰ {current_time} | 💰 XAUUSD (1H)
💾 SHARED DATA SOURCE

💰 MARKET DATA:
Open: ${open_price} | High: ${high_price}
Low: ${low_price} | Close: ${close_price}

🔍 PATTERN DETECTED:
{pattern_desc}
🤖 Method: {method} | 🎯 Confidence: {pattern_confidence}%

💹 TECHNICAL INDICATORS (SHARED):
RSI: {rsi} ({rsi_status})
EMA10: ${ema10} ({ema10_status})
EMA21: ${ema21} ({ema21_status})

🚦 PATTERN AI SIGNAL: {action_signal}""".format(
            current_time=current_time,
            open_price=f"${current_data['open']:,.2f}",
            high_price=f"${current_data['high']:,.2f}",
            low_price=f"${current_data['low']:,.2f}",
            close_price=f"${current_data['close']:,.2f}",
            pattern_desc=pattern_desc.get(pattern_info['pattern_name'], pattern_info['pattern_name']),
            method=pattern_info['method'],
            pattern_confidence=f"{pattern_info['confidence']*100:.1f}",
            rsi=f"{trading_signals['rsi']:.1f}",
            rsi_status='Oversold' if trading_signals['rsi']<30 else 'Overbought' if trading_signals['rsi']>70 else 'Neutral',
            ema10=f"${trading_signals['ema10']:,.2f}",
            ema10_status='Above' if trading_signals['current_price']>trading_signals['ema10'] else 'Below',
            ema21=f"${trading_signals['ema21']:,.2f}",
            ema21_status='Above' if trading_signals['current_price']>trading_signals['ema21'] else 'Below',
            action_signal=action_emoji[trading_signals['action']]
        )

        if trading_signals['action'] != 'WAIT':
            message += """

💼 TRADING SETUP:
🎯 Entry: ${entry_price}
🟢 TP1: ${tp1} | TP2: ${tp2} | TP3: ${tp3}
🔴 SL: ${sl}
💯 Pattern Confidence: {confidence}%

⚠️ Risk: ใช้เงินเพียง 1-2% ต่อออเดอร์""".format(
                entry_price=f"${trading_signals['entry_price']:,.2f}",
                tp1=f"${trading_signals['tp1']:,.2f}",
                tp2=f"${trading_signals['tp2']:,.2f}",
                tp3=f"${trading_signals['tp3']:,.2f}",
                sl=f"${trading_signals['sl']:,.2f}",
                confidence=f"{trading_signals['confidence']*100:.1f}"
            )
        else:
            message += """

⏳ รอ Pattern ที่ชัดเจนกว่า
💰 Current: ${current_price}
🔍 กำลังวิเคราะห์แพทเทิร์นใหม่...""".format(
                current_price=f"${trading_signals['current_price']:,.2f}"
            )

        return message, chart_buffer, pattern_description, pattern_info
        
    except Exception as e:
        return f"❌ PATTERN AI ERROR: {str(e)}", None, None, None

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

def run_ai_once_shared(shared_df):
    """Original AI system using shared data"""
    global last_signal
    try:
        if shared_df is None or len(shared_df) < 20:
            return "❌ ไม่สามารถดึงข้อมูลได้"
            
        # Use shared data และตรวจสอบว่ามี indicators ที่จำเป็น
        required_features = ["rsi", "ema", "price_change"]
        df = shared_df.copy()
        
        # ตรวจสอบว่า indicators พร้อมใช้งาน
        df_clean = df.dropna(subset=required_features)
        if len(df_clean) < 5:
            current_price = df["close"].iloc[-1]
            ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
            return (
                f"🤖 ORIGINAL BOT (RSI+EMA+Price Change)\n"
                f"⏰ {ts_txt}\n"
                f"XAUUSD ราคาปัจจุบัน ${current_price:,.2f}\n"
                f"⚠️ ข้อมูล indicators ไม่เพียงพอ"
            )

        # ใช้ข้อมูลล่าสุดที่มี indicators ครบ
        latest = df_clean.iloc[-1]
        
        x = [latest["rsi"], latest["ema"], latest["price_change"]]
        x = [val if not pd.isna(val) else 0 for val in x]  # Replace NaN with 0
        
        price = latest["close"]
        ema_val = latest["ema"]
        rsi_val = latest["rsi"]

        # OHLC data (ใช้จาก shared data)
        latest_raw = df.iloc[-1]  # ข้อมูลล่าสุด (อาจมี NaN ใน indicators แต่ OHLC ครบ)
        o = latest_raw["open"]
        h = latest_raw["high"]
        l = latest_raw["low"]
        c = latest_raw["close"]  # Real-time price

        # Prediction
        if model is not None:
            try:
                pred = int(model.predict([x])[0])
            except:
                pred = 1 if rsi_val < 50 else 0
        else:
            pred = 1 if rsi_val < 50 else 0
            
        reason_text, _ = explain_prediction(model, np.array(x), price, ema_val, rsi_val, pred)
        signal, (tp1, tp2, tp3, sl) = calc_targets(pred, price)
        ts_txt = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")

        msg = """🤖 ORIGINAL BOT (RSI+EMA+Price Change)
⏰ {timestamp}
💰 XAUUSD TF H1
💾 SHARED DATA SOURCE
Open = ${open_val} | High = ${high_val}
Low = ${low_val} | Close = ${close_val}
ราคาปัจจุบัน = ${current_val}

🎯 BOT ทำนาย: {signal_result}
{reasoning}

🎯 TP1: ${tp1_val} | TP2: ${tp2_val}
🎯 TP3: ${tp3_val} | 🔴 SL: ${sl_val}""".format(
            timestamp=ts_txt,
            open_val=f"${o:,.2f}",
            high_val=f"${h:,.2f}",
            low_val=f"${l:,.2f}",
            close_val=f"${c:,.2f}",
            current_val=f"${price:,.2f}",
            signal_result=signal,
            reasoning=reason_text,
            tp1_val=f"${tp1:,.2f}",
            tp2_val=f"${tp2:,.2f}",
            tp3_val=f"${tp3:,.2f}",
            sl_val=f"${sl:,.2f}"
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

    def predict_signals(self, df):
        """Predict trading signals based on patterns - รับ shared data"""
        try:
            current_price = df['close'].iloc[-1]
            
            # ใช้ indicators ที่คำนวณไว้แล้วใน shared data
            current_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
            current_ema10 = df['ema'].iloc[-1] if not pd.isna(df['ema'].iloc[-1]) else current_price
            current_ema21 = df['ema_21'].iloc[-1] if not pd.isna(df['ema_21'].iloc[-1]) else current_price
            
            pattern_id = 0  # Default to no pattern
            confidence = 0.5
            
            # Pattern-based signals (simplified for shared data)
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
                # Default pattern-based signal
                if current_price > current_ema10 and current_price > current_ema21:
                    action = "BUY"
                    entry_price = current_price + (current_price * 0.001)
                    tp1 = current_price * 1.005
                    tp2 = current_price * 1.010
                    tp3 = current_price * 1.015
                    sl = current_price * 0.990
                elif current_price < current_ema10 and current_price < current_ema21:
                    action = "SELL"
                    entry_price = current_price - (current_price * 0.001)
                    tp1 = current_price * 0.995
                    tp2 = current_price * 0.990
                    tp3 = current_price * 0.985
                    sl = current_price * 1.010
                else:
                    action = "WAIT"
                    entry_price = current_price
                    tp1 = tp2 = tp3 = sl = current_price
                
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

def run_pattern_ai_shared(shared_df):
    """Pattern AI system using shared data (legacy - text only)"""
    try:
        if shared_df is None or len(shared_df) < 20:
            return "❌ ไม่สามารถดึงข้อมูลสำหรับ Pattern Detection ได้"
        
        detector = AdvancedPatternDetector()
        all_patterns = detector.detect_all_patterns(shared_df.tail(50))
        pattern_info = all_patterns[0]  # ใช้ pattern แรกสำหรับการแสดงผลหลัก
        trading_signals = detector.predict_signals(shared_df)
        
        current_data = shared_df.iloc[-1]
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
        
        message = """🚀 AI PATTERN DETECTION BOT
⏰ {current_time} | 💰 XAUUSD (1H)
💾 SHARED DATA SOURCE

💰 MARKET DATA:
Open: {open_price} | High: {high_price}
Low: {low_price} | Close: {close_price}

🔍 PATTERN DETECTED:
{pattern_desc}
🤖 Method: {method} | 🎯 Confidence: {pattern_confidence}%

💹 TECHNICAL INDICATORS (SHARED):
RSI: {rsi} ({rsi_status})
EMA10: {ema10} ({ema10_status})
EMA21: {ema21} ({ema21_status})

🚦 PATTERN AI SIGNAL: {action_signal}""".format(
            current_time=current_time,
            open_price=f"${current_data['open']:,.2f}",
            high_price=f"${current_data['high']:,.2f}",
            low_price=f"${current_data['low']:,.2f}",
            close_price=f"${current_data['close']:,.2f}",
            pattern_desc=pattern_desc.get(pattern_info['pattern_name'], pattern_info['pattern_name']),
            method=pattern_info['method'],
            pattern_confidence=f"{pattern_info['confidence']*100:.1f}",
            rsi=f"{trading_signals['rsi']:.1f}",
            rsi_status='Oversold' if trading_signals['rsi']<30 else 'Overbought' if trading_signals['rsi']>70 else 'Neutral',
            ema10=f"${trading_signals['ema10']:,.2f}",
            ema10_status='Above' if trading_signals['current_price']>trading_signals['ema10'] else 'Below',
            ema21=f"${trading_signals['ema21']:,.2f}",
            ema21_status='Above' if trading_signals['current_price']>trading_signals['ema21'] else 'Below',
            action_signal=action_emoji[trading_signals['action']]
        )

        if trading_signals['action'] != 'WAIT':
            message += """

💼 TRADING SETUP:
🎯 Entry: {entry_price}
🟢 TP1: {tp1} | TP2: {tp2} | TP3: {tp3}
🔴 SL: {sl}
💯 Pattern Confidence: {confidence}%

⚠️ Risk: ใช้เงินเพียง 1-2% ต่อออเดอร์""".format(
                entry_price=f"${trading_signals['entry_price']:,.2f}",
                tp1=f"${trading_signals['tp1']:,.2f}",
                tp2=f"${trading_signals['tp2']:,.2f}",
                tp3=f"${trading_signals['tp3']:,.2f}",
                sl=f"${trading_signals['sl']:,.2f}",
                confidence=f"{trading_signals['confidence']*100:.1f}"
            )
        else:
            message += """

⏳ รอ Pattern ที่ชัดเจนกว่า
💰 Current: {current_price}
🔍 กำลังวิเคราะห์แพทเทิร์นใหม่...""".format(
                current_price=f"${trading_signals['current_price']:,.2f}"
            )

        return message
        
    except Exception as e:
        return f"❌ PATTERN AI ERROR: {str(e)}"


# ====================== Extended Pattern Detection System ======================
class AdvancedPatternDetector:
    def __init__(self):
        self.patterns = {
            0: "NO_PATTERN",
            1: "HEAD_SHOULDERS", 
            2: "DOUBLE_TOP",
            3: "DOUBLE_BOTTOM",
            4: "ASCENDING_TRIANGLE",
            5: "BULL_FLAG",
            # Chart Patterns
            6: "DESCENDING_TRIANGLE",
            7: "SYMMETRICAL_TRIANGLE", 
            8: "BEAR_FLAG",
            9: "PENNANT",
            10: "WEDGE_RISING",
            11: "WEDGE_FALLING",
            12: "CUP_AND_HANDLE",
            13: "INVERSE_HEAD_SHOULDERS",
            14: "RECTANGLE",
            15: "DIAMOND",
            # Single Candlestick Patterns
            16: "DOJI",
            17: "HAMMER",
            18: "HANGING_MAN",
            19: "SHOOTING_STAR",
            20: "INVERTED_HAMMER",
            21: "MARUBOZU",
            22: "SPINNING_TOP",
            # Multiple Candlestick Patterns
            23: "ENGULFING_BULLISH",
            24: "ENGULFING_BEARISH",
            25: "PIERCING_LINE",
            26: "DARK_CLOUD_COVER",
            27: "MORNING_STAR",
            28: "EVENING_STAR",
            29: "THREE_WHITE_SOLDIERS",
            30: "THREE_BLACK_CROWS",
            31: "HARAMI_BULLISH",
            32: "HARAMI_BEARISH",
            33: "TWEEZER_TOP",
            34: "TWEEZER_BOTTOM"
        }

    def predict_signals(self, df):
        """Predict trading signals based on patterns and technical indicators"""
        try:
            current_price = df['close'].iloc[-1]
            
            # ใช้ indicators ที่คำนวณไว้แล้วใน shared data
            current_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
            current_ema10 = df['ema'].iloc[-1] if not pd.isna(df['ema'].iloc[-1]) else current_price
            current_ema21 = df['ema_21'].iloc[-1] if not pd.isna(df['ema_21'].iloc[-1]) else current_price
            
            # ตรวจสอบ pattern หลัก
            all_patterns = self.detect_all_patterns(df.tail(50))
            main_pattern = all_patterns[0] if all_patterns else {'pattern_name': 'NO_PATTERN', 'confidence': 0.5}
            
            # สร้างสัญญาณตาม pattern และ indicators
            confidence = main_pattern.get('confidence', 0.5)
            
            # Pattern-based signal logic
            if main_pattern['pattern_name'] in ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'EVENING_STAR', 'SHOOTING_STAR', 'ENGULFING_BEARISH']:
                action = "SELL"
                entry_price = current_price - (current_price * 0.0005)
                tp1 = current_price * 0.997
                tp2 = current_price * 0.994
                tp3 = current_price * 0.991
                sl = current_price * 1.005
                
            elif main_pattern['pattern_name'] in ['DOUBLE_BOTTOM', 'ASCENDING_TRIANGLE', 'BULL_FLAG', 'HAMMER', 'MORNING_STAR', 'ENGULFING_BULLISH']:
                action = "BUY"
                entry_price = current_price + (current_price * 0.0005)
                tp1 = current_price * 1.003
                tp2 = current_price * 1.006
                tp3 = current_price * 1.009
                sl = current_price * 0.995
                
            elif main_pattern['pattern_name'] in ['BEAR_FLAG', 'DESCENDING_TRIANGLE', 'WEDGE_RISING']:
                action = "SELL"
                entry_price = current_price - (current_price * 0.001)
                tp1 = current_price * 0.995
                tp2 = current_price * 0.990
                tp3 = current_price * 0.985
                sl = current_price * 1.010
                
            elif main_pattern['pattern_name'] in ['WEDGE_FALLING', 'CUP_AND_HANDLE', 'INVERSE_HEAD_SHOULDERS']:
                action = "BUY"
                entry_price = current_price + (current_price * 0.001)
                tp1 = current_price * 1.005
                tp2 = current_price * 1.010
                tp3 = current_price * 1.015
                sl = current_price * 0.990
                
            else:
                # ใช้ RSI และ EMA เป็นหลัก เมื่อไม่มี pattern ชัดเจน
                if current_rsi < 30 and current_price < current_ema10:
                    action = "BUY"
                    entry_price = current_price + (current_price * 0.0005)
                    tp1 = current_price * 1.003
                    tp2 = current_price * 1.006
                    tp3 = current_price * 1.009
                    sl = current_price * 0.995
                    confidence = 0.65
                    
                elif current_rsi > 70 and current_price > current_ema10:
                    action = "SELL"
                    entry_price = current_price - (current_price * 0.0005)
                    tp1 = current_price * 0.997
                    tp2 = current_price * 0.994
                    tp3 = current_price * 0.991
                    sl = current_price * 1.005
                    confidence = 0.65
                    
                elif current_price > current_ema10 and current_price > current_ema21:
                    action = "BUY"
                    entry_price = current_price + (current_price * 0.001)
                    tp1 = current_price * 1.005
                    tp2 = current_price * 1.010
                    tp3 = current_price * 1.015
                    sl = current_price * 0.990
                    confidence = 0.55
                    
                elif current_price < current_ema10 and current_price < current_ema21:
                    action = "SELL"
                    entry_price = current_price - (current_price * 0.001)
                    tp1 = current_price * 0.995
                    tp2 = current_price * 0.990
                    tp3 = current_price * 0.985
                    sl = current_price * 1.010
                    confidence = 0.55
                    
                else:
                    action = "WAIT"
                    entry_price = current_price
                    tp1 = tp2 = tp3 = sl = current_price
                    confidence = 0.40
                
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
                'ema21': round(current_ema21, 2),
                'pattern_name': main_pattern['pattern_name'],
                'pattern_confidence': main_pattern.get('confidence', 0.5)
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
                'ema21': round(current_price, 2),
                'pattern_name': 'ERROR',
                'pattern_confidence': 0.30
            }

    def detect_all_patterns(self, df):
        """Detect ALL patterns instead of just the first one found"""
        try:
            if len(df) < 20:
                return [{
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'INSUFFICIENT_DATA'
                }]
        
            all_patterns = []
        
            # Check ALL candlestick patterns
            candlestick_patterns = self.detect_all_candlestick_patterns(df)
            all_patterns.extend(candlestick_patterns)
        
            # Check ALL chart patterns
            chart_patterns = self.detect_all_chart_patterns(df)
            all_patterns.extend(chart_patterns)
        
            # ถ้าไม่พบ pattern ใดเลย
            if not all_patterns or all(p['pattern_name'] == 'NO_PATTERN' for p in all_patterns):
                return [{
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'NO_PATTERNS_FOUND'
                }]
        
            # กรอง patterns ที่มีความมั่นใจสูงกว่า 60% และไม่ใช่ NO_PATTERN
            valid_patterns = [p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN' and p['confidence'] > 0.60]
        
            if not valid_patterns:
                return [{
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'LOW_CONFIDENCE_PATTERNS'
                }]
        
            # เรียงตาม confidence สูงสุด
            valid_patterns.sort(key=lambda x: x['confidence'], reverse=True)
            return valid_patterns[:5]  # ส่งสูงสุด 5 patterns
        
        except Exception as e:
            print(f"Multiple pattern detection error: {e}")
            return [{
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.30,
                'method': 'ERROR'
            }]

    def detect_all_candlestick_patterns(self, df):
        """Detect ALL candlestick patterns - FIXED VERSION"""
        try:
            patterns_found = []
            recent_data = df.tail(5)
            if len(recent_data) < 3:
                return patterns_found
        
            # Single candlestick patterns
            last_candle = recent_data.iloc[-1]
            single_patterns = self.detect_all_single_candlestick(last_candle)
            patterns_found.extend(single_patterns)
        
            # เพิ่ม Hanging Man detection สำหรับ single candle
            hanging_man = self.check_hanging_man(last_candle)
            patterns_found.extend(hanging_man)
        
            # Two candlestick patterns
            if len(recent_data) >= 2:
                two_patterns = self.detect_all_two_candlestick(recent_data.tail(2))
                patterns_found.extend(two_patterns)
            
                # เพิ่ม Tweezer patterns
                tweezer_patterns = self.check_tweezer_patterns(recent_data.tail(2))
                patterns_found.extend(tweezer_patterns)
        
            # Three candlestick patterns
            if len(recent_data) >= 3:
                three_patterns = self.detect_all_three_candlestick(recent_data.tail(3))
                patterns_found.extend(three_patterns)
        
            return patterns_found
        
        except Exception as e:
            print(f"All candlestick patterns error: {e}")
            return []   

    def detect_all_single_candlestick(self, candle):
        """Detect ALL single candlestick patterns - COMPLETE VERSION"""
        try:
            patterns = []
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
        
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            candle_range = high_price - low_price
        
            if candle_range == 0:
                return patterns
        
            body_ratio = body_size / candle_range
            upper_ratio = upper_shadow / candle_range
            lower_ratio = lower_shadow / candle_range
        
            # DOJI
            if body_ratio < 0.1 and (upper_ratio > 0.3 or lower_ratio > 0.3):
                patterns.append({'pattern_id': 16, 'pattern_name': 'DOJI', 'confidence': 0.80, 'method': 'SINGLE_CANDLESTICK'})
        
            # HAMMER
            if body_ratio < 0.3 and lower_ratio > 0.6 and upper_ratio < 0.1:
                patterns.append({'pattern_id': 17, 'pattern_name': 'HAMMER', 'confidence': 0.75, 'method': 'SINGLE_CANDLESTICK'})
        
            # SHOOTING STAR
            if body_ratio < 0.3 and upper_ratio > 0.6 and lower_ratio < 0.1:
                patterns.append({'pattern_id': 19, 'pattern_name': 'SHOOTING_STAR', 'confidence': 0.75, 'method': 'SINGLE_CANDLESTICK'})
        
            # INVERTED HAMMER
            if body_ratio < 0.3 and upper_ratio > 0.6 and lower_ratio < 0.1 and close_price > open_price:
                patterns.append({'pattern_id': 20, 'pattern_name': 'INVERTED_HAMMER', 'confidence': 0.70, 'method': 'SINGLE_CANDLESTICK'})
        
            # MARUBOZU - แยกเป็น Bullish/Bearish
            if body_ratio > 0.9:
                if close_price > open_price:
                    patterns.append({'pattern_id': 21, 'pattern_name': 'MARUBOZU', 'confidence': 0.85, 'method': 'SINGLE_CANDLESTICK'})
                else:
                    patterns.append({'pattern_id': 21, 'pattern_name': 'MARUBOZU', 'confidence': 0.85, 'method': 'SINGLE_CANDLESTICK'})
        
            # SPINNING TOP
            if body_ratio < 0.3 and upper_ratio > 0.2 and lower_ratio > 0.2:
                patterns.append({'pattern_id': 22, 'pattern_name': 'SPINNING_TOP', 'confidence': 0.65, 'method': 'SINGLE_CANDLESTICK'})
        
            return patterns
        
        except Exception as e:
            print(f"All single candlestick error: {e}")
            return []    

    def detect_all_two_candlestick(self, candles):
        """Detect ALL two-candlestick patterns"""
        try:
            patterns = []
            first = candles.iloc[0]
            second = candles.iloc[1]
        
            first_body = abs(first['close'] - first['open'])
            second_body = abs(second['close'] - second['open'])
        
            # ตรวจสอบทุก patterns
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                second_body > first_body * 1.1):
                patterns.append({'pattern_id': 23, 'pattern_name': 'ENGULFING_BULLISH', 'confidence': 0.80, 'method': 'TWO_CANDLESTICK'})
        
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                second_body > first_body * 1.1):
                patterns.append({'pattern_id': 24, 'pattern_name': 'ENGULFING_BEARISH', 'confidence': 0.80, 'method': 'TWO_CANDLESTICK'})
        
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] < first['low'] and second['close'] > (first['open'] + first['close']) / 2):
                patterns.append({'pattern_id': 25, 'pattern_name': 'PIERCING_LINE', 'confidence': 0.75, 'method': 'TWO_CANDLESTICK'})
        
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] > first['high'] and second['close'] < (first['open'] + first['close']) / 2):
                patterns.append({'pattern_id': 26, 'pattern_name': 'DARK_CLOUD_COVER', 'confidence': 0.75, 'method': 'TWO_CANDLESTICK'})
        
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                second_body < first_body * 0.6):
                patterns.append({'pattern_id': 31, 'pattern_name': 'HARAMI_BULLISH', 'confidence': 0.70, 'method': 'TWO_CANDLESTICK'})
        
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                second_body < first_body * 0.6):
                patterns.append({'pattern_id': 32, 'pattern_name': 'HARAMI_BEARISH', 'confidence': 0.70, 'method': 'TWO_CANDLESTICK'})
        
            return patterns
        
        except Exception as e:
            print(f"All two candlestick error: {e}")
            return []

    def detect_all_three_candlestick(self, candles):
        """Detect ALL three-candlestick patterns"""
        try:
            patterns = []
            first = candles.iloc[0]
            second = candles.iloc[1]
            third = candles.iloc[2]
        
            # ตรวจสอบทุก patterns
            if (first['close'] < first['open'] and third['close'] > third['open'] and
                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and
                second['high'] < first['close'] and third['open'] > second['high'] and
                third['close'] > (first['open'] + first['close']) / 2):
                patterns.append({'pattern_id': 27, 'pattern_name': 'MORNING_STAR', 'confidence': 0.85, 'method': 'THREE_CANDLESTICK'})
        
            if (first['close'] > first['open'] and third['close'] < third['open'] and
                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and
                second['low'] > first['close'] and third['open'] < second['low'] and
                third['close'] < (first['open'] + first['close']) / 2):
                patterns.append({'pattern_id': 28, 'pattern_name': 'EVENING_STAR', 'confidence': 0.85, 'method': 'THREE_CANDLESTICK'})
        
            if (first['close'] > first['open'] and second['close'] > second['open'] and third['close'] > third['open'] and
                second['close'] > first['close'] and third['close'] > second['close'] and
                second['open'] > first['open'] and third['open'] > second['open']):
                patterns.append({'pattern_id': 29, 'pattern_name': 'THREE_WHITE_SOLDIERS', 'confidence': 0.80, 'method': 'THREE_CANDLESTICK'})
        
            if (first['close'] < first['open'] and second['close'] < second['open'] and third['close'] < third['open'] and
                second['close'] < first['close'] and third['close'] < second['close'] and
                second['open'] < first['open'] and third['open'] < second['open']):
                patterns.append({'pattern_id': 30, 'pattern_name': 'THREE_BLACK_CROWS', 'confidence': 0.80, 'method': 'THREE_CANDLESTICK'})
        
            return patterns
        
        except Exception as e:
            print(f"All three candlestick error: {e}")
            return []

    def detect_all_chart_patterns(self, df):
        """Detect ALL chart patterns - FIXED VERSION with all 35 patterns"""
        try:
            patterns_found = []
            highs = df['high'].values[-30:]
            lows = df['low'].values[-30:]
            closes = df['close'].values[-30:]
        
            # ตรวจสอบ Chart Patterns ทั้งหมด
            patterns_found.extend(self.check_head_shoulders(df))
            patterns_found.extend(self.check_double_top(highs, lows))
            patterns_found.extend(self.check_double_bottom(lows, closes))  # เพิ่ม
            patterns_found.extend(self.check_ascending_triangle(highs, lows))
            patterns_found.extend(self.check_descending_triangle(highs, lows))
            patterns_found.extend(self.check_bull_flag(closes, highs, lows))
            patterns_found.extend(self.check_bear_flag(closes, highs, lows))
            patterns_found.extend(self.check_symmetrical_triangle(highs, lows))
            patterns_found.extend(self.check_wedge_patterns(highs, lows, closes))
            patterns_found.extend(self.check_cup_and_handle(closes, highs, lows))
            patterns_found.extend(self.check_inverse_head_shoulders(lows, closes))  # เพิ่ม
            patterns_found.extend(self.check_rectangle(highs, lows))
            patterns_found.extend(self.check_diamond_pattern(highs, lows))  # เพิ่ม
            patterns_found.extend(self.check_pennant_pattern(highs, lows, closes))  # เพิ่ม
        
            # กรอง patterns ที่ซ้ำกัน
            unique_patterns = []
            seen_patterns = set()
            for pattern in patterns_found:
                if pattern['pattern_name'] not in seen_patterns:
                    unique_patterns.append(pattern)
                    seen_patterns.add(pattern['pattern_name'])
        
            return unique_patterns
        
        except Exception as e:
            print(f"All chart patterns error: {e}")
            return []    

    def check_head_shoulders(self, df):
        """Check for head and shoulders - return as list"""
        pattern = self.detect_existing_patterns(df)
        if pattern['pattern_name'] == 'HEAD_SHOULDERS':
            return [pattern]
        return []

    def check_double_top(self, highs, lows):
        """Check for double top - return as list"""
        try:
            # Find peaks (local maxima)
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
        
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.02:
                    return [{'pattern_id': 2, 'pattern_name': 'DOUBLE_TOP', 'confidence': 0.75, 'method': 'CHART_PATTERN'}]
            return []
        except:
            return []

    def check_double_bottom(self, lows, closes):
        """Check for double bottom - return as list"""
        pattern = self.detect_double_bottom(lows, closes)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_ascending_triangle(self, highs, lows):
        """Check for ascending triangle - return as list"""
        try:
            resistance = np.max(highs[-15:])
            support_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
            if support_trend > 0.001:  # Ascending support
                resistance_touches = np.sum(np.abs(highs - resistance) < 0.001 * resistance)
                if resistance_touches >= 2:
                    return [{'pattern_id': 4, 'pattern_name': 'ASCENDING_TRIANGLE', 'confidence': 0.70, 'method': 'CHART_PATTERN'}]
            return []
        except:
            return []

    def check_inverse_head_shoulders(self, lows, closes):
        """Check for inverse head and shoulders - return as list"""
        pattern = self.detect_inverse_head_shoulders(lows, closes)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_diamond_pattern(self, highs, lows):
        """Check for diamond pattern - return as list"""
        pattern = self.detect_diamond_pattern(highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_pennant_pattern(self, highs, lows, closes):
        """Check for pennant pattern - return as list"""
        pattern = self.detect_pennant_pattern(highs, lows, closes)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []
    
    def check_descending_triangle(self, highs, lows):
        """Check for descending triangle - return as list"""
        pattern = self.detect_descending_triangle(highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_symmetrical_triangle(self, highs, lows):
        """Check for symmetrical triangle - return as list"""
        pattern = self.detect_symmetrical_triangle(highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_bear_flag(self, closes, highs, lows):
        """Check for bear flag - return as list"""
        pattern = self.detect_bear_flag(closes, highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_wedge_patterns(self, highs, lows, closes):
        """Check for wedge patterns - return as list"""
        pattern = self.detect_wedge_patterns(highs, lows, closes)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_cup_and_handle(self, closes, highs, lows):
        """Check for cup and handle - return as list"""
        pattern = self.detect_cup_and_handle(closes, highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_rectangle(self, highs, lows):
        """Check for rectangle - return as list"""
        pattern = self.detect_rectangle(highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def check_hanging_man(self, candle):
        """Check for hanging man - return as list"""
        pattern = self.detect_hanging_man(candle)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

# เพิ่มฟังก์ชันที่ขาดหายไปใน AdvancedPatternDetector class

    def check_bull_flag(self, closes, highs, lows):
        """Check for bull flag - return as list"""
        pattern = self.detect_bull_flag(closes, highs, lows)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []

    def detect_bull_flag(self, closes, highs, lows):
        """Detect bull flag pattern"""
        try:
            if len(closes) < 20:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
        
            # Check for strong uptrend (flagpole)
            flagpole_start = closes[-20]
            flagpole_end = closes[-10]
            flagpole_gain = (flagpole_end - flagpole_start) / flagpole_start
        
            if flagpole_gain > 0.03:  # At least 3% gain
                # Check for flag consolidation
                flag_prices = closes[-10:]
                flag_volatility = np.std(flag_prices) / np.mean(flag_prices)
            
                if flag_volatility < 0.02:  # Low volatility consolidation
                    return {'pattern_id': 5, 'pattern_name': 'BULL_FLAG', 'confidence': 0.75, 'method': 'CHART_PATTERN'}
        
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_BULL_FLAG'}
        
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'BULL_FLAG_ERROR'}
    
    def check_tweezer_patterns(self, candles):
        """Check for tweezer patterns - return as list"""
        pattern = self.detect_tweezer_patterns(candles)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []
    
    def check_existing_patterns(self, df):
        """Check for existing patterns - return as list"""
        pattern = self.detect_existing_patterns(df)
        return [pattern] if pattern['pattern_name'] != 'NO_PATTERN' else []
    
    def detect_pattern(self, df):
        """Advanced pattern detection with multiple pattern types"""
        try:
            if len(df) < 20:
                return {
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'INSUFFICIENT_DATA'
                }
            
            # Check candlestick patterns first (higher priority for recent signals)
            candlestick_pattern = self.detect_candlestick_patterns(df)
            if candlestick_pattern['pattern_name'] != 'NO_PATTERN':
                return candlestick_pattern
            
            # Check chart patterns
            chart_pattern = self.detect_chart_patterns(df)
            if chart_pattern['pattern_name'] != 'NO_PATTERN':
                return chart_pattern
            
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.50,
                'method': 'COMPREHENSIVE_SCAN'
            }
            
        except Exception as e:
            print(f"Advanced pattern detection error: {e}")
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.30,
                'method': 'ERROR'
            }

    def detect_candlestick_patterns(self, df):
        """Detect single and multiple candlestick patterns"""
        try:
            # Get last 5 candles for analysis
            recent_data = df.tail(5)
            if len(recent_data) < 3:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_CANDLES'}
            
            # Single candlestick patterns (check last candle)
            last_candle = recent_data.iloc[-1]
            single_pattern = self.detect_single_candlestick(last_candle)
            if single_pattern['pattern_name'] != 'NO_PATTERN':
                return single_pattern
            
            # Multiple candlestick patterns
            if len(recent_data) >= 2:
                two_candle_pattern = self.detect_two_candlestick(recent_data.tail(2))
                if two_candle_pattern['pattern_name'] != 'NO_PATTERN':
                    return two_candle_pattern
            
            if len(recent_data) >= 3:
                three_candle_pattern = self.detect_three_candlestick(recent_data.tail(3))
                if three_candle_pattern['pattern_name'] != 'NO_PATTERN':
                    return three_candle_pattern
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_CANDLESTICK_PATTERN'}
            
        except Exception as e:
            print(f"Candlestick pattern error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'CANDLESTICK_ERROR'}

    def detect_single_candlestick(self, candle):
        """Detect single candlestick patterns"""
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            candle_range = high_price - low_price
            
            if candle_range == 0:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INVALID_CANDLE'}
            
            body_ratio = body_size / candle_range
            upper_ratio = upper_shadow / candle_range
            lower_ratio = lower_shadow / candle_range
            
            # DOJI - Small body with long shadows
            if body_ratio < 0.1 and (upper_ratio > 0.3 or lower_ratio > 0.3):
                return {'pattern_id': 16, 'pattern_name': 'DOJI', 'confidence': 0.80, 'method': 'SINGLE_CANDLESTICK'}
            
            # HAMMER - Small body at top, long lower shadow
            if body_ratio < 0.3 and lower_ratio > 0.6 and upper_ratio < 0.1:
                return {'pattern_id': 17, 'pattern_name': 'HAMMER', 'confidence': 0.75, 'method': 'SINGLE_CANDLESTICK'}
            
            # SHOOTING STAR - Small body at bottom, long upper shadow
            if body_ratio < 0.3 and upper_ratio > 0.6 and lower_ratio < 0.1:
                return {'pattern_id': 19, 'pattern_name': 'SHOOTING_STAR', 'confidence': 0.75, 'method': 'SINGLE_CANDLESTICK'}
            
            # INVERTED HAMMER - Similar to shooting star but bullish context
            if body_ratio < 0.3 and upper_ratio > 0.6 and lower_ratio < 0.1 and close_price > open_price:
                return {'pattern_id': 20, 'pattern_name': 'INVERTED_HAMMER', 'confidence': 0.70, 'method': 'SINGLE_CANDLESTICK'}
            
            # MARUBOZU - Very large body, minimal shadows
            if body_ratio > 0.9:
                return {'pattern_id': 21, 'pattern_name': 'MARUBOZU', 'confidence': 0.85, 'method': 'SINGLE_CANDLESTICK'}
            
            # SPINNING TOP - Small body with upper and lower shadows
            if body_ratio < 0.3 and upper_ratio > 0.2 and lower_ratio > 0.2:
                return {'pattern_id': 22, 'pattern_name': 'SPINNING_TOP', 'confidence': 0.65, 'method': 'SINGLE_CANDLESTICK'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_SINGLE_PATTERN'}
            
        except Exception as e:
            print(f"Single candlestick error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'SINGLE_ERROR'}

    def detect_two_candlestick(self, candles):
        """Detect two-candlestick patterns"""
        try:
            first = candles.iloc[0]
            second = candles.iloc[1]
            
            first_body = abs(first['close'] - first['open'])
            second_body = abs(second['close'] - second['open'])
            
            # ENGULFING BULLISH - Large white candle engulfs previous black candle
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                second_body > first_body * 1.1):
                return {'pattern_id': 23, 'pattern_name': 'ENGULFING_BULLISH', 'confidence': 0.80, 'method': 'TWO_CANDLESTICK'}
            
            # ENGULFING BEARISH - Large black candle engulfs previous white candle
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                second_body > first_body * 1.1):
                return {'pattern_id': 24, 'pattern_name': 'ENGULFING_BEARISH', 'confidence': 0.80, 'method': 'TWO_CANDLESTICK'}
            
            # PIERCING LINE - Black candle followed by white candle that closes above midpoint
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] < first['low'] and second['close'] > (first['open'] + first['close']) / 2):
                return {'pattern_id': 25, 'pattern_name': 'PIERCING_LINE', 'confidence': 0.75, 'method': 'TWO_CANDLESTICK'}
            
            # DARK CLOUD COVER - White candle followed by black candle that opens above high and closes below midpoint
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] > first['high'] and second['close'] < (first['open'] + first['close']) / 2):
                return {'pattern_id': 26, 'pattern_name': 'DARK_CLOUD_COVER', 'confidence': 0.75, 'method': 'TWO_CANDLESTICK'}
            
            # HARAMI BULLISH - Large black candle followed by small white candle inside
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                second_body < first_body * 0.6):
                return {'pattern_id': 31, 'pattern_name': 'HARAMI_BULLISH', 'confidence': 0.70, 'method': 'TWO_CANDLESTICK'}
            
            # HARAMI BEARISH - Large white candle followed by small black candle inside
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                second_body < first_body * 0.6):
                return {'pattern_id': 32, 'pattern_name': 'HARAMI_BEARISH', 'confidence': 0.70, 'method': 'TWO_CANDLESTICK'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_TWO_PATTERN'}
            
        except Exception as e:
            print(f"Two candlestick error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'TWO_ERROR'}

    def detect_three_candlestick(self, candles):
        """Detect three-candlestick patterns"""
        try:
            first = candles.iloc[0]
            second = candles.iloc[1]
            third = candles.iloc[2]
            
            # MORNING STAR - Black, small body, white candles
            if (first['close'] < first['open'] and third['close'] > third['open'] and
                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and
                second['high'] < first['close'] and third['open'] > second['high'] and
                third['close'] > (first['open'] + first['close']) / 2):
                return {'pattern_id': 27, 'pattern_name': 'MORNING_STAR', 'confidence': 0.85, 'method': 'THREE_CANDLESTICK'}
            
            # EVENING STAR - White, small body, black candles
            if (first['close'] > first['open'] and third['close'] < third['open'] and
                abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3 and
                second['low'] > first['close'] and third['open'] < second['low'] and
                third['close'] < (first['open'] + first['close']) / 2):
                return {'pattern_id': 28, 'pattern_name': 'EVENING_STAR', 'confidence': 0.85, 'method': 'THREE_CANDLESTICK'}
            
            # THREE WHITE SOLDIERS - Three consecutive white candles with higher closes
            if (first['close'] > first['open'] and second['close'] > second['open'] and third['close'] > third['open'] and
                second['close'] > first['close'] and third['close'] > second['close'] and
                second['open'] > first['open'] and third['open'] > second['open']):
                return {'pattern_id': 29, 'pattern_name': 'THREE_WHITE_SOLDIERS', 'confidence': 0.80, 'method': 'THREE_CANDLESTICK'}
            
            # THREE BLACK CROWS - Three consecutive black candles with lower closes
            if (first['close'] < first['open'] and second['close'] < second['open'] and third['close'] < third['open'] and
                second['close'] < first['close'] and third['close'] < second['close'] and
                second['open'] < first['open'] and third['open'] < second['open']):
                return {'pattern_id': 30, 'pattern_name': 'THREE_BLACK_CROWS', 'confidence': 0.80, 'method': 'THREE_CANDLESTICK'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_THREE_PATTERN'}
            
        except Exception as e:
            print(f"Three candlestick error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'THREE_ERROR'}

    def detect_chart_patterns(self, df):
        """Detect chart patterns (existing + new ones)"""
        try:
            highs = df['high'].values[-30:]
            lows = df['low'].values[-30:]
            closes = df['close'].values[-30:]
            
            # DESCENDING TRIANGLE
            triangle_pattern = self.detect_descending_triangle(highs, lows)
            if triangle_pattern['pattern_name'] != 'NO_PATTERN':
                return triangle_pattern
            
            # SYMMETRICAL TRIANGLE
            sym_triangle = self.detect_symmetrical_triangle(highs, lows)
            if sym_triangle['pattern_name'] != 'NO_PATTERN':
                return sym_triangle
            
            # BEAR FLAG
            bear_flag = self.detect_bear_flag(closes, highs, lows)
            if bear_flag['pattern_name'] != 'NO_PATTERN':
                return bear_flag
            
            # WEDGE PATTERNS
            wedge_pattern = self.detect_wedge_patterns(highs, lows, closes)
            if wedge_pattern['pattern_name'] != 'NO_PATTERN':
                return wedge_pattern
            
            # CUP AND HANDLE
            cup_handle = self.detect_cup_and_handle(closes, highs, lows)
            if cup_handle['pattern_name'] != 'NO_PATTERN':
                return cup_handle
            
            # RECTANGLE
            rectangle = self.detect_rectangle(highs, lows)
            if rectangle['pattern_name'] != 'NO_PATTERN':
                return rectangle
            
            # Check existing patterns (HEAD_SHOULDERS, DOUBLE_TOP, etc.)
            existing_pattern = self.detect_existing_patterns(df)
            if existing_pattern['pattern_name'] != 'NO_PATTERN':
                return existing_pattern
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_CHART_PATTERN'}
            
        except Exception as e:
            print(f"Chart pattern error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'CHART_ERROR'}

    def detect_descending_triangle(self, highs, lows):
        """Detect descending triangle pattern"""
        try:
            # Find horizontal support and descending resistance
            recent_lows = lows[-15:]
            recent_highs = highs[-15:]
            
            # Check for horizontal support
            support_level = np.min(recent_lows)
            support_touches = np.sum(np.abs(recent_lows - support_level) < 0.001 * support_level)
            
            # Check for descending resistance
            if len(recent_highs) >= 10:
                slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
                if slope < -0.001 and support_touches >= 2:
                    return {'pattern_id': 6, 'pattern_name': 'DESCENDING_TRIANGLE', 'confidence': 0.75, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_DESC_TRIANGLE'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'DESC_TRIANGLE_ERROR'}

    def detect_symmetrical_triangle(self, highs, lows):
        """Detect symmetrical triangle pattern"""
        try:
            if len(highs) < 15 or len(lows) < 15:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # Descending highs
            high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            # Ascending lows
            low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            if high_slope < -0.001 and low_slope > 0.001:
                convergence = abs(recent_highs[-1] - recent_lows[-1]) < abs(recent_highs[0] - recent_lows[0]) * 0.8
                if convergence:
                    return {'pattern_id': 7, 'pattern_name': 'SYMMETRICAL_TRIANGLE', 'confidence': 0.70, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_SYM_TRIANGLE'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'SYM_TRIANGLE_ERROR'}

    def detect_bear_flag(self, closes, highs, lows):
        """Detect bear flag pattern"""
        try:
            if len(closes) < 20:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Check for strong downtrend (flagpole)
            flagpole_start = closes[-20]
            flagpole_end = closes[-10]
            flagpole_decline = (flagpole_end - flagpole_start) / flagpole_start
            
            if flagpole_decline < -0.03:  # At least 3% decline
                # Check for flag consolidation
                flag_prices = closes[-10:]
                flag_volatility = np.std(flag_prices) / np.mean(flag_prices)
                
                if flag_volatility < 0.02:  # Low volatility consolidation
                    return {'pattern_id': 8, 'pattern_name': 'BEAR_FLAG', 'confidence': 0.75, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_BEAR_FLAG'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'BEAR_FLAG_ERROR'}

    def detect_wedge_patterns(self, highs, lows, closes):
        """Detect rising and falling wedge patterns"""
        try:
            if len(highs) < 20:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Calculate slopes
            high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
            low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
            
            # RISING WEDGE - Both highs and lows ascending, but converging
            if high_slope > 0 and low_slope > 0 and high_slope < low_slope * 2:
                convergence_ratio = (recent_highs[-1] - recent_lows[-1]) / (recent_highs[0] - recent_lows[0])
                if convergence_ratio < 0.7:
                    return {'pattern_id': 10, 'pattern_name': 'WEDGE_RISING', 'confidence': 0.70, 'method': 'CHART_PATTERN'}
            
            # FALLING WEDGE - Both highs and lows descending, but converging
            if high_slope < 0 and low_slope < 0 and abs(low_slope) < abs(high_slope) * 2:
                convergence_ratio = (recent_highs[-1] - recent_lows[-1]) / (recent_highs[0] - recent_lows[0])
                if convergence_ratio < 0.7:
                    return {'pattern_id': 11, 'pattern_name': 'WEDGE_FALLING', 'confidence': 0.70, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_WEDGE'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'WEDGE_ERROR'}

    def detect_cup_and_handle(self, closes, highs, lows):
        """Detect cup and handle pattern"""
        try:
            if len(closes) < 30:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Cup formation - U-shaped recovery
            left_rim = closes[-30]
            cup_bottom = np.min(closes[-25:-5])
            right_rim = closes[-5]
            current_price = closes[-1]
            
            # Check cup proportions
            cup_depth = (left_rim - cup_bottom) / left_rim
            rim_similarity = abs(left_rim - right_rim) / left_rim
            
            if 0.1 < cup_depth < 0.5 and rim_similarity < 0.05:
                # Check for handle (slight pullback)
                handle_pullback = (right_rim - current_price) / right_rim
                if 0.01 < handle_pullback < 0.15:
                    return {'pattern_id': 12, 'pattern_name': 'CUP_AND_HANDLE', 'confidence': 0.75, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_CUP_HANDLE'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'CUP_HANDLE_ERROR'}

    def detect_rectangle(self, highs, lows):
        """Detect rectangle/trading range pattern"""
        try:
            if len(highs) < 20:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            
            # Check for horizontal resistance and support
            resistance = np.mean(recent_highs[recent_highs > np.percentile(recent_highs, 80)])
            support = np.mean(recent_lows[recent_lows < np.percentile(recent_lows, 20)])
            
            resistance_touches = np.sum(np.abs(recent_highs - resistance) < 0.002 * resistance)
            support_touches = np.sum(np.abs(recent_lows - support) < 0.002 * support)
            
            if resistance_touches >= 2 and support_touches >= 2:
                range_ratio = (resistance - support) / support
                if 0.02 < range_ratio < 0.15:  # Reasonable trading range
                    return {'pattern_id': 14, 'pattern_name': 'RECTANGLE', 'confidence': 0.70, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_RECTANGLE'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'RECTANGLE_ERROR'}

    def detect_existing_patterns(self, df):
        """Detect existing patterns from original SimplePatternDetector"""
        try:
            highs = df['high'].values[-20:]
            lows = df['low'].values[-20:]
            closes = df['close'].values[-20:]
            
            # HEAD & SHOULDERS
            if len(highs) >= 5:
                mid_idx = len(highs) // 2
                if mid_idx >= 2 and mid_idx + 2 < len(highs):
                    left_shoulder = highs[mid_idx-2]
                    head = highs[mid_idx]
                    right_shoulder = highs[mid_idx+2]
                    
                    if head > left_shoulder and head > right_shoulder:
                        if abs(left_shoulder - right_shoulder) / left_shoulder < 0.02:
                            return {'pattern_id': 1, 'pattern_name': 'HEAD_SHOULDERS', 'confidence': 0.75, 'method': 'RULE_BASED'}
            
            # DOUBLE TOP
            peaks = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.01:
                    return {'pattern_id': 2, 'pattern_name': 'DOUBLE_TOP', 'confidence': 0.70, 'method': 'RULE_BASED'}
            
            # ASCENDING TRIANGLE
            resistance = np.max(highs[-15:])
            support_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            
            if support_trend > 0.001:  # Ascending support
                resistance_touches = np.sum(np.abs(highs - resistance) < 0.001 * resistance)
                if resistance_touches >= 2:
                    return {'pattern_id': 4, 'pattern_name': 'ASCENDING_TRIANGLE', 'confidence': 0.65, 'method': 'RULE_BASED'}
            
            # BULL FLAG
            if len(closes) >= 10:
                slope = (closes[-1] - closes[0]) / len(closes)
                if slope > 0:
                    return {'pattern_id': 5, 'pattern_name': 'BULL_FLAG', 'confidence': 0.60, 'method': 'TREND_ANALYSIS'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_EXISTING_PATTERN'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'EXISTING_ERROR'}

    def detect_double_bottom(self, lows, closes):
        """Detect double bottom pattern"""
        try:
            if len(lows) < 20:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Find valleys (local minima)
            valleys = []
            for i in range(1, len(lows)-1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    valleys.append((i, lows[i]))
            
            if len(valleys) >= 2:
                last_two_valleys = valleys[-2:]
                # Check if valleys are at similar levels
                if abs(last_two_valleys[0][1] - last_two_valleys[1][1]) / last_two_valleys[0][1] < 0.02:
                    # Check if there's recovery between valleys
                    between_idx_start = last_two_valleys[0][0]
                    between_idx_end = last_two_valleys[1][0]
                    peak_between = max(lows[between_idx_start:between_idx_end+1])
                    if peak_between > last_two_valleys[0][1] * 1.02:
                        return {'pattern_id': 3, 'pattern_name': 'DOUBLE_BOTTOM', 'confidence': 0.75, 'method': 'RULE_BASED'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_DOUBLE_BOTTOM'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'DOUBLE_BOTTOM_ERROR'}

    def detect_inverse_head_shoulders(self, lows, closes):
        """Detect inverse head and shoulders pattern"""
        try:
            if len(lows) < 15:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Find the lowest point (head) and shoulders
            mid_idx = len(lows) // 2
            if mid_idx >= 2 and mid_idx + 2 < len(lows):
                left_shoulder = lows[mid_idx-2]
                head = lows[mid_idx]
                right_shoulder = lows[mid_idx+2]
                
                # Head should be lower than both shoulders
                if head < left_shoulder and head < right_shoulder:
                    # Shoulders should be at similar levels
                    if abs(left_shoulder - right_shoulder) / left_shoulder < 0.03:
                        return {'pattern_id': 13, 'pattern_name': 'INVERSE_HEAD_SHOULDERS', 'confidence': 0.75, 'method': 'RULE_BASED'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_INV_HEAD_SHOULDERS'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'INV_HEAD_SHOULDERS_ERROR'}

    def detect_diamond_pattern(self, highs, lows):
        """Detect diamond pattern"""
        try:
            if len(highs) < 25:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Diamond pattern: expanding then contracting volatility
            first_quarter = highs[:len(highs)//4]
            second_quarter = highs[len(highs)//4:len(highs)//2]
            third_quarter = highs[len(highs)//2:3*len(highs)//4]
            fourth_quarter = highs[3*len(highs)//4:]
            
            # Calculate volatility for each quarter
            vol1 = np.std(first_quarter)
            vol2 = np.std(second_quarter)
            vol3 = np.std(third_quarter)
            vol4 = np.std(fourth_quarter)
            
            # Check for expansion then contraction
            if vol2 > vol1 and vol3 > vol2 and vol4 < vol3 and vol4 < vol2:
                return {'pattern_id': 15, 'pattern_name': 'DIAMOND', 'confidence': 0.65, 'method': 'VOLATILITY_ANALYSIS'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_DIAMOND'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'DIAMOND_ERROR'}

    def detect_pennant_pattern(self, highs, lows, closes):
        """Detect pennant pattern"""
        try:
            if len(closes) < 20:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Check for strong move (flagpole)
            flagpole_start = closes[0]
            flagpole_end = closes[len(closes)//2]
            flagpole_move = abs(flagpole_end - flagpole_start) / flagpole_start
            
            if flagpole_move > 0.05:  # At least 5% move
                # Check for converging pennant
                pennant_highs = highs[len(highs)//2:]
                pennant_lows = lows[len(lows)//2:]
                
                high_slope = (pennant_highs[-1] - pennant_highs[0]) / len(pennant_highs)
                low_slope = (pennant_lows[-1] - pennant_lows[0]) / len(pennant_lows)
                
                # Converging slopes
                if (high_slope < 0 and low_slope > 0) or (high_slope > 0 and low_slope < 0):
                    convergence = abs(pennant_highs[-1] - pennant_lows[-1]) < abs(pennant_highs[0] - pennant_lows[0]) * 0.6
                    if convergence:
                        return {'pattern_id': 9, 'pattern_name': 'PENNANT', 'confidence': 0.70, 'method': 'CHART_PATTERN'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_PENNANT'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'PENNANT_ERROR'}

    def detect_hanging_man(self, candle):
        """Detect hanging man pattern (similar to hammer but bearish context)"""
        try:
            open_price = float(candle['open'])
            high_price = float(candle['high'])
            low_price = float(candle['low'])
            close_price = float(candle['close'])
            
            body_size = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            candle_range = high_price - low_price
            
            if candle_range == 0:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INVALID_CANDLE'}
            
            body_ratio = body_size / candle_range
            upper_ratio = upper_shadow / candle_range
            lower_ratio = lower_shadow / candle_range
            
            # Small body at top, long lower shadow (bearish context)
            if body_ratio < 0.3 and lower_ratio > 0.6 and upper_ratio < 0.1 and close_price < open_price:
                return {'pattern_id': 18, 'pattern_name': 'HANGING_MAN', 'confidence': 0.75, 'method': 'SINGLE_CANDLESTICK'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_HANGING_MAN'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'HANGING_MAN_ERROR'}

    def detect_tweezer_patterns(self, candles):
        """Detect tweezer top and bottom patterns"""
        try:
            if len(candles) < 2:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_CANDLES'}
            
            first = candles.iloc[0]
            second = candles.iloc[1]
            
            # TWEEZER TOP - Similar highs at resistance
            if abs(first['high'] - second['high']) / first['high'] < 0.005:
                if first['high'] > first['close'] and second['high'] > second['close']:
                    return {'pattern_id': 33, 'pattern_name': 'TWEEZER_TOP', 'confidence': 0.70, 'method': 'TWO_CANDLESTICK'}
            
            # TWEEZER BOTTOM - Similar lows at support
            if abs(first['low'] - second['low']) / first['low'] < 0.005:
                if first['low'] < first['close'] and second['low'] < second['close']:
                    return {'pattern_id': 34, 'pattern_name': 'TWEEZER_BOTTOM', 'confidence': 0.70, 'method': 'TWO_CANDLESTICK'}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_TWEEZER'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'TWEEZER_ERROR'}

    def get_pattern_info(self, pattern_name):
        """Get detailed information about a pattern"""
        pattern_info = {
            'HEAD_SHOULDERS': {'type': 'reversal', 'direction': 'bearish', 'reliability': 'high'},
            'DOUBLE_TOP': {'type': 'reversal', 'direction': 'bearish', 'reliability': 'medium'},
            'DOUBLE_BOTTOM': {'type': 'reversal', 'direction': 'bullish', 'reliability': 'medium'},
            'ASCENDING_TRIANGLE': {'type': 'continuation', 'direction': 'bullish', 'reliability': 'medium'},
            'DESCENDING_TRIANGLE': {'type': 'continuation', 'direction': 'bearish', 'reliability': 'medium'},
            'SYMMETRICAL_TRIANGLE': {'type': 'neutral', 'direction': 'breakout', 'reliability': 'medium'},
            'BULL_FLAG': {'type': 'continuation', 'direction': 'bullish', 'reliability': 'high'},
            'BEAR_FLAG': {'type': 'continuation', 'direction': 'bearish', 'reliability': 'high'},
            'WEDGE_RISING': {'type': 'reversal', 'direction': 'bearish', 'reliability': 'medium'},
            'WEDGE_FALLING': {'type': 'reversal', 'direction': 'bullish', 'reliability': 'medium'},
            'CUP_AND_HANDLE': {'type': 'continuation', 'direction': 'bullish', 'reliability': 'high'},
            'RECTANGLE': {'type': 'neutral', 'direction': 'breakout', 'reliability': 'low'},
            'DOJI': {'type': 'reversal', 'direction': 'neutral', 'reliability': 'medium'},
            'HAMMER': {'type': 'reversal', 'direction': 'bullish', 'reliability': 'medium'},
            'SHOOTING_STAR': {'type': 'reversal', 'direction': 'bearish', 'reliability': 'medium'},
            'ENGULFING_BULLISH': {'type': 'reversal', 'direction': 'bullish', 'reliability': 'high'},
            'ENGULFING_BEARISH': {'type': 'reversal', 'direction': 'bearish', 'reliability': 'high'},
            'MORNING_STAR': {'type': 'reversal', 'direction': 'bullish', 'reliability': 'high'},
            'EVENING_STAR': {'type': 'reversal', 'direction': 'bearish', 'reliability': 'high'}
        }
        return pattern_info.get(pattern_name, {'type': 'unknown', 'direction': 'unknown', 'reliability': 'unknown'})

    def validate_pattern(self, pattern_result, df):
        """Validate pattern with additional checks"""
        try:
            if pattern_result['pattern_name'] == 'NO_PATTERN':
                return pattern_result
            
            # Add volume confirmation if available
            if 'volume' in df.columns:
                recent_volume = df['volume'].tail(5).mean()
                avg_volume = df['volume'].tail(20).mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                # Adjust confidence based on volume
                if volume_ratio > 1.2:
                    pattern_result['confidence'] = min(0.95, pattern_result['confidence'] * 1.1)
                elif volume_ratio < 0.8:
                    pattern_result['confidence'] = max(0.3, pattern_result['confidence'] * 0.9)
            
            # Add trend context
            closes = df['close'].tail(20).values
            trend_slope = (closes[-1] - closes[0]) / len(closes)
            pattern_result['trend_context'] = 'uptrend' if trend_slope > 0 else 'downtrend'
            
            return pattern_result
            
        except Exception as e:
            pattern_result['validation_error'] = str(e)
            return pattern_result
            

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
                    # ใช้ shared data
                    shared_df = get_shared_xau_data()
                    if shared_df is not None:
                        result = run_ai_once_shared(shared_df)
                        send_status = send_telegram(result)
                        print(f"✅ [{current_time}] Original AI sent to Telegram: Status {send_status}")
                        print(f"Original message preview: {result[:150]}...")
                    else:
                        error_msg = f"❌ Original AI Data Error @ {current_time}\nCannot fetch market data"
                        send_telegram(error_msg)
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

# แก้ไขฟังก์ชัน run_pattern_bot ใน Flask route
@app.route('/run-pattern-bot')
def run_pattern_bot():
    """Run pattern AI system - Send Telegram once per hour - Fixed Version"""
    global last_pattern_sent_hour, message_sent_this_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        current_time = now_th.strftime("%Y-%m-%d %H:%M")
        
        # Reset message tracking เมื่อเปลี่ยนชั่วโมง
        if current_hour != last_pattern_sent_hour:
            last_pattern_sent_hour = current_hour
            message_sent_this_hour['pattern'] = None
        
        # ตรวจสอบว่าส่งข้อความในชั่วโมงนี้แล้วหรือยัง
        if message_sent_this_hour['pattern'] != current_hour:
            message_sent_this_hour['pattern'] = current_hour
            
            def send_pattern_task():
                try:
                    shared_df = get_shared_xau_data()
                    if shared_df is None:
                        error_msg = f"❌ Pattern AI Data Error @ {current_time}\nCannot fetch market data"
                        send_telegram(error_msg)
                        return
                        
                    if len(shared_df) < 20:
                        error_msg = f"❌ Pattern AI Data Error @ {current_time}\nInsufficient data for analysis"
                        send_telegram(error_msg)
                        return
                    
                    detector = AdvancedPatternDetector()
                    all_patterns = detector.detect_all_patterns(shared_df.tail(50))
                    
                    # กรอง patterns ที่มีคุณภาพ
                    quality_patterns = [p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN' and p['confidence'] > 0.60]
                    
                    if len(quality_patterns) > 1:
                        # ใช้ฟังก์ชันใหม่ที่ส่งแบบหลายข้อความ
                        send_status = send_multiple_patterns_message(quality_patterns, shared_df)
                        print(f"✅ [{current_time}] Multiple patterns messages sent: Status {send_status}")
                    else:
                        # ส่งแบบเดิมถ้ามี pattern เดียว
                        result, chart_buffer, pattern_description, pattern_info = run_pattern_ai_shared_with_chart(shared_df)
                        send_status = send_telegram_with_chart(result, chart_buffer)
                        
                        if pattern_description and pattern_description != "ไม่มีข้อมูลแพทเทิร์นนี้":
                            time.sleep(3)
                            send_pattern_theory_explanation(pattern_info['pattern_name'], pattern_description)
                        
                        print(f"✅ [{current_time}] Single pattern message sent: Status {send_status}")
                        
                except Exception as e:
                    print(f"❌ [{current_time}] Pattern AI send error: {e}")
                    error_msg = f"❌ Pattern AI Error @ {current_time}\nError: {str(e)[:100]}"
                    send_telegram(error_msg)
            
            Thread(target=send_pattern_task, daemon=True).start()
            
            return jsonify({
                "status": "✅ Pattern AI - Messages Sent", 
                "mode": "TELEGRAM_SENT",
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": True,
                "system": "Advanced Pattern Detection",
                "note": f"🚀 PATTERN signals sent at {current_time}",
                "sent_count_this_hour": 1
            })
        else:
            return jsonify({
                "status": "✅ Pattern AI - Keep Alive",
                "mode": "KEEP_ALIVE", 
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": False,
                "system": "Advanced Pattern Detection",
                "note": f"Pattern signals already sent in hour {current_hour}",
                "next_signal_time": f"{current_hour + 1}:00"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500
    
@app.route('/test-pattern-chart')
def test_pattern_chart():
    """Test pattern AI with chart generation"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is not None:
            result, chart_buffer, pattern_description, pattern_info = run_pattern_ai_shared_with_chart(shared_df)
            
            # ทดสอบส่งกราฟ
            if chart_buffer:
                send_status = send_telegram_with_chart(result, chart_buffer)
                
                # ส่งคำอธิบายแพทเทิร์น
                if pattern_description and pattern_description != "ไม่มีข้อมูลแพทเทิร์นนี้":
                    time.sleep(2)
                    send_telegram(f"📚 รายละเอียดแพทเทิร์น:\n{pattern_description}")
                
                return jsonify({
                    "status": "success",
                    "message": "Pattern chart sent to Telegram",
                    "telegram_status": send_status,
                    "has_chart": True,
                    "has_pattern_description": bool(pattern_description)
                })
            else:
                return jsonify({
                    "status": "warning",
                    "message": "Chart generation failed, sent text only",
                    "has_chart": False
                })
        else:
            return jsonify({
                "status": "error", 
                "message": "Cannot fetch market data"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
        
# แก้ไข Flask route
@app.route('/test-new-format')
def test_new_format():
    """ทดสอบรูปแบบข้อความใหม่ - Fixed Version"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None:
            return jsonify({
                "status": "error",
                "message": "Cannot fetch market data"
            }), 500
        
        if len(shared_df) < 20:
            return jsonify({
                "status": "error", 
                "message": "Insufficient data for pattern detection"
            }), 400
        
        detector = AdvancedPatternDetector()
        all_patterns = detector.detect_all_patterns(shared_df.tail(50))
        
        # กรอง patterns ที่มีคุณภาพ
        quality_patterns = [p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN' and p['confidence'] > 0.60]
        
        if len(quality_patterns) > 1:
            # ส่งข้อความในรูปแบบใหม่
            send_status = send_multiple_patterns_message(quality_patterns, shared_df)
            
            return jsonify({
                "status": "success",
                "message": "New format messages sent to Telegram",
                "patterns_count": len(quality_patterns),
                "patterns": [p['pattern_name'] for p in quality_patterns],
                "telegram_status": send_status,
                "confidence_range": {
                    "min": min([p['confidence'] for p in quality_patterns]),
                    "max": max([p['confidence'] for p in quality_patterns])
                }
            })
        else:
            # ไม่มี patterns เพียงพอ - ใช้รูปแบบเดิม
            result, chart_buffer, pattern_description, pattern_info = run_pattern_ai_shared_with_chart(shared_df)
            send_status = send_telegram_with_chart(result, chart_buffer)
            
            return jsonify({
                "status": "info",
                "message": "Not enough quality patterns for new format, used standard format",
                "patterns_count": len(quality_patterns),
                "fallback_used": True,
                "telegram_status": send_status,
                "main_pattern": quality_patterns[0]['pattern_name'] if quality_patterns else 'NO_PATTERN'
            })
            
    except Exception as e:
        print(f"Test new format error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error testing new format: {str(e)}"
        }), 500

@app.route('/test-theory-diagram')
def test_theory_diagram():
    """Test pattern theory diagram generation"""
    try:
        # Test with different patterns
        test_patterns = ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM', 
                        'ASCENDING_TRIANGLE', 'BULL_FLAG', 'NO_PATTERN']
        
        pattern_name = request.args.get('pattern', 'HEAD_SHOULDERS')
        
        if pattern_name not in test_patterns:
            return jsonify({
                "status": "error",
                "message": f"Invalid pattern. Available: {test_patterns}"
            })
        
        pattern_description = get_pattern_description(pattern_name)
        send_status = send_pattern_theory_explanation(pattern_name, pattern_description)
        
        return jsonify({
            "status": "success",
            "message": f"Theory diagram for {pattern_name} sent to Telegram",
            "telegram_status": send_status,
            "pattern": pattern_name
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
        shared_df = get_shared_xau_data()
        if shared_df is not None:
            result = run_pattern_ai_shared(shared_df)
        else:
            result = "❌ Cannot fetch market data for pattern testing"
            
        return jsonify({
            "status": "success", 
            "message": "Pattern AI test completed",
            "result": result[:500] + "..." if len(result) > 500 else result,
            "data_source": "shared"
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
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 10:
            return jsonify({
                "status": "error", 
                "message": "Cannot fetch sufficient shared data"
            })
        
        detector = AdvancedPatternDetector()
        all_patterns = detector.detect_all_patterns(shared_df.tail(50))
        pattern_info = all_patterns[0]  # ใช้ pattern แรกสำหรับการแสดงผลหลัก
        
        current_price = float(shared_df['close'].iloc[-1])
        current_rsi = float(shared_df['rsi'].iloc[-1]) if not pd.isna(shared_df['rsi'].iloc[-1]) else 50.0
        current_ema10 = float(shared_df['ema'].iloc[-1]) if not pd.isna(shared_df['ema'].iloc[-1]) else current_price
        current_ema21 = float(shared_df['ema_21'].iloc[-1]) if not pd.isna(shared_df['ema_21'].iloc[-1]) else current_price
        
        return jsonify({
            "status": "success",
            "data_source": "shared",
            "current_price": current_price,
            "indicators": {
                "rsi": current_rsi,
                "ema10": current_ema10,
                "ema21": current_ema21
            },
            "pattern_detection": pattern_info,
            "timestamp": datetime.now().isoformat(),
            "data_points": len(shared_df)
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
            
            <h2>Trading Systems</h2>
            <ul>
                <li><strong>Original System:</strong> RSI + EMA + Price Change Analysis</li>
                <li><strong>Pattern AI System:</strong> Chart Pattern Detection + Technical Analysis</li>
            </ul>
            
            <h2>API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong>
                <p>Health check endpoint for monitoring services</p>
                <p><em>Returns:</em> Simple "OK" response</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/run-ai</strong>
                <p><span class="status">ACTIVE MODE:</span> Execute original AI system with Telegram alerts</p>
                <p><em>Frequency:</em> Every 3 minutes (recommended)</p>
                <p><em>Output:</em> Telegram message <strong>once per hour</strong> with RSI+EMA+Price Change signals</p>
                <p><em>Logic:</em> First ping of each hour = send signal, subsequent pings = keep alive</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/run-pattern-bot</strong>
                <p><span class="status">ACTIVE MODE:</span> Execute pattern AI trading system with Telegram alerts</p>
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
            
            <h2>Recommended UptimeRobot Setup</h2>
            <div style="background-color: #2a2a2a; padding: 20px; border-radius: 8px; border-left: 4px solid #00ff88;">
                <h3 style="margin-top: 0; color: #00ff88;">Dual System Strategy:</h3>
                <p><strong>Monitor 1:</strong> <code>/run-ai</code> - Every 3 minutes</p>
                <p style="margin-left: 20px;">→ Sends ORIGINAL system signals <strong>once per hour</strong></p>
                
                <p><strong>Monitor 2:</strong> <code>/run-pattern-bot</code> - Every 3 minutes</p>
                <p style="margin-left: 20px;">→ Sends PATTERN AI signals <strong>once per hour</strong></p>
                
                <p style="color: #ffaa00;"><strong>Result:</strong> <span style="color: #00ff88;">Exactly 2 trading signals per hour</span> via Telegram</p>
                <p style="color: #ffaa00;"><strong>Benefit:</strong> Compare both systems + Service never sleeps + No duplicate messages</p>
                
                <h4 style="color: #00ff88;">Expected Telegram Messages per Hour:</h4>
                <p>🤖 <strong>Original AI Signal</strong> - RSI + EMA + Price Change analysis</p>
                <p>🚀 <strong>Pattern AI Signal</strong> - CNN + RNN + Pattern detection</p>
                <p style="color: #666;">Each system sends exactly once per hour, independent tracking</p>
                
                <h4 style="color: #ffaa00;">How It Works (Shared Data System):</h4>
                <p style="margin-left: 10px;">• Both systems use <strong>identical data source</strong> from single API call</p>
                <p style="margin-left: 10px;">• Same OHLC data, same RSI calculation, same real-time price</p>
                <p style="margin-left: 10px;">• Only analysis methods differ: Original uses ML, Pattern uses rule-based detection</p>
                <p style="margin-left: 10px;">• First ping to each endpoint in each hour = Send respective signal</p>
                <p style="margin-left: 10px;">• Subsequent pings = Keep service alive only</p>
                <p style="color: #00ff88; margin-left: 10px;"><strong>✅ Result: Consistent data, different perspectives</strong></p>
            </div>
            
            <h2>Configuration</h2>
            <p>The bot requires the following environment variables:</p>
            <ul>
                <li><code>BOT_TOKEN</code> - Telegram bot token</li>
                <li><code>CHAT_ID</code> - Telegram chat ID for messages</li>
                <li><code>API_KEY</code> - TwelveData API key for market data</li>
            </ul>
            
            <h2>Usage</h2>
            <p>Use monitoring services like UptimeRobot to ping:</p>
            <ul>
                <li><code>/health</code> for keeping the service alive</li>
                <li><code>/run-ai</code> for original system signals</li>
                <li><code>/run-pattern-bot</code> for pattern-based signals</li>
            </ul>
            
            <h2>Risk Disclaimer</h2>
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
    print(f"Health Check: /health")
    print(f"Original System: /run-ai")
    print(f"Pattern AI: /run-pattern-bot")
    print(f"Test Telegram: /test-telegram")
    print(f"Test Pattern: /test-pattern-ai")
    print(f"Pattern Status: /pattern-status")
    print(f"System Status: /status")
    print("=" * 60)
    print(f"Libraries Available:")
    print(f"   • TensorFlow: {'✅' if HAS_TENSORFLOW else '❌'}")
    print(f"   • Scikit-learn: {'✅' if HAS_SKLEARN else '❌'}")
    print(f"   • TA-Lib: {'✅' if HAS_TA else '❌'}")
    print(f"   • Charts: {'✅' if HAS_CHARTS else '❌'}")
    print("=" * 60)
    print(f"Configuration:")
    print(f"   • Bot Token: {'✅ Configured' if BOT_TOKEN else '❌ Missing'}")
    print(f"   • Chat ID: {'✅ Configured' if CHAT_ID else '❌ Missing'}")
    print(f"   • API Key: {'✅ Configured' if API_KEY else '❌ Missing'}")
    print("=" * 60)
    print("🚀 Ready for AI-powered trading!")
    print("💰 Asset: XAU/USD | Timeframe: 1H")
    print("Monitoring: Configure UptimeRobot with endpoints above")
    print("=" * 60)
    
    # Get port from environment
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=False)
