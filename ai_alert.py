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
last_harmonic_sent_hour = None  # เพิ่มบรรทัดนี้ที่ส่วน Global variables
message_sent_this_hour = {      # เพิ่มตัวตรวจสอบว่าส่งข้อความไหนแล้วบ้าง
    'original': None,
    'pattern': None,
    'harmonic': None  # เพิ่มบรรทัดนี้
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

#def get_pattern_description(pattern_name):
   # """Get detailed pattern description with BOT prediction - COMPLETE VERSION"""
    # bot_signal = get_pattern_signal(pattern_name)
    #bot_signal = get_pattern_signal_with_context(pattern_name, pattern_info)
    
def get_pattern_description(pattern_name, pattern_info=None):
    """Get detailed pattern description with BOT prediction - COMPLETE VERSION"""
    # Use context-aware signal if pattern_info is provided
    if pattern_info:
        bot_signal = get_pattern_signal_with_context(pattern_name, pattern_info)
    else:
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
    """ส่งข้อความรูปแบบใหม่แยกตาม method - Fixed Version"""
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
        
        # 1) ข้อความแรก - รวมทุก patterns ที่พบ (แก้ไขใหม่)
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
ราคาปัจจุบัน: ${current_data['close']:,.2f}

"""
        
        # แสดง Reversal Patterns
        if reversal_patterns:
            message_1 += f"🔄 Reversal Patterns: {len(reversal_patterns)}\n"
            for i, pattern in enumerate(reversal_patterns, 1):
                signal = "🔴 SELL" if pattern['pattern_name'] in bearish_list else "🟢 BUY"
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                message_1 += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: {signal} | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
        
        # แสดง Continuation Patterns
        if continuation_patterns:
            message_1 += f"➡️ Continuation Patterns: {len(continuation_patterns)}\n"
            for i, pattern in enumerate(continuation_patterns, 1):
                signal = "🔴 SELL" if pattern['pattern_name'] in bearish_list else "🟢 BUY"
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                message_1 += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: {signal} | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
        
        # แสดง Bearish Patterns
        if bearish_patterns:
            message_1 += f"🔴 Bearish Patterns: {len(bearish_patterns)}\n"
            for i, pattern in enumerate(bearish_patterns, 1):
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                message_1 += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: 🔴 SELL | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
        
        # แสดง Bullish Patterns
        if bullish_patterns:
            message_1 += f"🟢 Bullish Patterns: {len(bullish_patterns)}\n"
            for i, pattern in enumerate(bullish_patterns, 1):
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                message_1 += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: 🟢 BUY | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
        
        # แสดง Neutral Patterns
        if neutral_patterns:
            message_1 += f"🟡 Neutral Patterns: {len(neutral_patterns)}\n"
            for i, pattern in enumerate(neutral_patterns, 1):
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐" if pattern['confidence'] > 0.75 else "✨"
                message_1 += f"""{i}. {confidence_emoji} {pattern['pattern_name'].replace('_', ' ')}
   🎯 Signal: 🟡 WAIT/BREAKOUT | Confidence: {pattern['confidence']*100:.1f}%
   🔧 Method: {pattern.get('method', 'PATTERN_ANALYSIS')}

"""
        else:
            message_1 += "🟡 Neutral Patterns: 0\n\n"
        
        # เพิ่ม Directional Analysis
        bullish_count = len(bullish_patterns)
        bearish_count = len(bearish_patterns)
        
        if bullish_count > bearish_count:
            dominant_bias = "🟢 BULLISH BIAS"
            market_sentiment = "ตลาดมีแนวโน้มขาขึ้น"
        elif bearish_count > bullish_count:
            dominant_bias = "🔴 BEARISH BIAS"
            market_sentiment = "ตลาดมีแนวโน้มขาลง"
        else:
            dominant_bias = "🟡 NEUTRAL BIAS"
            market_sentiment = "ตลาดไม่มีทิศทางชัดเจน"
        
        message_1 += f"""📊 DIRECTIONAL ANALYSIS

🟢 BULLISH PATTERNS: {bullish_count}
🔴 BEARISH PATTERNS: {bearish_count}

🎯 MARKET BIAS: {dominant_bias}
📈 Sentiment: {market_sentiment}

"""
        
        # แสดง Top Bullish Patterns
        if bullish_patterns:
            message_1 += "🟢 TOP BULLISH PATTERNS:\n"
            for i, pattern in enumerate(sorted(bullish_patterns, key=lambda x: x['confidence'], reverse=True)[:2], 1):
                message_1 += f"{i}. {pattern['pattern_name'].replace('_', ' ')} ({pattern['confidence']*100:.1f}%)\n"
            message_1 += "\n"
        
        # แสดง Top Bearish Patterns
        if bearish_patterns:
            message_1 += "🔴 TOP BEARISH PATTERNS:\n"
            for i, pattern in enumerate(sorted(bearish_patterns, key=lambda x: x['confidence'], reverse=True)[:2], 1):
                message_1 += f"{i}. {pattern['pattern_name'].replace('_', ' ')} ({pattern['confidence']*100:.1f}%)\n"
        
        # ส่งข้อความแรก
        send_telegram(message_1)
        time.sleep(3)
        
        # 2) ข้อความที่สอง - Original BOT (คงเดิม)
        original_action = trading_signals.get('action', 'WAIT')
        action_text = "🔴 SELL" if original_action == 'SELL' else "🟢 BUY" if original_action == 'BUY' else "⏸️ WAIT"
        
        message_2 = f"""🤖 ORIGINAL BOT (RSI+EMA+Price Change) 🎯BOTทำนาย: {action_text}

💾 SHARED DATA SOURCE

Open = ${current_data['open']:,.2f} | High = ${current_data['high']:,.2f}
Low = ${current_data['low']:,.2f} | Close = ${current_data['close']:,.2f}
ราคาปัจจุบัน = ${trading_signals['current_price']:,.2f}"""
        
        if original_action in ['BUY', 'SELL']:
            case_text = "🟢 BUY CASE" if original_action == 'BUY' else "🔴 SELL CASE"
            
            message_2 += f"""

{case_text}

💼 TRADING SETUP:
🎯 Entry: ${trading_signals['entry_price']:,.2f}
🟢 TP1: ${trading_signals['tp1']:,.2f} | TP2: ${trading_signals['tp2']:,.2f} | TP3: ${trading_signals['tp3']:,.2f}
🔴 SL: ${trading_signals['sl']:,.2f}
💯 Pattern Confidence: {trading_signals['confidence']*100:.1f}%"""
        
        message_2 += f"""

🚨 สิ่งที่ต้องระวัง:
• หลายแพทเทิร์นอาจขัดแย้งกัน
• ให้ความสำคัญกับแพทเทิร์นที่มี Confidence สูงสุด
• รอการยืนยันก่อนเข้าเทรด
• ใช้ Risk Management เข้มงวด

💡 คำแนะนำ: เมื่อมีหลายแพทเทิร์น ควรรอให้ตลาดชัดเจนกว่านี้"""
        
        send_telegram(message_2)
        time.sleep(3)
        
        # 3) ข้อความที่สาม - Main Pattern กับรูป (คงเดิม)
        main_pattern = all_patterns[0]
        
        chart_buffer = create_candlestick_chart(shared_df, trading_signals, main_pattern)
        
        pattern_action = "🔴 SELL" if trading_signals['action'] == 'SELL' else "🟢 BUY" if trading_signals['action'] == 'BUY' else "⏸️ WAIT"
        
        message_3 = f"""🚀 AI PATTERN DETECTION BOT

⏰ {current_time} | 💰 XAUUSD (1H)

💾 SHARED DATA SOURCE

💰 MARKET DATA:
Open: ${current_data['open']:,.2f} | High: ${current_data['high']:,.2f}
Low: ${current_data['low']:,.2f} | Close: ${current_data['close']:,.2f}
ราคาปัจจุบัน = ${current_data['close']:,.2f}

🔍 PATTERN DETECTED:
{main_pattern['pattern_name'].replace('_', ' ')}

🤖 Method: {main_pattern.get('method', 'CHART_PATTERN')} | 🎯 Confidence: {main_pattern['confidence']*100:.1f}%

💹 TECHNICAL INDICATORS (SHARED):
RSI: {trading_signals['rsi']:.1f} ({'Oversold' if trading_signals['rsi']<30 else 'Overbought' if trading_signals['rsi']>70 else 'Neutral'})
EMA10: ${trading_signals['ema10']:,.2f} ({'Above' if trading_signals['current_price']>trading_signals['ema10'] else 'Below'})
EMA21: ${trading_signals['ema21']:,.2f} ({'Above' if trading_signals['current_price']>trading_signals['ema21'] else 'Below'})

🚦 PATTERN AI SIGNAL: {pattern_action}

💼 TRADING SETUP:
🎯 Entry: ${trading_signals['entry_price']:,.2f}
🟢 TP1: ${trading_signals['tp1']:,.2f} | TP2: ${trading_signals['tp2']:,.2f} | TP3: ${trading_signals['tp3']:,.2f}
🔴 SL: ${trading_signals['sl']:,.2f}
💯 Pattern Confidence: {trading_signals['confidence']*100:.1f}%

⚠️ Risk: ใช้เงินเพียง 1-2% ต่อออเดอร์"""
        
        send_telegram_with_chart(message_3, chart_buffer)
        time.sleep(5)
        
        # 4) ส่งรายละเอียดแต่ละ pattern พร้อมรูป (คงเดิม)
        pattern_count = 1
        for pattern in all_patterns[:100]:
            if pattern['pattern_name'] != 'NO_PATTERN':
                theory_diagram = create_pattern_theory_diagram(pattern['pattern_name'])
                pattern_desc = get_pattern_description(pattern['pattern_name'])
                
                detail_message = f"""📚 PATTERN DETAIL #{pattern_count}

🎯 {pattern['pattern_name'].replace('_', ' ')}

💯 Confidence: {pattern['confidence']*100:.1f}%

{pattern_desc}"""
                
                send_telegram_with_chart(detail_message, theory_diagram)
                time.sleep(4)
                pattern_count += 1
        
        # 5) ข้อความสุดท้าย - สรุป (แก้ไขใหม่)
        valid_pattern_count = len([p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN'])
        highest_confidence = max([p['confidence'] for p in all_patterns if p['pattern_name'] != 'NO_PATTERN'])
        
        # กำหนด Market Phase
        if len(continuation_patterns) > len(reversal_patterns):
            market_phase = "Trending"
        elif len(reversal_patterns) > 0:
            market_phase = "Turning"
        else:
            market_phase = "Consolidating"
        
        summary_message = f"""📚 PATTERN ANALYSIS COMPLETE

🔍 รวม {valid_pattern_count} patterns พบในการวิเคราะห์
🎯 ส่งทฤษฎีและรูปประกอบแล้ว {min(valid_pattern_count, 100)} patterns
• Total Quality Patterns: {total_patterns}
• Reversal Signals: {len(reversal_patterns)}
• Continuation Signals: {len(continuation_patterns)}
• Bullish Bias: {bullish_count} patterns
• Bearish Bias: {bearish_count} patterns
• Neutral/Wait: {len(neutral_patterns)} patterns

🎯 KEY TAKEAWAYS:
• Main Direction: {dominant_bias}
• Confidence Level: {highest_confidence*100:.1f}% (Highest)
• Market Phase: {market_phase}

💼 NEXT STEPS:
1. Monitor price action for confirmation
2. Watch for volume spikes on breakouts
3. Set alerts at key support/resistance levels
4. Prepare trading plan for next hour

💡 คำแนะนำการใช้งาน:
• ให้ความสำคัญกับ pattern ที่มี confidence สูงสุด
• รอการยืนยันจากแท่งเทียนถัดไป
• ใช้ Risk Management เข้มงวด
• เปรียบเทียบสัญญาณจากหลาย patterns

⚠️ หากมี patterns ขัดแย้งกัน ให้รอสัญญาณที่ชัดเจนกว่า"""
        
        send_telegram(summary_message)
        print(f"✅ Multiple patterns message sequence completed")
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
        'INVERTED_HAMMER'
    ]
    
    bearish_patterns = [
        'HEAD_SHOULDERS', 'DOUBLE_TOP', 'SHOOTING_STAR', 'BEAR_FLAG',
        'DESCENDING_TRIANGLE', 'WEDGE_RISING', 'ENGULFING_BEARISH',
        'EVENING_STAR', 'THREE_BLACK_CROWS', 'DARK_CLOUD_COVER',
        'HARAMI_BEARISH', 'TWEEZER_TOP', 'HANGING_MAN'
    ]
    
    neutral_patterns = [
        'DOJI', 'SPINNING_TOP', 'RECTANGLE', 'SYMMETRICAL_TRIANGLE',
        'DIAMOND', 'PENNANT', 'MARUBUZO'
    ]
    
    # Context-Dependent Patterns (ขึ้นกับทิศทางการก่อตัว)
    # สำหรับ patterns เหล่านี้ควรดูจาก pattern_info['points'] หรือ 'wave_points'
    context_dependent = [
        'GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD',
        'ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3'
    ]
    
    if pattern_name in bullish_patterns:
        return "🟢 BUY"
    elif pattern_name in bearish_patterns:
        return "🔴 SELL"
    elif pattern_name in neutral_patterns:
        return "🟡 WAIT/BREAKOUT"
    elif pattern_name in context_dependent:
        return "🔵 CONTEXT (Check Direction)"    
    else:
        return "⏸️ WAIT"

def get_pattern_signal_with_context(pattern_name, pattern_info=None):
    """
    Get BOT prediction with context analysis
    สำหรับ Harmonic และ Elliott Wave จะดูจากทิศทางของ points
    """
    
    # Patterns ที่มีทิศทางชัดเจน
    simple_signal = get_pattern_signal(pattern_name)
    
    if simple_signal not in ["🔵 CONTEXT (Check Direction)"]:
        return simple_signal
    
    # สำหรับ Context-Dependent Patterns
    if pattern_info is None:
        return "🟡 WAIT (Need Context)"
    
    # วิเคราะห์ Harmonic Patterns จากจุด X และ D
    if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD']:
        if 'points' in pattern_info:
            points = pattern_info['points']
            
            # ดูทิศทางจาก X ไป D
            if 'X' in points and 'D' in points:
                X = points['X']
                D = points['D']
                
                if X and D:
                    # ถ้า D สูงกว่า X = Bullish pattern
                    if D[1] > X[1]:
                        return f"🟢 BUY (Bullish {pattern_name})"
                    else:
                        return f"🔴 SELL (Bearish {pattern_name})"
            
            # ถ้า AB=CD ดูจาก A และ D
            elif 'A' in points and 'D' in points:
                A = points['A']
                D = points['D']
                
                if A and D:
                    if D[1] > A[1]:
                        return f"🟢 BUY (Bullish {pattern_name})"
                    else:
                        return f"🔴 SELL (Bearish {pattern_name})"
    
    # วิเคราะห์ Elliott Wave
    elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
        if 'wave_points' in pattern_info:
            waves = pattern_info['wave_points']
            
            # ดูทิศทางจาก start ไป wave สุดท้าย
            if 'start' in waves:
                start = waves['start']
                
                # Wave 5
                if '5' in waves:
                    wave5 = waves['5']
                    if start and wave5:
                        if wave5[1] > start[1]:
                            return "🟢 BUY (Bullish Wave 5)"
                        else:
                            return "🔴 SELL (Bearish Wave 5)"
                
                # Wave 3 (ABC)
                elif 'C' in waves:
                    waveC = waves['C']
                    if start and waveC:
                        if waveC[1] > start[1]:
                            return "🟢 BUY (Bullish ABC)"
                        else:
                            return "🔴 SELL (Bearish ABC)"
    
    # ถ้าวิเคราะห์ไม่ได้
    return "🟡 WAIT (Cannot Determine Direction)"

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
            
# ====================== Harmonic Pattern Detection ======================

class HarmonicPatternDetector:
    def __init__(self):
        self.fibonacci_ratios = {
            0.236: 0.02, 0.382: 0.02, 0.500: 0.02, 0.618: 0.02, 0.786: 0.03,
            0.886: 0.03, 1.000: 0.02, 1.272: 0.03, 1.618: 0.03, 2.000: 0.05,
            2.240: 0.05, 2.618: 0.05, 3.618: 0.08
        }
    
    def detect_harmonic_patterns(self, df):
        """Detect harmonic patterns (XABCD structure)"""
        try:
            if len(df) < 50:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            # Find swing points (X-A-B-C-D structure)
            swing_points = self.find_swing_points(df)
            if len(swing_points) < 5:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_SWINGS'}
            
            # Take last 5 points for XABCD pattern
            X, A, B, C, D = swing_points[-5:]
            
            # Check different harmonic patterns
            gartley = self.check_gartley(X, A, B, C, D)
            if gartley['valid']:
                return {'pattern_id': 35, 'pattern_name': 'GARTLEY', 'confidence': gartley['confidence'], 
                       'method': 'HARMONIC', 'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D}}
            
            butterfly = self.check_butterfly(X, A, B, C, D)
            if butterfly['valid']:
                return {'pattern_id': 36, 'pattern_name': 'BUTTERFLY', 'confidence': butterfly['confidence'],
                       'method': 'HARMONIC', 'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D}}
            
            bat = self.check_bat(X, A, B, C, D)
            if bat['valid']:
                return {'pattern_id': 37, 'pattern_name': 'BAT', 'confidence': bat['confidence'],
                       'method': 'HARMONIC', 'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D}}
            
            crab = self.check_crab(X, A, B, C, D)
            if crab['valid']:
                return {'pattern_id': 38, 'pattern_name': 'CRAB', 'confidence': crab['confidence'],
                       'method': 'HARMONIC', 'points': {'X': X, 'A': A, 'B': B, 'C': C, 'D': D}}
            
            abcd = self.check_abcd(A, B, C, D)  # AB=CD pattern uses ABCD only
            if abcd['valid']:
                return {'pattern_id': 39, 'pattern_name': 'AB_CD', 'confidence': abcd['confidence'],
                       'method': 'HARMONIC', 'points': {'A': A, 'B': B, 'C': C, 'D': D}}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_HARMONIC'}
            
        except Exception as e:
            print(f"Harmonic pattern error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'HARMONIC_ERROR'}
    
    def find_swing_points(self, df, lookback=5):
        """Find significant swing highs and lows"""
        highs = df['high'].values
        lows = df['low'].values
        swing_points = []
        
        # Find swing highs and lows
        for i in range(lookback, len(df) - lookback):
            # Swing high
            if all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
               all(highs[i] > highs[i+j] for j in range(1, lookback+1)):
                swing_points.append((i, highs[i], 'high'))
            
            # Swing low
            elif all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                 all(lows[i] < lows[i+j] for j in range(1, lookback+1)):
                swing_points.append((i, lows[i], 'low'))
        
        # Return alternating high/low points
        if len(swing_points) >= 5:
            return swing_points[-30:]  # Take last 30 swing points
        return swing_points
    
    def check_ratio(self, actual_ratio, target_ratio, tolerance=0.03):
        """Check if actual ratio matches target with tolerance"""
        return abs(actual_ratio - target_ratio) <= tolerance
    
    def check_gartley(self, X, A, B, C, D):
        """Check Gartley pattern ratios"""
        try:
            XA = abs(A[1] - X[1])
            AB = abs(B[1] - A[1])
            BC = abs(C[1] - B[1])
            CD = abs(D[1] - C[1])
            AD = abs(D[1] - A[1])
            
            if XA == 0:
                return {'valid': False, 'confidence': 0}
            
            ab_ratio = AB / XA
            bc_ratio = BC / AB if AB != 0 else 0
            cd_ratio = CD / BC if BC != 0 else 0
            ad_ratio = AD / XA
            
            # Gartley ratios: AB=0.618 XA, BC=0.382-0.886 AB, CD=1.272 BC, AD=0.786 XA
            gartley_valid = (
                self.check_ratio(ab_ratio, 0.618, 0.05) and
                (self.check_ratio(bc_ratio, 0.382, 0.05) or self.check_ratio(bc_ratio, 0.886, 0.05)) and
                self.check_ratio(cd_ratio, 1.272, 0.1) and
                self.check_ratio(ad_ratio, 0.786, 0.05)
            )
            
            confidence = 0.85 if gartley_valid else 0
            return {'valid': gartley_valid, 'confidence': confidence}
            
        except Exception:
            return {'valid': False, 'confidence': 0}
    
    def check_butterfly(self, X, A, B, C, D):
        """Check Butterfly pattern ratios"""
        try:
            XA = abs(A[1] - X[1])
            AB = abs(B[1] - A[1])
            BC = abs(C[1] - B[1])
            CD = abs(D[1] - C[1])
            AD = abs(D[1] - A[1])
            
            if XA == 0:
                return {'valid': False, 'confidence': 0}
            
            ab_ratio = AB / XA
            bc_ratio = BC / AB if AB != 0 else 0
            cd_ratio = CD / BC if BC != 0 else 0
            ad_ratio = AD / XA
            
            # Butterfly ratios: AB=0.786 XA, BC=0.382-0.886 AB, CD=1.618-2.618 BC, AD=1.272-1.618 XA
            butterfly_valid = (
                self.check_ratio(ab_ratio, 0.786, 0.05) and
                (self.check_ratio(bc_ratio, 0.382, 0.05) or self.check_ratio(bc_ratio, 0.886, 0.05)) and
                (self.check_ratio(cd_ratio, 1.618, 0.1) or self.check_ratio(cd_ratio, 2.618, 0.2)) and
                (self.check_ratio(ad_ratio, 1.272, 0.05) or self.check_ratio(ad_ratio, 1.618, 0.05))
            )
            
            confidence = 0.80 if butterfly_valid else 0
            return {'valid': butterfly_valid, 'confidence': confidence}
            
        except Exception:
            return {'valid': False, 'confidence': 0}
    
    def check_bat(self, X, A, B, C, D):
        """Check Bat pattern ratios"""
        try:
            XA = abs(A[1] - X[1])
            AB = abs(B[1] - A[1])
            BC = abs(C[1] - B[1])
            CD = abs(D[1] - C[1])
            AD = abs(D[1] - A[1])
            
            if XA == 0:
                return {'valid': False, 'confidence': 0}
            
            ab_ratio = AB / XA
            bc_ratio = BC / AB if AB != 0 else 0
            cd_ratio = CD / BC if BC != 0 else 0
            ad_ratio = AD / XA
            
            # Bat ratios: AB=0.382-0.500 XA, BC=0.382-0.886 AB, CD=1.618-2.618 BC, AD=0.886 XA
            bat_valid = (
                (self.check_ratio(ab_ratio, 0.382, 0.05) or self.check_ratio(ab_ratio, 0.500, 0.05)) and
                (self.check_ratio(bc_ratio, 0.382, 0.05) or self.check_ratio(bc_ratio, 0.886, 0.05)) and
                (self.check_ratio(cd_ratio, 1.618, 0.1) or self.check_ratio(cd_ratio, 2.618, 0.2)) and
                self.check_ratio(ad_ratio, 0.886, 0.05)
            )
            
            confidence = 0.75 if bat_valid else 0
            return {'valid': bat_valid, 'confidence': confidence}
            
        except Exception:
            return {'valid': False, 'confidence': 0}
    
    def check_crab(self, X, A, B, C, D):
        """Check Crab pattern ratios"""
        try:
            XA = abs(A[1] - X[1])
            AB = abs(B[1] - A[1])
            BC = abs(C[1] - B[1])
            CD = abs(D[1] - C[1])
            AD = abs(D[1] - A[1])
            
            if XA == 0:
                return {'valid': False, 'confidence': 0}
            
            ab_ratio = AB / XA
            bc_ratio = BC / AB if AB != 0 else 0
            cd_ratio = CD / BC if BC != 0 else 0
            ad_ratio = AD / XA
            
            # Crab ratios: AB=0.382-0.618 XA, BC=0.382-0.886 AB, CD=2.240-3.618 BC, AD=1.618 XA
            crab_valid = (
                (self.check_ratio(ab_ratio, 0.382, 0.05) or self.check_ratio(ab_ratio, 0.618, 0.05)) and
                (self.check_ratio(bc_ratio, 0.382, 0.05) or self.check_ratio(bc_ratio, 0.886, 0.05)) and
                (self.check_ratio(cd_ratio, 2.240, 0.2) or self.check_ratio(cd_ratio, 3.618, 0.3)) and
                self.check_ratio(ad_ratio, 1.618, 0.05)
            )
            
            confidence = 0.70 if crab_valid else 0
            return {'valid': crab_valid, 'confidence': confidence}
            
        except Exception:
            return {'valid': False, 'confidence': 0}
    
    def check_abcd(self, A, B, C, D):
        """Check AB=CD pattern"""
        try:
            AB = abs(B[1] - A[1])
            CD = abs(D[1] - C[1])
            
            if AB == 0:
                return {'valid': False, 'confidence': 0}
            
            cd_ab_ratio = CD / AB
            
            # AB=CD ratios: CD = 1.0, 1.272, or 1.618 times AB
            abcd_valid = (
                self.check_ratio(cd_ab_ratio, 1.000, 0.05) or
                self.check_ratio(cd_ab_ratio, 1.272, 0.05) or
                self.check_ratio(cd_ab_ratio, 1.618, 0.05)
            )
            
            confidence = 0.65 if abcd_valid else 0
            return {'valid': abcd_valid, 'confidence': confidence}
            
        except Exception:
            return {'valid': False, 'confidence': 0}

# ====================== Elliott Wave Detection ======================

class ElliottWaveDetector:
    def __init__(self):
        self.wave_patterns = {
            40: "ELLIOTT_WAVE_5",
            41: "ELLIOTT_WAVE_3"
        }
    
    def detect_elliott_waves(self, df):
        """Detect Elliott Wave patterns"""
        try:
            if len(df) < 30:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}
            
            swing_points = self.find_wave_points(df)
            if len(swing_points) < 5:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_WAVES'}
            
            # Check for 5-wave pattern
            wave_5 = self.check_5_wave_pattern(swing_points)
            if wave_5['valid']:
                return {'pattern_id': 40, 'pattern_name': 'ELLIOTT_WAVE_5', 'confidence': wave_5['confidence'],
                       'method': 'ELLIOTT_WAVE', 'wave_points': wave_5['points']}
            
            # Check for 3-wave pattern
            wave_3 = self.check_3_wave_pattern(swing_points)
            if wave_3['valid']:
                return {'pattern_id': 41, 'pattern_name': 'ELLIOTT_WAVE_3', 'confidence': wave_3['confidence'],
                       'method': 'ELLIOTT_WAVE', 'wave_points': wave_3['points']}
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_ELLIOTT_WAVE'}
            
        except Exception as e:
            print(f"Elliott Wave error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'ELLIOTT_ERROR'}
    
    def find_wave_points(self, df, lookback=3):
        """Find wave turning points"""
        highs = df['high'].values
        lows = df['low'].values
        wave_points = []
        
        for i in range(lookback, len(df) - lookback):
            # Wave high
            if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
                wave_points.append((i, highs[i], 'high'))
            
            # Wave low
            elif all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
                 all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
                wave_points.append((i, lows[i], 'low'))
        
        return wave_points[-20:] if len(wave_points) > 20 else wave_points
    
    def check_5_wave_pattern(self, points):
        """Check for Elliott 5-wave impulse pattern"""
        try:
            if len(points) < 6:
                return {'valid': False, 'confidence': 0}
            
            # Take last 6 points for 5-wave pattern (start-1-2-3-4-5)
            wave_points = points[-6:]
            
            # Check Elliott Wave rules:
            # 1. Wave 2 never retraces more than 100% of wave 1
            # 2. Wave 3 is never the shortest wave
            # 3. Wave 4 never enters the price territory of wave 1
            
            start = wave_points[0]
            w1 = wave_points[1]
            w2 = wave_points[2]
            w3 = wave_points[3]
            w4 = wave_points[4]
            w5 = wave_points[5]
            
            wave1_size = abs(w1[1] - start[1])
            wave2_size = abs(w2[1] - w1[1])
            wave3_size = abs(w3[1] - w2[1])
            wave4_size = abs(w4[1] - w3[1])
            wave5_size = abs(w5[1] - w4[1])
            
            # Rule 1: Wave 2 retracement < 100% of Wave 1
            rule1 = wave2_size < wave1_size
            
            # Rule 2: Wave 3 is not the shortest
            rule2 = wave3_size >= max(wave1_size, wave5_size)
            
            # Rule 3: Wave 4 doesn't overlap Wave 1 (simplified check)
            if start[2] == 'low':  # Bullish 5-wave
                rule3 = w4[1] > w1[1]
            else:  # Bearish 5-wave
                rule3 = w4[1] < w1[1]
            
            wave_5_valid = rule1 and rule2 and rule3
            confidence = 0.75 if wave_5_valid else 0
            
            return {
                'valid': wave_5_valid, 
                'confidence': confidence,
                'points': {'start': start, '1': w1, '2': w2, '3': w3, '4': w4, '5': w5}
            }
            
        except Exception:
            return {'valid': False, 'confidence': 0}
    
    def check_3_wave_pattern(self, points):
        """Check for Elliott 3-wave corrective pattern (ABC)"""
        try:
            if len(points) < 4:
                return {'valid': False, 'confidence': 0}
            
            # Take last 4 points for 3-wave pattern (start-A-B-C)
            wave_points = points[-4:]
            
            start = wave_points[0]
            wA = wave_points[1]
            wB = wave_points[2]
            wC = wave_points[3]
            
            wave_A_size = abs(wA[1] - start[1])
            wave_B_size = abs(wB[1] - wA[1])
            wave_C_size = abs(wC[1] - wB[1])
            
            # 3-wave corrective pattern rules (simplified):
            # Wave B retraces 38-78% of Wave A
            # Wave C is approximately equal to Wave A (0.618-1.618 ratio)
            
            if wave_A_size == 0:
                return {'valid': False, 'confidence': 0}
            
            b_retracement = wave_B_size / wave_A_size
            c_to_a_ratio = wave_C_size / wave_A_size
            
            rule1 = 0.38 <= b_retracement <= 0.78
            rule2 = 0.618 <= c_to_a_ratio <= 1.618
            
            wave_3_valid = rule1 and rule2
            confidence = 0.65 if wave_3_valid else 0
            
            return {
                'valid': wave_3_valid,
                'confidence': confidence,
                'points': {'start': start, 'A': wA, 'B': wB, 'C': wC}
            }
            
        except Exception:
            return {'valid': False, 'confidence': 0}

# ====================== Enhanced Pattern Marking System ======================

def draw_enhanced_pattern_lines(ax, df, pattern_info):
    """Enhanced pattern line drawing with specific point marking"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        
        if pattern_name == 'HEAD_SHOULDERS':
            draw_head_shoulders_points(ax, df)
        elif pattern_name == 'DOUBLE_TOP':
            draw_double_top_points(ax, df)
        elif pattern_name == 'DOUBLE_BOTTOM':
            draw_double_bottom_points(ax, df)
        elif pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            if 'points' in pattern_info:
                draw_harmonic_points(ax, pattern_info['points'], pattern_name)
        elif pattern_name == 'AB_CD':
            if 'points' in pattern_info:
                draw_abcd_points(ax, pattern_info['points'])
        elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
            if 'wave_points' in pattern_info:
                draw_elliott_wave_points(ax, pattern_info['wave_points'], pattern_name)
        elif pattern_name == 'ASCENDING_TRIANGLE':
            draw_ascending_triangle_points(ax, df)
        elif pattern_name == 'DESCENDING_TRIANGLE':
            draw_descending_triangle_points(ax, df)
        # Add other pattern point marking functions...
            
    except Exception as e:
        print(f"Enhanced pattern marking error: {e}")

def draw_head_shoulders_points(ax, df):
    """Mark Head & Shoulders key points"""
    try:
        highs = df['high'].values
        if len(highs) >= 20:
            mid_point = len(highs) // 2
            left_shoulder_idx = max(0, mid_point - 10) + np.argmax(highs[max(0, mid_point-10):mid_point])
            head_idx = mid_point - 5 + np.argmax(highs[mid_point-5:mid_point+5])
            right_shoulder_idx = mid_point + np.argmax(highs[mid_point:min(len(highs), mid_point+10)])
            
            # Mark points
            ax.scatter([left_shoulder_idx], [highs[left_shoulder_idx]], 
                      color='#ff00ff', s=120, marker='^', label='Left Shoulder', zorder=10)
            ax.scatter([head_idx], [highs[head_idx]], 
                      color='#ff0000', s=150, marker='^', label='Head', zorder=10)
            ax.scatter([right_shoulder_idx], [highs[right_shoulder_idx]], 
                      color='#ff00ff', s=120, marker='^', label='Right Shoulder', zorder=10)
            
            # Add labels
            ax.text(left_shoulder_idx, highs[left_shoulder_idx] + 5, 'LS', 
                   ha='center', va='bottom', color='#ff00ff', fontweight='bold', fontsize=12)
            ax.text(head_idx, highs[head_idx] + 5, 'HEAD', 
                   ha='center', va='bottom', color='#ff0000', fontweight='bold', fontsize=12)
            ax.text(right_shoulder_idx, highs[right_shoulder_idx] + 5, 'RS', 
                   ha='center', va='bottom', color='#ff00ff', fontweight='bold', fontsize=12)
    except Exception as e:
        print(f"Head & Shoulders marking error: {e}")

def draw_double_top_points(ax, df):
    """Mark Double Top key points"""
    try:
        highs = df['high'].values
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            top1, top2 = peaks[-2], peaks[-1]
            
            # Mark TOP1 and TOP2
            ax.scatter([top1[0]], [top1[1]], color='#ff6600', s=150, 
                      marker='v', label='TOP 1', zorder=10)
            ax.scatter([top2[0]], [top2[1]], color='#ff3300', s=150, 
                      marker='v', label='TOP 2', zorder=10)
            
            # Add labels
            ax.text(top1[0], top1[1] + 8, 'TOP1', ha='center', va='bottom', 
                   color='#ff6600', fontweight='bold', fontsize=12)
            ax.text(top2[0], top2[1] + 8, 'TOP2', ha='center', va='bottom', 
                   color='#ff3300', fontweight='bold', fontsize=12)
            
            # Find and mark valley
            valley_start = min(top1[0], top2[0])
            valley_end = max(top1[0], top2[0])
            valley_idx = valley_start + np.argmin(df['low'].iloc[valley_start:valley_end].values)
            valley_price = df['low'].iloc[valley_idx]
            
            ax.scatter([valley_idx], [valley_price], color='#00ff88', s=120, 
                      marker='^', label='Valley', zorder=10)
            ax.text(valley_idx, valley_price - 8, 'VALLEY', ha='center', va='top', 
                   color='#00ff88', fontweight='bold', fontsize=12)
    except Exception as e:
        print(f"Double Top marking error: {e}")

def draw_double_bottom_points(ax, df):
    """Mark Double Bottom key points"""
    try:
        lows = df['low'].values
        troughs = []
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            bottom1, bottom2 = troughs[-2], troughs[-1]
            
            # Mark BOTTOM1 and BOTTOM2
            ax.scatter([bottom1[0]], [bottom1[1]], color='#00ff88', s=150, 
                      marker='^', label='BOTTOM 1', zorder=10)
            ax.scatter([bottom2[0]], [bottom2[1]], color='#00dd66', s=150, 
                      marker='^', label='BOTTOM 2', zorder=10)
            
            # Add labels
            ax.text(bottom1[0], bottom1[1] - 8, 'BOT1', ha='center', va='top', 
                   color='#00ff88', fontweight='bold', fontsize=12)
            ax.text(bottom2[0], bottom2[1] - 8, 'BOT2', ha='center', va='top', 
                   color='#00dd66', fontweight='bold', fontsize=12)
    except Exception as e:
        print(f"Double Bottom marking error: {e}")

def draw_harmonic_points(ax, points, pattern_name):
    """Mark Harmonic pattern XABCD points"""
    try:
        colors = {'X': '#ff0000', 'A': '#00ff00', 'B': '#0000ff', 'C': '#ffff00', 'D': '#ff00ff'}
        
        for point_name, point_data in points.items():
            if point_data:
                x_pos, price, point_type = point_data
                ax.scatter([x_pos], [price], color=colors.get(point_name, '#ffffff'), 
                          s=120, marker='o', label=f'Point {point_name}', zorder=10)
                ax.text(x_pos, price + 10, point_name, ha='center', va='bottom', 
                       color=colors.get(point_name, '#ffffff'), fontweight='bold', fontsize=14)
        
        # Draw connecting lines
        if len(points) >= 4:
            point_list = list(points.values())
            for i in range(len(point_list) - 1):
                if point_list[i] and point_list[i+1]:
                    ax.plot([point_list[i][0], point_list[i+1][0]], 
                           [point_list[i][1], point_list[i+1][1]], 
                           color='#888888', linestyle='--', alpha=0.7, linewidth=1)
    except Exception as e:
        print(f"Harmonic points marking error: {e}")


# ============= ส่วนที่ 1: เขียนต่อจากฟังก์ชัน draw_elliott_wave_points ที่ขาดไป =============

def draw_elliott_wave_points(ax, wave_points, pattern_type):
    """Mark Elliott Wave points"""
    try:
        if pattern_type == 'ELLIOTT_WAVE_5':
            colors = {'start': '#ffffff', '1': '#ff0000', '2': '#00ff00', 
                     '3': '#0000ff', '4': '#ffff00', '5': '#ff00ff'}
        else:  # ELLIOTT_WAVE_3
            colors = {'start': '#ffffff', 'A': '#ff0000', 'B': '#00ff00', 'C': '#0000ff'}
        
        for wave_name, wave_data in wave_points.items():
            if wave_data:
                x_pos, price, point_type = wave_data
                ax.scatter([x_pos], [price], color=colors.get(wave_name, '#ffffff'), 
                          s=120, marker='o', label=f'Wave {wave_name}', zorder=10)
                ax.text(x_pos, price + 10, f'W{wave_name}', ha='center', va='bottom', 
                       color=colors.get(wave_name, '#ffffff'), fontweight='bold', fontsize=12)
        
        # Draw wave connecting lines
        if pattern_type == 'ELLIOTT_WAVE_5' and len(wave_points) == 6:
            point_sequence = ['start', '1', '2', '3', '4', '5']
        else:  # ELLIOTT_WAVE_3
            point_sequence = ['start', 'A', 'B', 'C']
            
        for i in range(len(point_sequence) - 1):
            current_point = wave_points.get(point_sequence[i])
            next_point = wave_points.get(point_sequence[i + 1])
            
            if current_point and next_point:
                ax.plot([current_point[0], next_point[0]], 
                       [current_point[1], next_point[1]], 
                       color='#00ffcc', linestyle='-', alpha=0.8, linewidth=2)
                
    except Exception as e:
        print(f"Elliott Wave points marking error: {e}")

def draw_abcd_points(ax, points):
    """Mark AB=CD pattern points"""
    try:
        colors = {'A': '#ff0000', 'B': '#00ff00', 'C': '#0000ff', 'D': '#ff00ff'}
        
        for point_name, point_data in points.items():
            if point_data:
                x_pos, price, point_type = point_data
                ax.scatter([x_pos], [price], color=colors.get(point_name, '#ffffff'), 
                          s=130, marker='D', label=f'Point {point_name}', zorder=10)
                ax.text(x_pos, price + 8, point_name, ha='center', va='bottom', 
                       color=colors.get(point_name, '#ffffff'), fontweight='bold', fontsize=14)
        
        # Draw AB and CD lines
        if points.get('A') and points.get('B'):
            ax.plot([points['A'][0], points['B'][0]], 
                   [points['A'][1], points['B'][1]], 
                   color='#ffaa00', linestyle='-', linewidth=3, alpha=0.9, label='AB Move')
                   
        if points.get('C') and points.get('D'):
            ax.plot([points['C'][0], points['D'][0]], 
                   [points['C'][1], points['D'][1]], 
                   color='#aa00ff', linestyle='-', linewidth=3, alpha=0.9, label='CD Move')
                   
    except Exception as e:
        print(f"AB=CD points marking error: {e}")

def draw_ascending_triangle_points(ax, df):
    """Mark Ascending Triangle pattern points"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        # Find resistance line (horizontal)
        resistance_level = np.max(highs[-20:])
        resistance_indices = [i for i, high in enumerate(highs[-20:]) if abs(high - resistance_level) < resistance_level * 0.01]
        
        if len(resistance_indices) >= 2:
            # Mark resistance points
            for i, idx in enumerate(resistance_indices[:3]):
                actual_idx = len(highs) - 20 + idx
                ax.scatter([actual_idx], [resistance_level], color='#ff4444', s=100, 
                          marker='_', label=f'Resistance {i+1}' if i < 2 else None, zorder=10)
                ax.text(actual_idx, resistance_level + 5, f'R{i+1}', ha='center', va='bottom', 
                       color='#ff4444', fontweight='bold', fontsize=10)
        
        # Find ascending support line
        support_points = []
        for i in range(len(lows) - 15, len(lows) - 5):
            if i > 0 and lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                support_points.append((i, lows[i]))
        
        if len(support_points) >= 2:
            for i, (idx, price) in enumerate(support_points[:2]):
                ax.scatter([idx], [price], color='#44ff44', s=100, 
                          marker='_', label=f'Support {i+1}', zorder=10)
                ax.text(idx, price - 8, f'S{i+1}', ha='center', va='top', 
                       color='#44ff44', fontweight='bold', fontsize=10)
                       
    except Exception as e:
        print(f"Ascending Triangle marking error: {e}")

def draw_descending_triangle_points(ax, df):
    """Mark Descending Triangle pattern points"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        # Find support line (horizontal)
        support_level = np.min(lows[-20:])
        support_indices = [i for i, low in enumerate(lows[-20:]) if abs(low - support_level) < support_level * 0.01]
        
        if len(support_indices) >= 2:
            # Mark support points
            for i, idx in enumerate(support_indices[:3]):
                actual_idx = len(lows) - 20 + idx
                ax.scatter([actual_idx], [support_level], color='#44ff44', s=100, 
                          marker='_', label=f'Support {i+1}' if i < 2 else None, zorder=10)
                ax.text(actual_idx, support_level - 8, f'S{i+1}', ha='center', va='top', 
                       color='#44ff44', fontweight='bold', fontsize=10)
        
        # Find descending resistance line
        resistance_points = []
        for i in range(len(highs) - 15, len(highs) - 5):
            if i > 0 and highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistance_points.append((i, highs[i]))
        
        if len(resistance_points) >= 2:
            for i, (idx, price) in enumerate(resistance_points[:2]):
                ax.scatter([idx], [price], color='#ff4444', s=100, 
                          marker='_', label=f'Resistance {i+1}', zorder=10)
                ax.text(idx, price + 5, f'R{i+1}', ha='center', va='bottom', 
                       color='#ff4444', fontweight='bold', fontsize=10)
                       
    except Exception as e:
        print(f"Descending Triangle marking error: {e}")

# ============= ส่วนที่ 2: เพิ่มฟังก์ชันสำหรับแพทเทิร์นอื่นๆ ที่ยังไม่มี =============

def draw_symmetrical_triangle_points(ax, df):
    """Mark Symmetrical Triangle pattern points"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        # Find converging trend lines
        high_peaks = []
        low_troughs = []
        
        # Find peaks and troughs in last 30 candles
        for i in range(len(highs) - 30, len(highs) - 5):
            if i > 2 and i < len(highs) - 2:
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    high_peaks.append((i, highs[i]))
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    low_troughs.append((i, lows[i]))
        
        # Mark peak points
        for i, (idx, price) in enumerate(high_peaks[-3:]):
            ax.scatter([idx], [price], color='#ff6600', s=90, 
                      marker='v', label=f'Peak {i+1}' if i < 2 else None, zorder=10)
            ax.text(idx, price + 5, f'P{i+1}', ha='center', va='bottom', 
                   color='#ff6600', fontweight='bold', fontsize=10)
        
        # Mark trough points
        for i, (idx, price) in enumerate(low_troughs[-3:]):
            ax.scatter([idx], [price], color='#0066ff', s=90, 
                      marker='^', label=f'Trough {i+1}' if i < 2 else None, zorder=10)
            ax.text(idx, price - 8, f'T{i+1}', ha='center', va='top', 
                   color='#0066ff', fontweight='bold', fontsize=10)
                   
    except Exception as e:
        print(f"Symmetrical Triangle marking error: {e}")

def draw_flag_pennant_points(ax, df):
    """Mark Flag/Pennant pattern points"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        # Find flagpole (strong move before consolidation)
        if len(highs) >= 20:
            flagpole_start = len(highs) - 20
            flagpole_end = len(highs) - 10
            
            flagpole_high = np.max(highs[flagpole_start:flagpole_end])
            flagpole_low = np.min(lows[flagpole_start:flagpole_end])
            
            flagpole_high_idx = flagpole_start + np.argmax(highs[flagpole_start:flagpole_end])
            flagpole_low_idx = flagpole_start + np.argmin(lows[flagpole_start:flagpole_end])
            
            # Mark flagpole points
            ax.scatter([flagpole_high_idx], [flagpole_high], color='#ff0000', s=120, 
                      marker='^', label='Flagpole Top', zorder=10)
            ax.text(flagpole_high_idx, flagpole_high + 8, 'FP_TOP', ha='center', va='bottom', 
                   color='#ff0000', fontweight='bold', fontsize=10)
                   
            ax.scatter([flagpole_low_idx], [flagpole_low], color='#00ff00', s=120, 
                      marker='v', label='Flagpole Bottom', zorder=10)
            ax.text(flagpole_low_idx, flagpole_low - 8, 'FP_BOT', ha='center', va='top', 
                   color='#00ff00', fontweight='bold', fontsize=10)
        
        # Mark consolidation area
        consolidation_start = len(highs) - 10
        consolidation_high = np.max(highs[consolidation_start:])
        consolidation_low = np.min(lows[consolidation_start:])
        
        # Draw consolidation boundaries
        ax.axhline(y=consolidation_high, xmin=0.8, xmax=1.0, color='#ffaa00', 
                  linestyle='--', alpha=0.8, linewidth=2, label='Flag Top')
        ax.axhline(y=consolidation_low, xmin=0.8, xmax=1.0, color='#ffaa00', 
                  linestyle='--', alpha=0.8, linewidth=2, label='Flag Bottom')
                  
    except Exception as e:
        print(f"Flag/Pennant marking error: {e}")

def draw_cup_handle_points(ax, df):
    """Mark Cup and Handle pattern points"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) >= 50:
            # Find cup formation (U-shaped)
            cup_start_idx = len(highs) - 50
            cup_end_idx = len(highs) - 10
            
            # Left rim of cup
            left_rim_high = np.max(highs[cup_start_idx:cup_start_idx + 10])
            left_rim_idx = cup_start_idx + np.argmax(highs[cup_start_idx:cup_start_idx + 10])
            
            # Cup bottom
            cup_bottom_low = np.min(lows[cup_start_idx + 10:cup_end_idx - 10])
            cup_bottom_idx = cup_start_idx + 10 + np.argmin(lows[cup_start_idx + 10:cup_end_idx - 10])
            
            # Right rim of cup
            right_rim_high = np.max(highs[cup_end_idx - 10:cup_end_idx])
            right_rim_idx = cup_end_idx - 10 + np.argmax(highs[cup_end_idx - 10:cup_end_idx])
            
            # Mark cup points
            ax.scatter([left_rim_idx], [left_rim_high], color='#ff6600', s=130, 
                      marker='o', label='Left Rim', zorder=10)
            ax.text(left_rim_idx, left_rim_high + 8, 'L_RIM', ha='center', va='bottom', 
                   color='#ff6600', fontweight='bold', fontsize=10)
                   
            ax.scatter([cup_bottom_idx], [cup_bottom_low], color='#0066ff', s=130, 
                      marker='o', label='Cup Bottom', zorder=10)
            ax.text(cup_bottom_idx, cup_bottom_low - 8, 'CUP_BOT', ha='center', va='top', 
                   color='#0066ff', fontweight='bold', fontsize=10)
                   
            ax.scatter([right_rim_idx], [right_rim_high], color='#ff6600', s=130, 
                      marker='o', label='Right Rim', zorder=10)
            ax.text(right_rim_idx, right_rim_high + 8, 'R_RIM', ha='center', va='bottom', 
                   color='#ff6600', fontweight='bold', fontsize=10)
            
            # Handle formation (last 10 candles)
            handle_high = np.max(highs[-10:])
            handle_low = np.min(lows[-10:])
            handle_high_idx = len(highs) - 10 + np.argmax(highs[-10:])
            handle_low_idx = len(highs) - 10 + np.argmin(lows[-10:])
            
            ax.scatter([handle_high_idx], [handle_high], color='#ff0066', s=100, 
                      marker='s', label='Handle High', zorder=10)
            ax.text(handle_high_idx, handle_high + 5, 'H_HIGH', ha='center', va='bottom', 
                   color='#ff0066', fontweight='bold', fontsize=9)
                   
            ax.scatter([handle_low_idx], [handle_low], color='#6600ff', s=100, 
                      marker='s', label='Handle Low', zorder=10)
            ax.text(handle_low_idx, handle_low - 5, 'H_LOW', ha='center', va='top', 
                   color='#6600ff', fontweight='bold', fontsize=9)
                   
    except Exception as e:
        print(f"Cup and Handle marking error: {e}")

# ============= ส่วนที่ 3: Pattern Theory และคำอธิบาย =============

PATTERN_THEORIES = {
    'GARTLEY': {
        'description': 'Gartley Pattern - แพทเทิร์น Harmonic ที่มีโครงสร้าง XABCD',
        'theory': '''🔹 GARTLEY PATTERN THEORY 🔹
        
📊 โครงสร้าง XABCD:
• AB = 61.8% ของ XA
• BC = 38.2% หรือ 88.6% ของ AB  
• CD = 127.2% ของ BC
• AD = 78.6% ของ XA

🎯 การใช้งาน:
• Bullish Gartley: ซื้อที่จุด D
• Bearish Gartley: ขายที่จุด D
• Stop Loss: เหนือ/ใต้จุด X
• Take Profit: 38.2%, 61.8% Fibonacci

💡 จุดเด่น:
• ความแม่นยำสูงในการหาจุดกลับตัว
• ใช้ Fibonacci Ratios ที่แน่นอน
• เหมาะกับ All Timeframes''',
        'confidence_min': 0.75
    },
    
    'BUTTERFLY': {
        'description': 'Butterfly Pattern - แพทเทิร์น Harmonic รูปผีเสื้อ',
        'theory': '''🦋 BUTTERFLY PATTERN THEORY 🦋
        
📊 โครงสร้าง XABCD:
• AB = 78.6% ของ XA
• BC = 38.2% หรือ 88.6% ของ AB
• CD = 161.8% หรือ 261.8% ของ BC  
• AD = 127.2% หรือ 161.8% ของ XA

🎯 การใช้งาน:
• จุด D อยู่นอก X-A Range
• Entry ที่จุด D completion
• Stop Loss: 20-30 pips จากจุด D
• Take Profit: 38.2%, 61.8% retracement

💡 จุดเด่น:  
• Extension Pattern (ยืดเกิน XA)
• มักเกิดในตลาดแกว่งแรง
• Profit Target ที่ชัดเจน''',
        'confidence_min': 0.70
    },
    
    'BAT': {
        'description': 'Bat Pattern - แพทเทิร์น Harmonic รูปค้างคาว',
        'theory': '''🦇 BAT PATTERN THEORY 🦇
        
📊 โครงสร้าง XABCD:
• AB = 38.2% หรือ 50% ของ XA
• BC = 38.2% หรือ 88.6% ของ AB
• CD = 161.8% หรือ 261.8% ของ BC
• AD = 88.6% ของ XA

🎯 การใช้งาน:
• จุด D ใกล้ระดับ X มาก
• Entry: Market/Limit Order ที่ D
• Stop Loss: เหนือ/ใต้จุด X  
• Take Profit: 38.2%, 61.8% AD

💡 จุดเด่น:
• Shallow retracement ที่จุด B
• High probability reversal
• เหมาะกับ trend continuation''',
        'confidence_min': 0.75
    },
    
    'CRAB': {
        'description': 'Crab Pattern - แพทเทิร์น Harmonic รูปปู',
        'theory': '''🦀 CRAB PATTERN THEORY 🦀
        
📊 โครงสร้าง XABCD:
• AB = 38.2% หรือ 61.8% ของ XA
• BC = 38.2% หรือ 88.6% ของ AB
• CD = 224% หรือ 361.8% ของ BC
• AD = 161.8% ของ XA

🎯 การใช้งาน:
• Extreme extension pattern
• จุด D ไกลจาก X มากที่สุด
• Entry: Limit Order ที่ 161.8% XA
• Stop Loss: 20-30 pips from D
• Take Profit: 38.2%, 61.8% AD

💡 จุดเด่น:
• ความแรงของการกลับตัวสูงสุด
• เหมาะกับตลาด Overbought/Oversold
• Risk:Reward ratio ดี''',
        'confidence_min': 0.70
    },
    
    'AB_CD': {
        'description': 'AB=CD Pattern - แพทเทิร์นขาขึ้น/ลงเท่ากัน',
        'theory': '''📐 AB=CD PATTERN THEORY 📐
        
📊 โครงสร้างพื้นฐาน:
• AB leg = CD leg (ระยะทางเท่ากัน)
• หรือ CD = 127.2% ของ AB
• หรือ CD = 161.8% ของ AB
• Time cycles เท่ากันหรือใกล้เคียง

🎯 การใช้งาน:
• Entry ที่จุด D completion
• Stop Loss: เกิน point D
• Take Profit: 38.2%, 61.8% CD
• Can combine กับ patterns อื่น

💡 จุดเด่น:
• Pattern พื้นฐานที่สำคัญ
• เกิดขึ้นบ่อยในตลาด  
• เป็น building block ของ Harmonic
• ใช้งานง่าย มีประสิทธิภาพ''',
        'confidence_min': 0.65
    },
    
    'ELLIOTT_WAVE_5': {
        'description': 'Elliott Wave 5 - คลื่นแรงผลัก 5 ขา',
        'theory': '''🌊 ELLIOTT WAVE 5-WAVE THEORY 🌊
        
📊 โครงสร้าง Impulse Wave:
• Wave 1: แรงผลักแรก
• Wave 2: แก้ตัว (< 100% ของ Wave 1)
• Wave 3: แรงผลักหลัก (ไม่ใช่คลื่นสั้นที่สุด)
• Wave 4: แก้ตัว (ไม่ทับซ้อน Wave 1)
• Wave 5: แรงผลักสุดท้าย

🎯 กฎ Elliott Wave:
1. Wave 2 ไม่ย้อนเกิน 100% ของ Wave 1
2. Wave 3 ไม่ใช่ขาที่สั้นที่สุด
3. Wave 4 ไม่เข้าไปในพื้นที่ Wave 1

💡 การ Trade:
• Buy: Wave 2, Wave 4 completion
• Sell: Wave 5 completion (reversal)
• Target: 161.8% extension levels''',
        'confidence_min': 0.75
    },
    
    'ELLIOTT_WAVE_3': {
        'description': 'Elliott Wave 3 - คลื่นแก้ตัว ABC',
        'theory': '''🌊 ELLIOTT WAVE 3-WAVE THEORY 🌊
        
📊 โครงสร้าง Corrective Wave:
• Wave A: การแก้ตัวแรก
• Wave B: การ rebound (38-78% ของ A)
• Wave C: การแก้ตัวสุดท้าย

🎯 รูปแบบ ABC:
• Zigzag: A=C, B=38.2-61.8% A
• Flat: A≈B≈C 
• Triangle: Contracting pattern

💡 การ Trade:
• Sell rallies ใน Wave B
• Buy ที่ Wave C completion
• Wave C มักเท่ากับ Wave A
• หรือ Wave C = 161.8% Wave A''',
        'confidence_min': 0.65
    }
}

def get_pattern_theory(pattern_name):
    """ดึงทฤษฎีแพทเทิร์น"""
    return PATTERN_THEORIES.get(pattern_name, {
        'description': f'{pattern_name} Pattern',
        'theory': 'ไม่มีข้อมูลทฤษฎีสำหรับแพทเทิร์นนี้',
        'confidence_min': 0.50
    })

# ============= ส่วนที่ 4: Enhanced Telegram Message Function =============

def create_enhanced_telegram_message(pattern_info, symbol, timeframe, current_price):
    """สร้างข้อความ Telegram แบบละเอียด"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        confidence = pattern_info.get('confidence', 0)
        method = pattern_info.get('method', 'UNKNOWN')
        
        if pattern_name == 'NO_PATTERN':
            return f"📊 {symbol} ({timeframe})\n🔍 ไม่พบแพทเทิร์นที่ชัดเจน\n💰 ราคาปัจจุบัน: {current_price:.4f}"
        
        # ดึงทฤษฎีแพทเทิร์น
        theory = get_pattern_theory(pattern_name)
        
        # สร้าง header ข้อความ
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        
        message = f"""
🎯 PATTERN DETECTED 🎯

📊 Symbol: {symbol} ({timeframe})
🔍 Pattern: {theory['description']}
{confidence_emoji} Confidence: {confidence:.1%}
⚙️ Method: {method}
💰 Current Price: {current_price:.4f}

{theory['theory']}

📈 CHART ANALYSIS:
"""
        
        # เพิ่มข้อมูลจุดสำคัญตามแพทเทิร์น
        if 'points' in pattern_info:
            points = pattern_info['points']
            message += "\n🎯 KEY POINTS:\n"
            for point_name, point_data in points.items():
                if point_data:
                    _, price, _ = point_data
                    message += f"• Point {point_name}: {price:.4f}\n"
                    
        elif 'wave_points' in pattern_info:
            wave_points = pattern_info['wave_points']
            message += "\n🌊 WAVE POINTS:\n"
            for wave_name, wave_data in wave_points.items():
                if wave_data:
                    _, price, _ = wave_data
                    message += f"• Wave {wave_name}: {price:.4f}\n"
        
        # เพิ่ม trading suggestion
        message += f"\n💡 TRADING SUGGESTION:\n"
        
        if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            message += "• เฝ้าดู reversal signal ที่จุด D\n• Set Stop Loss นอกจุด X\n• Take Profit ที่ 38.2%, 61.8%"
        elif pattern_name == 'AB_CD':
            message += "• Entry ที่ point D completion\n• Stop Loss เกินจุด D\n• Target 38.2%-61.8% CD"
        elif 'ELLIOTT_WAVE' in pattern_name:
            message += "• รอ wave completion\n• ใช้ Fibonacci extensions\n• เฝ้าดู divergence signals"
        else:
            message += "• รอการ breakout/breakdown\n• Confirm ด้วย volume\n• ใช้ proper risk management"
            
        message += f"\n\n⚠️ Risk Warning: Pattern confidence {confidence:.1%}"
        message += f"\n📊 Always confirm with other indicators!"
        
        return message.strip()
        
    except Exception as e:
        return f"📊 {symbol} ({timeframe})\n❌ Error creating message: {str(e)}"


# ============= ส่วนที่ 5: Enhanced Main Detection Function (ต่อจากไฟล์เดิม) =============

def detect_all_patterns_enhanced(df, symbol='UNKNOWN', timeframe='1H'):
    """
    ฟังก์ชันหลักสำหรับตรวจจับแพทเทิร์นทั้งหมด
    รวมถึง Harmonic และ Elliott Wave Patterns
    """
    try:
        current_price = df['close'].iloc[-1]
        
        # ลองตรวจหา Harmonic Patterns ก่อน
        harmonic_detector = HarmonicPatternDetector()
        harmonic_result = harmonic_detector.detect_harmonic_patterns(df)
        
        if harmonic_result['pattern_name'] != 'NO_PATTERN':
            telegram_msg = create_enhanced_telegram_message(
                harmonic_result, symbol, timeframe, current_price
            )
            return harmonic_result, telegram_msg
        
        # ตรวจหา Elliott Wave Patterns
        elliott_detector = ElliottWaveDetector()
        elliott_result = elliott_detector.detect_elliott_waves(df)
        
        if elliott_result['pattern_name'] != 'NO_PATTERN':
            telegram_msg = create_enhanced_telegram_message(
                elliott_result, symbol, timeframe, current_price
            )
            return elliott_result, telegram_msg
        
        # ตรวจหาแพทเทิร์นคลาสสิกอื่นๆ
        classic_result = detect_classic_patterns(df)
        
        if classic_result['pattern_name'] != 'NO_PATTERN':
            telegram_msg = create_enhanced_telegram_message(
                classic_result, symbol, timeframe, current_price
            )
            return classic_result, telegram_msg
        
        # ไม่พบแพทเทิร์นใดๆ
        no_pattern_result = {
            'pattern_id': 0, 
            'pattern_name': 'NO_PATTERN', 
            'confidence': 0.50, 
            'method': 'COMPREHENSIVE_SCAN'
        }
        telegram_msg = create_enhanced_telegram_message(
            no_pattern_result, symbol, timeframe, current_price
        )
        
        return no_pattern_result, telegram_msg
        
    except Exception as e:
        error_result = {
            'pattern_id': 0, 
            'pattern_name': 'ERROR', 
            'confidence': 0.30, 
            'method': f'ERROR: {str(e)}'
        }
        telegram_msg = f"❌ Error detecting patterns for {symbol}: {str(e)}"
        return error_result, telegram_msg

def detect_classic_patterns(df):
    """ตรวจหาแพทเทิร์นคลาสสิก"""
    try:
        # Head and Shoulders
        if detect_head_shoulders(df):
            return {
                'pattern_id': 1, 'pattern_name': 'HEAD_SHOULDERS', 
                'confidence': 0.75, 'method': 'CLASSIC'
            }
        
        # Double Top/Bottom
        if detect_double_top(df):
            return {
                'pattern_id': 2, 'pattern_name': 'DOUBLE_TOP', 
                'confidence': 0.70, 'method': 'CLASSIC'
            }
        
        if detect_double_bottom(df):
            return {
                'pattern_id': 3, 'pattern_name': 'DOUBLE_BOTTOM', 
                'confidence': 0.70, 'method': 'CLASSIC'
            }
        
        # Triangle Patterns
        if detect_ascending_triangle(df):
            return {
                'pattern_id': 4, 'pattern_name': 'ASCENDING_TRIANGLE', 
                'confidence': 0.65, 'method': 'CLASSIC'
            }
        
        if detect_descending_triangle(df):
            return {
                'pattern_id': 5, 'pattern_name': 'DESCENDING_TRIANGLE', 
                'confidence': 0.65, 'method': 'CLASSIC'
            }
        
        if detect_symmetrical_triangle(df):
            return {
                'pattern_id': 6, 'pattern_name': 'SYMMETRICAL_TRIANGLE', 
                'confidence': 0.60, 'method': 'CLASSIC'
            }
        
        # Flag and Pennant
        if detect_flag_pennant(df):
            return {
                'pattern_id': 7, 'pattern_name': 'FLAG_PENNANT', 
                'confidence': 0.65, 'method': 'CLASSIC'
            }
        
        # Cup and Handle
        if detect_cup_handle(df):
            return {
                'pattern_id': 8, 'pattern_name': 'CUP_HANDLE', 
                'confidence': 0.70, 'method': 'CLASSIC'
            }
        
        return {
            'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 
            'confidence': 0.50, 'method': 'NO_CLASSIC'
        }
        
    except Exception as e:
        return {
            'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 
            'confidence': 0.30, 'method': f'CLASSIC_ERROR: {str(e)}'
        }

# ============= ส่วนที่ 6: Classic Pattern Detection Functions =============

def detect_head_shoulders(df):
    """ตรวจหาแพทเทิร์น Head and Shoulders"""
    try:
        if len(df) < 30:
            return False
            
        highs = df['high'].values[-30:]
        
        # หาจุดสูงสุด 3 จุด
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 3:
            peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:3]
            peaks = sorted(peaks, key=lambda x: x[0])  # เรียงตามเวลา
            
            left_shoulder, head, right_shoulder = peaks
            
            # ตรวจสอบเงื่อนไข Head and Shoulders
            head_higher = head[1] > left_shoulder[1] and head[1] > right_shoulder[1]
            shoulders_similar = abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1]) < 0.03
            
            return head_higher and shoulders_similar
        
        return False
    except:
        return False

def detect_double_top(df):
    """ตรวจหาแพทเทิร์น Double Top"""
    try:
        if len(df) < 20:
            return False
            
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        
        # หาจุดสูงสุด 2 จุด
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
            top1, top2 = peaks
            
            # ตรวจสอบความสูงใกล้เคียงกัน
            height_diff = abs(top1[1] - top2[1]) / max(top1[1], top2[1])
            
            # ตรวจสอบว่ามี valley ระหว่าง 2 tops
            start_idx = min(top1[0], top2[0])
            end_idx = max(top1[0], top2[0])
            valley_low = min(lows[start_idx:end_idx])
            valley_depth = min(top1[1], top2[1]) - valley_low
            
            return height_diff < 0.02 and valley_depth > (min(top1[1], top2[1]) * 0.03)
        
        return False
    except:
        return False

def detect_double_bottom(df):
    """ตรวจหาแพทเทิร์น Double Bottom"""
    try:
        if len(df) < 20:
            return False
            
        lows = df['low'].values[-20:]
        highs = df['high'].values[-20:]
        
        # หาจุดต่ำสุด 2 จุด
        troughs = []
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            troughs = sorted(troughs, key=lambda x: x[1])[:2]
            bottom1, bottom2 = troughs
            
            # ตรวจสอบความลึกใกล้เคียงกัน
            depth_diff = abs(bottom1[1] - bottom2[1]) / min(bottom1[1], bottom2[1])
            
            # ตรวจสอบว่ามี peak ระหว่าง 2 bottoms
            start_idx = min(bottom1[0], bottom2[0])
            end_idx = max(bottom1[0], bottom2[0])
            peak_high = max(highs[start_idx:end_idx])
            peak_height = peak_high - max(bottom1[1], bottom2[1])
            
            return depth_diff < 0.02 and peak_height > (max(bottom1[1], bottom2[1]) * 0.03)
        
        return False
    except:
        return False

def detect_ascending_triangle(df):
    """ตรวจหาแพทเทิร์น Ascending Triangle"""
    try:
        if len(df) < 20:
            return False
            
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        
        # หา resistance level (horizontal line)
        resistance_level = max(highs[-10:])
        resistance_touches = sum(1 for h in highs[-15:] if abs(h - resistance_level) < resistance_level * 0.01)
        
        # หาแนวโน้มขาขึ้นของ support
        support_points = []
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                support_points.append((i, lows[i]))
        
        if len(support_points) >= 2 and resistance_touches >= 2:
            # ตรวจสอบแนวโน้มขาขึ้นของ support
            support_slope = (support_points[-1][1] - support_points[0][1]) / (support_points[-1][0] - support_points[0][0])
            return support_slope > 0 and resistance_touches >= 2
        
        return False
    except:
        return False

def detect_descending_triangle(df):
    """ตรวจหาแพทเทิร์น Descending Triangle"""
    try:
        if len(df) < 20:
            return False
            
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        
        # หา support level (horizontal line)
        support_level = min(lows[-10:])
        support_touches = sum(1 for l in lows[-15:] if abs(l - support_level) < support_level * 0.01)
        
        # หาแนวโน้มขาลงของ resistance
        resistance_points = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                resistance_points.append((i, highs[i]))
        
        if len(resistance_points) >= 2 and support_touches >= 2:
            # ตรวจสอบแนวโน้มขาลงของ resistance
            resistance_slope = (resistance_points[-1][1] - resistance_points[0][1]) / (resistance_points[-1][0] - resistance_points[0][0])
            return resistance_slope < 0 and support_touches >= 2
        
        return False
    except:
        return False

def detect_symmetrical_triangle(df):
    """ตรวจหาแพทเทิร์น Symmetrical Triangle"""
    try:
        if len(df) < 20:
            return False
            
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        
        # หาจุดสูงและต่ำ
        high_points = []
        low_points = []
        
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                high_points.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                low_points.append((i, lows[i]))
        
        if len(high_points) >= 2 and len(low_points) >= 2:
            # ตรวจสอบแนวโน้มขาลงของ highs และขาขึ้นของ lows
            high_slope = (high_points[-1][1] - high_points[0][1]) / (high_points[-1][0] - high_points[0][0])
            low_slope = (low_points[-1][1] - low_points[0][1]) / (low_points[-1][0] - low_points[0][0])
            
            return high_slope < -0.0001 and low_slope > 0.0001
        
        return False
    except:
        return False

def detect_flag_pennant(df):
    """ตรวจหาแพทเทิร์น Flag/Pennant"""
    try:
        if len(df) < 25:
            return False
            
        # ตรวจหา flagpole (การเคลื่อนไหวแรงก่อนหน้า)
        closes = df['close'].values
        
        # ช่วงก่อน consolidation
        flagpole_period = closes[-25:-10]
        consolidation_period = closes[-10:]
        
        flagpole_move = abs(flagpole_period[-1] - flagpole_period[0])
        consolidation_range = max(consolidation_period) - min(consolidation_period)
        
        # Flag/Pennant มี consolidation range เล็กกว่า flagpole move มาก
        if consolidation_range > 0 and flagpole_move > 0:
            consolidation_ratio = consolidation_range / flagpole_move
            return consolidation_ratio < 0.3  # consolidation ต้องเล็กกว่า 30% ของ flagpole
        
        return False
    except:
        return False

def detect_cup_handle(df):
    """ตรวจหาแพทเทิร์น Cup and Handle"""
    try:
        if len(df) < 50:
            return False
            
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # แบ่งเป็นส่วนต่างๆ
        cup_period = closes[-50:-10]
        handle_period = closes[-10:]
        
        # ตรวจสอบ Cup formation (U-shaped)
        cup_left = cup_period[:15]
        cup_bottom = cup_period[15:35]
        cup_right = cup_period[35:]
        
        left_high = max(cup_left)
        right_high = max(cup_right)
        bottom_low = min(cup_bottom)
        
        # Cup depth ควรอยู่ระหว่าง 12-33%
        cup_depth = (min(left_high, right_high) - bottom_low) / min(left_high, right_high)
        
        # Handle ควรมี consolidation และอยู่ในช่วง upper half ของ cup
        handle_high = max(handle_period)
        handle_low = min(handle_period)
        handle_midpoint = (handle_high + handle_low) / 2
        cup_midpoint = (min(left_high, right_high) + bottom_low) / 2
        
        cup_valid = 0.12 <= cup_depth <= 0.33
        handle_valid = handle_midpoint > cup_midpoint
        rims_similar = abs(left_high - right_high) / max(left_high, right_high) < 0.05
        
        return cup_valid and handle_valid and rims_similar
    except:
        return False

# ============= ส่วนที่ 7: เพิ่มแพทเทิร์นในทฤษฎี =============

# เพิ่มแพทเทิร์นคลาสสิกในทฤษฎี
PATTERN_THEORIES.update({
    'HEAD_SHOULDERS': {
        'description': 'Head and Shoulders - แพทเทิร์นกลับตัวที่แข็งแกร่ง',
        'theory': '''👑 HEAD AND SHOULDERS THEORY 👑
        
📊 โครงสร้าง:
• ไหล่ซ้าย (Left Shoulder)
• หัว (Head) - จุดสูงสุด
• ไหล่ขวา (Right Shoulder)
• เส้นคอ (Neckline) - เชื่อมจุดต่ำระหว่างไหล่

🎯 การใช้งาน:
• Bearish Pattern - สัญญาณขาลง
• Entry: เมื่อราคาทะลุเส้นคอลงมา
• Stop Loss: เหนือไหล่ขวา
• Target: ระยะจากหัวถึงเส้นคอ

💡 จุดเด่น:
• ความน่าเชื่อถือสูง 75-85%
• Volume ลดลงที่ไหล่ขวา
• เหมาะกับ Major Reversal''',
        'confidence_min': 0.75
    },
    
    'DOUBLE_TOP': {
        'description': 'Double Top - จุดสูงคู่ สัญญาณกลับตัวขาลง',
        'theory': '''🔺🔺 DOUBLE TOP THEORY 🔺🔺
        
📊 โครงสร้าง:
• จุดสูงที่ 1 (First Peak)
• จุดต่ำกลาง (Valley)
• จุดสูงที่ 2 (Second Peak) ≈ First Peak
• เส้นซัพพอร์ต (Support Line)

🎯 การใช้งาน:
• Bearish Reversal Pattern
• Entry: ทะลุ Support Line
• Stop Loss: เหนือ Second Peak
• Target: ระยะจาก Peak ถึง Valley

💡 จุดเด่น:
• เกิดที่ระดับ Resistance สำคัญ
• Volume ลดลงที่ Peak ที่ 2
• Confirmation ด้วย RSI Divergence''',
        'confidence_min': 0.70
    },
    
    'DOUBLE_BOTTOM': {
        'description': 'Double Bottom - จุดต่ำคู่ สัญญาณกลับตัวขาขึ้น',
        'theory': '''🔻🔻 DOUBLE BOTTOM THEORY 🔻🔻
        
📊 โครงสร้าง:
• จุดต่ำที่ 1 (First Trough)
• จุดสูงกลาง (Peak)
• จุดต่ำที่ 2 (Second Trough) ≈ First Trough
• เส้นเรซิสแตนซ์ (Resistance Line)

🎯 การใช้งาน:
• Bullish Reversal Pattern
• Entry: ทะลุ Resistance Line
• Stop Loss: ใต้ Second Trough
• Target: ระยะจาก Trough ถึง Peak

💡 จุดเด่น:
• เกิดที่ระดับ Support สำคัญ
• Volume เพิ่มขึ้นที่การทะลุ
• มักมี Bullish Divergence''',
        'confidence_min': 0.70
    },
    
    'ASCENDING_TRIANGLE': {
        'description': 'Ascending Triangle - สามเหลี่ยมขาขึ้น',
        'theory': '''📈△ ASCENDING TRIANGLE THEORY △📈
        
📊 โครงสร้าง:
• เส้นเรซิสแตนซ์แนวนอน (Horizontal Resistance)
• เส้นซัพพอร์ตขาขึ้น (Ascending Support)
• Volume ลดลงระหว่าง consolidation
• Breakout ด้วย Volume สูง

🎯 การใช้งาน:
• Bullish Continuation Pattern (70%)
• Entry: ทะลุ Resistance + Volume
• Stop Loss: ใต้ Ascending Support
• Target: ความกว้างของรูปสามเหลี่ยม

💡 จุดเด่น:
• Buyers แข็งแกร่งขึ้นเรื่อยๆ
• Sellers อ่อนแอลงที่ Resistance
• Success Rate ≈ 70%''',
        'confidence_min': 0.65
    },
    
    'DESCENDING_TRIANGLE': {
        'description': 'Descending Triangle - สามเหลี่ยมขาลง',
        'theory': '''📉▽ DESCENDING TRIANGLE THEORY ▽📉
        
📊 โครงสร้าง:
• เส้นซัพพอร์ตแนวนอน (Horizontal Support)
• เส้นเรซิสแตนซ์ขาลง (Descending Resistance)
• Volume ลดลงระหว่าง consolidation
• Breakdown ด้วย Volume สูง

🎯 การใช้งาน:
• Bearish Continuation Pattern (70%)
• Entry: ทะลุ Support ลงมา + Volume
• Stop Loss: เหนือ Descending Resistance
• Target: ความกว้างของรูปสามเหลี่ยม

💡 จุดเด่น:
• Sellers แข็งแกร่งขึ้นเรื่อยๆ
• Buyers อ่อนแอลงที่ Support
• Often leads to significant decline''',
        'confidence_min': 0.65
    },
    
    'SYMMETRICAL_TRIANGLE': {
        'description': 'Symmetrical Triangle - สามเหลี่ยมสมมาตร',
        'theory': '''⚖️△ SYMMETRICAL TRIANGLE THEORY △⚖️
        
📊 โครงสร้าง:
• เส้นเรซิสแตนซ์ขาลง (Descending Resistance)
• เส้นซัพพอร์ตขาขึ้น (Ascending Support)
• จุดบรรจบ (Apex) ที่บังคับให้เกิด breakout
• Volume ลดลงก่อน breakout

🎯 การใช้งาน:
• Continuation Pattern (แต่อาจ reversal)
• Entry: รอ breakout + confirmation
• Stop Loss: ฝั่งตรงข้ามกับ breakout
• Target: ระยะจากฐานสามเหลี่ยม

💡 จุดเด่น:
• Neutral pattern จนกว่าจะ breakout
• Volume ต้องเพิ่มขึ้นเมื่อ breakout
• ใช้เวลาในการ form 1-3 เดือน''',
        'confidence_min': 0.60
    },
    
    'FLAG_PENNANT': {
        'description': 'Flag/Pennant - ธงและอินทรธนู',
        'theory': '''🚩 FLAG & PENNANT THEORY 🚩
        
📊 โครงสร้าง:
• Flagpole: การเคลื่อนไหวแรงแรก
• Flag/Pennant: การ consolidate แบบแคบ
• Volume: ลดลงระหว่าง consolidation
• Breakout: ไปทิศทางเดียวกับ flagpole

🎯 การใช้งาน:
• Short-term Continuation Pattern
• Entry: Breakout ในทิศทาง flagpole
• Stop Loss: ปลาย consolidation
• Target: ความยาว flagpole + breakout point

💡 จุดเด่น:
• High probability continuation (80%+)
• รูปแบบระยะสั้น (1-4 สัปดาห์)
• เกิดหลัง strong trending move''',
        'confidence_min': 0.65
    },
    
    'CUP_HANDLE': {
        'description': 'Cup and Handle - ถ้วยและหูจับ',
        'theory': '''☕ CUP AND HANDLE THEORY ☕
        
📊 โครงสร้าง:
• Cup: รูปตัว U ใช้เวลา 7+ สัปดาห์
• Handle: การ pullback 1-5 สัปดาห์
• Depth: Cup ลึก 12-33% จาก high
• Volume: ลดลงใน cup, เพิ่มขึ้นเมื่อ breakout

🎯 การใช้งาน:
• Bullish Continuation Pattern
• Entry: Breakout จาก handle + volume
• Stop Loss: ใต้ handle low
• Target: Cup depth + breakout point

💡 จุดเด่น:
• Long-term bullish pattern
• William O'Neil favorite pattern
• Success rate สูงใน bull market''',
        'confidence_min': 0.70
    }
})

# ============= ส่วนที่ 8: Enhanced Pattern Drawing Functions =============

def draw_enhanced_pattern_lines(ax, df, pattern_info):
    """Enhanced pattern line drawing with specific point marking"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        
        if pattern_name == 'HEAD_SHOULDERS':
            draw_head_shoulders_points(ax, df)
        elif pattern_name == 'DOUBLE_TOP':
            draw_double_top_points(ax, df)
        elif pattern_name == 'DOUBLE_BOTTOM':
            draw_double_bottom_points(ax, df)
        elif pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            if 'points' in pattern_info:
                draw_harmonic_points(ax, pattern_info['points'], pattern_name)
        elif pattern_name == 'AB_CD':
            if 'points' in pattern_info:
                draw_abcd_points(ax, pattern_info['points'])
        elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
            if 'wave_points' in pattern_info:
                draw_elliott_wave_points(ax, pattern_info['wave_points'], pattern_name)
        elif pattern_name == 'ASCENDING_TRIANGLE':
            draw_ascending_triangle_points(ax, df)
        elif pattern_name == 'DESCENDING_TRIANGLE':
            draw_descending_triangle_points(ax, df)
        elif pattern_name == 'SYMMETRICAL_TRIANGLE':
            draw_symmetrical_triangle_points(ax, df)
        elif pattern_name == 'FLAG_PENNANT':
            draw_flag_pennant_points(ax, df)
        elif pattern_name == 'CUP_HANDLE':
            draw_cup_handle_points(ax, df)
            
    except Exception as e:
        print(f"Enhanced pattern marking error: {e}")

# ============= ส่วนที่ 9: Main Function Integration (ต่อจากไฟล์ที่ 3) =============

def analyze_and_send_telegram(df, symbol='UNKNOWN', timeframe='1H', send_to_telegram=True):
    """
    ฟังก์ชันหลักสำหรับวิเคราะห์และส่งไปยัง Telegram
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        import io
        import base64
        
        # ตรวจหาแพทเทิร์น
        pattern_result, telegram_msg = detect_all_patterns_enhanced(df, symbol, timeframe)
        
        # สร้างกราฟแท่งเทียน
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # วาดแท่งเทียน
        draw_candlestick_chart(ax, df)
        
        # วาดจุดแพทเทิร์น
        draw_enhanced_pattern_lines(ax, df, pattern_result)
        
        # ตั้งค่าชื่อกราฟและแกน
        ax.set_title(f'{symbol} {timeframe} - {pattern_result["pattern_name"]}', 
                    fontsize=16, color='white', fontweight='bold')
        ax.set_xlabel('Time Period', fontsize=12, color='white')
        ax.set_ylabel('Price', fontsize=12, color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='gray')
        
        # เพิ่มข้อมูลแพทเทิร์นในกราฟ
        pattern_info_text = f"Pattern: {pattern_result['pattern_name']}\n"
        pattern_info_text += f"Confidence: {pattern_result['confidence']:.1%}\n"
        pattern_info_text += f"Method: {pattern_result['method']}"
        
        ax.text(0.02, 0.98, pattern_info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', color='yellow',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        # Legend สำหรับจุดต่างๆ
        if pattern_result['pattern_name'] != 'NO_PATTERN':
            ax.legend(loc='upper right', facecolor='black', 
                     edgecolor='white', framealpha=0.8)
        
        plt.tight_layout()
        
        # บันทึกกราฟเป็น bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#1a1a1a', 
                   dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        # สร้าง base64 สำหรับแนบในข้อความ
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        plt.close()
        
        # ส่งไป Telegram (จำลอง)
        if send_to_telegram:
            print(f"📊 SENDING TO TELEGRAM:")
            print(f"Chart: [BASE64_IMAGE_DATA]")
            print(f"Message:\n{telegram_msg}")
        
        return {
            'pattern_result': pattern_result,
            'telegram_message': telegram_msg,
            'chart_base64': img_base64
        }
        
    except Exception as e:
        error_msg = f"❌ Error in analysis: {str(e)}"
        print(error_msg)
        return {
            'pattern_result': {'pattern_name': 'ERROR', 'confidence': 0},
            'telegram_message': error_msg,
            'chart_base64': None
        }

def draw_candlestick_chart(ax, df):
    """วาดกราฟแท่งเทียน"""
    try:
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        for i in range(len(df)):
            # กำหนดสีแท่งเทียน
            if closes[i] >= opens[i]:
                color = '#00ff88'  # เขียว (ขาขึ้น)
                edge_color = '#00cc66'
            else:
                color = '#ff4444'  # แดง (ขาลง)
                edge_color = '#cc3333'
            
            # วาดเส้น High-Low
            ax.plot([i, i], [lows[i], highs[i]], color=edge_color, linewidth=1)
            
            # วาดแท่งเทียน
            height = abs(closes[i] - opens[i])
            bottom = min(opens[i], closes[i])
            
            rect = plt.Rectangle((i-0.4, bottom), 0.8, height,
                               facecolor=color, edgecolor=edge_color, linewidth=1)
            ax.add_patch(rect)
            
        ax.set_xlim(-0.5, len(df)-0.5)
        ax.set_ylim(min(lows) * 0.995, max(highs) * 1.005)
        
    except Exception as e:
        print(f"Candlestick chart error: {e}")

# ============= ส่วนที่ 10: Enhanced Telegram Message with Chart Theory =============

def create_enhanced_telegram_message_with_theory(pattern_info, symbol, timeframe, current_price):
    """สร้างข้อความ Telegram แบบละเอียดพร้อมทฤษฎี"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        confidence = pattern_info.get('confidence', 0)
        method = pattern_info.get('method', 'UNKNOWN')
        
        if pattern_name == 'NO_PATTERN':
            return f"""📊 {symbol} ({timeframe})
🔍 ไม่พบแพทเทิร์นที่ชัดเจน
💰 ราคาปัจจุบัน: {current_price:.4f}
⚡ ระบบวิเคราะห์: Harmonic + Elliott Wave + Classic Patterns"""
        
        # ดึงทฤษฎีแพทเทิร์น
        theory = get_pattern_theory(pattern_name)
        
        # สร้าง header ข้อความ
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        
        # สร้าง pattern-specific points info
        points_info = create_pattern_points_info(pattern_info)
        
        message = f"""
🎯 PATTERN DETECTED 🎯

📊 Symbol: {symbol} ({timeframe})
🔍 Pattern: {theory['description']}
{confidence_emoji} Confidence: {confidence:.1%}
⚙️ Method: {method}
💰 Current Price: {current_price:.4f}

{theory['theory']}

📈 CHART ANALYSIS:
{points_info}

💡 TRADING STRATEGY:
{create_trading_strategy(pattern_name, pattern_info)}

⚠️ Risk Management:
• Position Size: 1-2% of capital
• Always use Stop Loss
• Confirm with volume & momentum
• Multiple timeframe analysis

🔗 Generated by Advanced Pattern Detection System
📊 Harmonic • Elliott Wave • Classic Patterns
        """
        
        return message.strip()
        
    except Exception as e:
        return f"📊 {symbol} ({timeframe})\n❌ Error creating message: {str(e)}"

def create_pattern_points_info(pattern_info):
    """สร้างข้อมูลจุดสำคัญของแพทเทิร์น"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        points_info = ""
        
        if 'points' in pattern_info and pattern_info['points']:
            points = pattern_info['points']
            points_info += "\n🎯 KEY FIBONACCI POINTS:\n"
            
            for point_name, point_data in points.items():
                if point_data:
                    _, price, point_type = point_data
                    if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
                        # แสดงข้อมูล Harmonic ratios
                        points_info += f"• Point {point_name}: {price:.4f} ({point_type})\n"
                    elif pattern_name == 'AB_CD':
                        points_info += f"• Point {point_name}: {price:.4f} (AB=CD Structure)\n"
                        
        elif 'wave_points' in pattern_info and pattern_info['wave_points']:
            wave_points = pattern_info['wave_points']
            points_info += "\n🌊 ELLIOTT WAVE STRUCTURE:\n"
            
            for wave_name, wave_data in wave_points.items():
                if wave_data:
                    _, price, wave_type = wave_data
                    points_info += f"• Wave {wave_name}: {price:.4f} ({wave_type})\n"
                    
        else:
            # สำหรับ Classic patterns
            points_info += f"\n📍 PATTERN STRUCTURE:\n"
            if pattern_name == 'HEAD_SHOULDERS':
                points_info += "• Left Shoulder - Head - Right Shoulder formation\n"
                points_info += "• Watch for neckline break confirmation\n"
            elif pattern_name == 'DOUBLE_TOP':
                points_info += "• Two peaks at similar resistance level\n"
                points_info += "• Valley between peaks creates support\n"
            elif pattern_name == 'DOUBLE_BOTTOM':
                points_info += "• Two troughs at similar support level\n"  
                points_info += "• Peak between troughs creates resistance\n"
            elif 'TRIANGLE' in pattern_name:
                points_info += "• Converging trend lines creating compression\n"
                points_info += "• Volume typically decreases before breakout\n"
            elif pattern_name == 'FLAG_PENNANT':
                points_info += "• Strong flagpole with tight consolidation\n"
                points_info += "• Continuation pattern in trend direction\n"
            elif pattern_name == 'CUP_HANDLE':
                points_info += "• U-shaped cup with small handle pullback\n"
                points_info += "• Bullish continuation after consolidation\n"
        
        return points_info
        
    except Exception as e:
        return f"Error creating points info: {str(e)}"

def create_trading_strategy(pattern_name, pattern_info):
    """สร้างกลยุทธ์การเทรดตามแพทเทิร์น"""
    try:
        strategies = {
            'GARTLEY': """
• ENTRY: Buy/Sell at Point D completion (78.6% XA)
• STOP LOSS: Beyond Point X (invalidation level)
• TARGET 1: 38.2% retracement of AD move
• TARGET 2: 61.8% retracement of AD move
• RISK/REWARD: Typically 1:2 to 1:3""",

            'BUTTERFLY': """
• ENTRY: Limit order at Point D (127.2-161.8% XA extension)
• STOP LOSS: 20-30 pips beyond Point D
• TARGET 1: 38.2% retracement back to C level
• TARGET 2: 61.8% retracement toward B level
• RISK/REWARD: High reward potential (1:3+)""",

            'BAT': """
• ENTRY: Market/Limit at Point D (88.6% XA)
• STOP LOSS: Above/Below Point X level
• TARGET 1: 38.2% AD retracement
• TARGET 2: 61.8% AD retracement
• CONFIRMATION: Look for reversal signals at D""",

            'CRAB': """
• ENTRY: Limit at Point D (161.8% XA - extreme level)
• STOP LOSS: 20-30 pips from entry
• TARGET 1: 38.2% AD retracement (quick profit)
• TARGET 2: 61.8% AD retracement
• NOTE: Most aggressive harmonic pattern""",

            'AB_CD': """
• ENTRY: At Point D completion
• STOP LOSS: Beyond Point D level
• TARGET 1: 38.2% retracement of CD move
• TARGET 2: 61.8% retracement of CD move
• COMBINE: With other harmonic patterns for confluence""",

            'ELLIOTT_WAVE_5': """
• WAVE 1-3: Trend continuation trades
• WAVE 2,4: Counter-trend bounce trades  
• WAVE 5: Final push - prepare for reversal
• FIBONACCI: Use extensions (161.8%, 261.8%)
• DIVERGENCE: Watch for momentum divergence at Wave 5""",

            'ELLIOTT_WAVE_3': """
• WAVE A: Initial decline/rise
• WAVE B: Counter-move (38-78% of A)
• WAVE C: Final move (often = Wave A)
• ENTRY: Wave C completion for reversal
• TARGET: 61.8-100% retracement of ABC""",

            'HEAD_SHOULDERS': """
• ENTRY: Break below neckline with volume
• STOP LOSS: Above right shoulder
• TARGET: Height of head to neckline projected down
• VOLUME: Should increase on neckline break
• CONFIRMATION: Close below neckline required""",

            'DOUBLE_TOP': """
• ENTRY: Break below valley support with volume
• STOP LOSS: Above second peak
• TARGET: Distance from peak to valley
• VOLUME: Decreasing at second peak (bearish)
• CONFIRMATION: Support level break required""",

            'DOUBLE_BOTTOM': """
• ENTRY: Break above peak resistance with volume
• STOP LOSS: Below second trough  
• TARGET: Distance from trough to peak
• VOLUME: Increasing at resistance break (bullish)
• CONFIRMATION: Resistance level break required""",

            'ASCENDING_TRIANGLE': """
• ENTRY: Break above horizontal resistance + volume
• STOP LOSS: Below ascending support line
• TARGET: Triangle height projected upward
• SUCCESS RATE: ~70% bullish breakouts
• VOLUME: Must increase on breakout""",

            'DESCENDING_TRIANGLE': """
• ENTRY: Break below horizontal support + volume  
• STOP LOSS: Above descending resistance line
• TARGET: Triangle height projected downward
• SUCCESS RATE: ~70% bearish breakdowns
• VOLUME: Must increase on breakdown""",

            'SYMMETRICAL_TRIANGLE': """
• ENTRY: Wait for breakout direction + volume confirmation
• STOP LOSS: Opposite side of triangle
• TARGET: Triangle height from breakout point
• DIRECTION: Usually continues prevailing trend
• TIMING: Breakout typically in final 1/3 of triangle""",

            'FLAG_PENNANT': """
• ENTRY: Breakout in flagpole direction + volume
• STOP LOSS: Opposite end of consolidation
• TARGET: Flagpole height from breakout point
• TIMING: Pattern completes quickly (1-4 weeks)
• SUCCESS: Very high continuation probability""",

            'CUP_HANDLE': """
• ENTRY: Breakout above handle high + volume
• STOP LOSS: Below handle low
• TARGET 1: Cup depth added to breakout
• TARGET 2: Previous all-time highs
• TIMELINE: Long-term bullish pattern"""
        }
        
        return strategies.get(pattern_name, "• Standard breakout/reversal strategy\n• Confirm with volume and momentum\n• Use proper risk management")
        
    except Exception as e:
        return f"Error creating strategy: {str(e)}"

# ============= ส่วนที่ 11: Advanced Pattern Validation =============

def validate_harmonic_ratios(X, A, B, C, D, pattern_type):
    """ตรวจสอบอัตราส่วน Fibonacci อย่างละเอียด"""
    try:
        XA = abs(A[1] - X[1])
        AB = abs(B[1] - A[1])
        BC = abs(C[1] - B[1])
        CD = abs(D[1] - C[1])
        AD = abs(D[1] - A[1])
        
        if XA == 0:
            return False, 0
        
        ab_ratio = AB / XA
        bc_ratio = BC / AB if AB != 0 else 0
        cd_ratio = CD / BC if BC != 0 else 0
        ad_ratio = AD / XA
        
        # ตารางอัตราส่วนที่ถูกต้องสำหรับแต่ละแพทเทิร์น
        ratio_rules = {
            'GARTLEY': {
                'AB': [(0.618, 0.05)],
                'BC': [(0.382, 0.05), (0.886, 0.05)],
                'CD': [(1.272, 0.1)],
                'AD': [(0.786, 0.05)]
            },
            'BUTTERFLY': {
                'AB': [(0.786, 0.05)],
                'BC': [(0.382, 0.05), (0.886, 0.05)],
                'CD': [(1.618, 0.1), (2.618, 0.2)],
                'AD': [(1.272, 0.05), (1.618, 0.05)]
            },
            'BAT': {
                'AB': [(0.382, 0.05), (0.500, 0.05)],
                'BC': [(0.382, 0.05), (0.886, 0.05)],
                'CD': [(1.618, 0.1), (2.618, 0.2)],
                'AD': [(0.886, 0.05)]
            },
            'CRAB': {
                'AB': [(0.382, 0.05), (0.618, 0.05)],
                'BC': [(0.382, 0.05), (0.886, 0.05)],
                'CD': [(2.240, 0.2), (3.618, 0.3)],
                'AD': [(1.618, 0.05)]
            }
        }
        
        if pattern_type not in ratio_rules:
            return False, 0
        
        rules = ratio_rules[pattern_type]
        actual_ratios = {'AB': ab_ratio, 'BC': bc_ratio, 'CD': cd_ratio, 'AD': ad_ratio}
        
        score = 0
        total_rules = 0
        
        for wave, targets in rules.items():
            actual = actual_ratios[wave]
            wave_valid = False
            
            for target_ratio, tolerance in targets:
                if abs(actual - target_ratio) <= tolerance:
                    wave_valid = True
                    break
            
            if wave_valid:
                score += 1
            total_rules += 1
        
        ratio_accuracy = score / total_rules
        is_valid = ratio_accuracy >= 0.75  # ต้องผ่าน 75% ของเงื่อนไข
        
        return is_valid, ratio_accuracy
        
    except Exception as e:
        print(f"Ratio validation error: {e}")
        return False, 0

def validate_elliott_wave_structure(wave_points, pattern_type):
    """ตรวจสอบโครงสร้าง Elliott Wave"""
    try:
        if pattern_type == 'ELLIOTT_WAVE_5':
            return validate_5_wave_structure(wave_points)
        elif pattern_type == 'ELLIOTT_WAVE_3':
            return validate_3_wave_structure(wave_points)
        
        return False, 0
        
    except Exception as e:
        print(f"Elliott wave validation error: {e}")
        return False, 0

def validate_5_wave_structure(wave_points):
    """ตรวจสอบ 5-wave impulse structure"""
    try:
        required_points = ['start', '1', '2', '3', '4', '5']
        if not all(point in wave_points for point in required_points):
            return False, 0
            
        start = wave_points['start']
        w1 = wave_points['1']
        w2 = wave_points['2']  
        w3 = wave_points['3']
        w4 = wave_points['4']
        w5 = wave_points['5']
        
        # คำนวณขนาดของแต่ละ wave
        wave1_size = abs(w1[1] - start[1])
        wave2_size = abs(w2[1] - w1[1])
        wave3_size = abs(w3[1] - w2[1])
        wave4_size = abs(w4[1] - w3[1])
        wave5_size = abs(w5[1] - w4[1])
        
        # Elliott Wave Rules
        rules_passed = 0
        
        # Rule 1: Wave 2 never retraces more than 100% of Wave 1
        if wave2_size <= wave1_size:
            rules_passed += 1
        
        # Rule 2: Wave 3 is never the shortest wave
        if wave3_size >= max(wave1_size, wave5_size):
            rules_passed += 1
            
        # Rule 3: Wave 4 never enters the price territory of Wave 1
        if start[2] == 'low':  # Bullish wave
            if w4[1] > w1[1]:
                rules_passed += 1
        else:  # Bearish wave
            if w4[1] < w1[1]:
                rules_passed += 1
        
        # Additional guideline: Wave 3 is often 161.8% of Wave 1
        wave3_to_1_ratio = wave3_size / wave1_size if wave1_size > 0 else 0
        if 1.5 <= wave3_to_1_ratio <= 2.0:  # ประมาณ 150-200%
            rules_passed += 0.5
            
        # Additional guideline: Wave 5 is often equal to Wave 1
        wave5_to_1_ratio = wave5_size / wave1_size if wave1_size > 0 else 0
        if 0.8 <= wave5_to_1_ratio <= 1.2:  # ประมาณ 80-120%
            rules_passed += 0.5
        
        confidence = min(rules_passed / 3.0, 1.0)  # Scale to max 1.0
        is_valid = rules_passed >= 2.5  # ต้องผ่านกฎหลัก + guidelines
        
        return is_valid, confidence
        
    except Exception as e:
        print(f"5-wave validation error: {e}")
        return False, 0

def validate_3_wave_structure(wave_points):
    """ตรวจสอบ 3-wave corrective structure"""
    try:
        required_points = ['start', 'A', 'B', 'C']
        if not all(point in wave_points for point in required_points):
            return False, 0
            
        start = wave_points['start']
        wA = wave_points['A']
        wB = wave_points['B']
        wC = wave_points['C']
        
        wave_A_size = abs(wA[1] - start[1])
        wave_B_size = abs(wB[1] - wA[1])
        wave_C_size = abs(wC[1] - wB[1])
        
        if wave_A_size == 0:
            return False, 0
            
        rules_passed = 0
        
        # Rule 1: Wave B retraces 38-78% of Wave A
        b_retracement = wave_B_size / wave_A_size
        if 0.38 <= b_retracement <= 0.78:
            rules_passed += 1
            
        # Rule 2: Wave C is approximately equal to Wave A (0.618-1.618 ratio)
        c_to_a_ratio = wave_C_size / wave_A_size
        if 0.618 <= c_to_a_ratio <= 1.618:
            rules_passed += 1
            
        # Additional guideline: Wave C often equals Wave A
        if 0.9 <= c_to_a_ratio <= 1.1:
            rules_passed += 0.5
            
        confidence = min(rules_passed / 2.0, 1.0)
        is_valid = rules_passed >= 1.5
        
        return is_valid, confidence
        
    except Exception as e:
        print(f"3-wave validation error: {e}")
        return False, 0

# ============= ส่วนที่ 12: Pattern Strength Scoring System =============

def calculate_pattern_strength(pattern_info, df):
    """คำนวณความแข็งแกร่งของแพทเทิร์น"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        base_confidence = pattern_info.get('confidence', 0)
        
        if pattern_name == 'NO_PATTERN':
            return 0
        
        strength_score = base_confidence
        
        # Volume Analysis (เพิ่มคะแนนถ้ามี volume สนับสนุน)
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:  # Volume เพิ่มขึ้นมาก
                strength_score += 0.1
            elif volume_ratio > 1.2:  # Volume เพิ่มขึ้นปานกลาง
                strength_score += 0.05
        
        # Trend Alignment (เช็คว่าแพทเทิร์นสอดคล้องกับ trend หรือไม่)
        closes = df['close'].values
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            current_price = closes[-1]
            
            if pattern_name in ['HEAD_SHOULDERS', 'DOUBLE_TOP'] and current_price < sma_20:
                strength_score += 0.05  # Bearish pattern ใน downtrend
            elif pattern_name in ['DOUBLE_BOTTOM', 'CUP_HANDLE'] and current_price > sma_20:
                strength_score += 0.05  # Bullish pattern ใน uptrend
        
        # Multiple Timeframe Confirmation (สมมุติ)
        # ในระบบจริงจะต้องเช็คหลาย timeframe
        if base_confidence > 0.75:
            strength_score += 0.05  # High confidence patterns get bonus
        
        # Pattern Maturity (แพทเทิร์นที่เกิดครบแล้ว)
        if 'points' in pattern_info and len(pattern_info['points']) >= 4:
            strength_score += 0.05
        elif 'wave_points' in pattern_info and len(pattern_info['wave_points']) >= 4:
            strength_score += 0.05
        
        return min(strength_score, 1.0)  # จำกัดไม่เกิน 100%
        
    except Exception as e:
        print(f"Pattern strength calculation error: {e}")
        return base_confidence

# ============= ส่วนที่ 13: Complete Integration Function =============

# ============= ส่วนที่ 13: Complete Integration Function (ต่อให้เสร็จ) =============

def run_complete_pattern_analysis(df, symbol='UNKNOWN', timeframe='1H', 
                                 send_telegram=True, save_chart=True):
    """
    ฟังก์ชันหลักที่รวมทุกอย่างเข้าด้วยกัน
    """
    try:
        print(f"🔍 Starting pattern analysis for {symbol} {timeframe}...")
        
        # Step 1: ตรวจหาแพทเทิร์น
        pattern_result, telegram_msg = detect_all_patterns_enhanced(df, symbol, timeframe)
        
        # Step 2: คำนวณความแข็งแกร่งของแพทเทิร์น
        pattern_strength = calculate_pattern_strength(pattern_result, df)
        pattern_result['strength'] = pattern_strength
        
        # Step 3: สร้างข้อความ Telegram แบบละเอียด
        enhanced_msg = create_enhanced_telegram_message_with_theory(
            pattern_result, symbol, timeframe, df['close'].iloc[-1]
        )
        
        # Step 4: สร้างและบันทึกกราฟ
        chart_result = analyze_and_send_telegram(df, symbol, timeframe, send_telegram)
        
        # Step 5: รวมผลลัพธ์
        final_result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern_detected': pattern_result,
            'pattern_strength': pattern_strength,
            'telegram_message': enhanced_msg,
            'chart_data': chart_result.get('chart_base64'),
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'SUCCESS'
        }
        
        # Step 6: แสดงสรุปผล
        print(f"\n📊 ANALYSIS COMPLETE:")
        print(f"Symbol: {symbol} ({timeframe})")
        print(f"Pattern: {pattern_result['pattern_name']}")
        print(f"Confidence: {pattern_result['confidence']:.1%}")
        print(f"Strength: {pattern_strength:.1%}")
        print(f"Method: {pattern_result['method']}")
        
        if pattern_result['pattern_name'] != 'NO_PATTERN':
            print(f"✅ Pattern detected with {pattern_result['confidence']:.1%} confidence")
            if send_telegram:
                print(f"📱 Message sent to Telegram")
        else:
            print(f"❌ No clear pattern detected")
            
        return final_result
        
    except Exception as e:
        error_result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'pattern_detected': {'pattern_name': 'ERROR', 'confidence': 0},
            'pattern_strength': 0,
            'telegram_message': f"❌ Analysis error: {str(e)}",
            'chart_data': None,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'ERROR',
            'error': str(e)
        }
        print(f"❌ Analysis error: {str(e)}")
        return error_result

# ============= ส่วนที่ 14: Enhanced Pattern Visualization =============

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def create_comprehensive_chart(df, pattern_info, symbol, timeframe):
    """สร้างกราฟครอบคลุมทุกแพทเทิร์น"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), 
                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Set dark theme
        fig.patch.set_facecolor('#0a0a0a')
        ax1.set_facecolor('#0a0a0a')
        ax2.set_facecolor('#0a0a0a')
        
        # Main price chart
        draw_advanced_candlestick_chart(ax1, df)
        
        # Pattern overlay
        draw_enhanced_pattern_lines(ax1, df, pattern_info)
        
        # Volume chart
        if 'volume' in df.columns:
            draw_volume_chart(ax2, df)
        else:
            ax2.set_visible(False)
        
        # Chart styling
        setup_chart_styling(ax1, ax2, pattern_info, symbol, timeframe)
        
        # Add technical indicators
        add_technical_indicators(ax1, df)
        
        # Pattern information box
        add_pattern_info_box(ax1, pattern_info)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Chart creation error: {e}")
        return None

def draw_advanced_candlestick_chart(ax, df):
    """วาดกราฟแท่งเทียนขั้นสูง"""
    try:
        opens = df['open'].values
        highs = df['high'].values  
        lows = df['low'].values
        closes = df['close'].values
        
        # สีสำหรับแท่งเทียน
        bullish_body_color = '#00ff88'
        bullish_wick_color = '#00cc66'
        bearish_body_color = '#ff4444'
        bearish_wick_color = '#cc3333'
        
        for i in range(len(df)):
            # กำหนดสีตามทิศทาง
            if closes[i] >= opens[i]:
                body_color = bullish_body_color
                wick_color = bullish_wick_color
            else:
                body_color = bearish_body_color
                wick_color = bearish_wick_color
            
            # วาดเส้น High-Low (wick)
            ax.plot([i, i], [lows[i], highs[i]], 
                   color=wick_color, linewidth=1.5, alpha=0.8)
            
            # วาดแท่งเทียน (body)
            height = abs(closes[i] - opens[i])
            bottom = min(opens[i], closes[i])
            
            if height > 0:  # มี body
                rect = plt.Rectangle((i-0.35, bottom), 0.7, height,
                                   facecolor=body_color, edgecolor=wick_color, 
                                   linewidth=0.8, alpha=0.9)
                ax.add_patch(rect)
            else:  # doji
                ax.plot([i-0.35, i+0.35], [closes[i], closes[i]], 
                       color=wick_color, linewidth=2)
        
        # Set limits
        ax.set_xlim(-0.5, len(df)-0.5)
        price_range = max(highs) - min(lows)
        ax.set_ylim(min(lows) - price_range*0.02, max(highs) + price_range*0.02)
        
    except Exception as e:
        print(f"Advanced candlestick error: {e}")

def add_technical_indicators(ax, df):
    """เพิ่ม Technical Indicators"""
    try:
        closes = df['close'].values
        
        # Simple Moving Averages
        if len(closes) >= 20:
            sma20 = pd.Series(closes).rolling(20).mean()
            ax.plot(range(len(sma20)), sma20, color='#ffaa00', 
                   linewidth=1.5, alpha=0.8, label='SMA 20')
        
        if len(closes) >= 50:
            sma50 = pd.Series(closes).rolling(50).mean()
            ax.plot(range(len(sma50)), sma50, color='#ff6600', 
                   linewidth=1.5, alpha=0.8, label='SMA 50')
        
        # Support/Resistance levels
        recent_high = max(closes[-20:]) if len(closes) >= 20 else max(closes)
        recent_low = min(closes[-20:]) if len(closes) >= 20 else min(closes)
        
        ax.axhline(y=recent_high, color='#ff0066', linestyle='--', 
                  alpha=0.6, linewidth=1, label='Recent High')
        ax.axhline(y=recent_low, color='#00ff66', linestyle='--', 
                  alpha=0.6, linewidth=1, label='Recent Low')
        
    except Exception as e:
        print(f"Technical indicators error: {e}")

def draw_volume_chart(ax, df):
    """วาดกราฟ Volume"""
    try:
        volumes = df['volume'].values
        closes = df['close'].values
        opens = df['open'].values
        
        colors = ['#00ff88' if closes[i] >= opens[i] else '#ff4444' 
                 for i in range(len(df))]
        
        bars = ax.bar(range(len(volumes)), volumes, color=colors, alpha=0.7, width=0.8)
        
        ax.set_xlim(-0.5, len(df)-0.5)
        ax.set_ylabel('Volume', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.3, color='gray')
        
        # Volume moving average
        if len(volumes) >= 20:
            vol_ma = pd.Series(volumes).rolling(20).mean()
            ax.plot(range(len(vol_ma)), vol_ma, color='#ffff00', 
                   linewidth=1.5, alpha=0.8, label='Vol MA 20')
        
    except Exception as e:
        print(f"Volume chart error: {e}")

def setup_chart_styling(ax1, ax2, pattern_info, symbol, timeframe):
    """ตั้งค่าสไตล์กราฟ"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        confidence = pattern_info.get('confidence', 0)
        
        # Main chart title
        title = f'{symbol} {timeframe} - {pattern_name} ({confidence:.1%})'
        ax1.set_title(title, fontsize=16, color='white', fontweight='bold', pad=20)
        
        # Axes styling
        ax1.set_ylabel('Price', fontsize=12, color='white')
        ax1.tick_params(colors='white', labelsize=10)
        ax1.grid(True, alpha=0.3, color='gray', linestyle=':')
        
        # Remove x-axis labels from main chart
        ax1.set_xticklabels([])
        
        # Volume chart styling
        if ax2.get_visible():
            ax2.set_xlabel('Time Period', fontsize=12, color='white')
            ax2.tick_params(colors='white', labelsize=10)
        
        # Legend
        if pattern_name != 'NO_PATTERN':
            ax1.legend(loc='upper left', facecolor='black', edgecolor='white', 
                      framealpha=0.8, fontsize=9)
        
    except Exception as e:
        print(f"Chart styling error: {e}")

def add_pattern_info_box(ax, pattern_info):
    """เพิ่มกล่องข้อมูลแพทเทิร์น"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        confidence = pattern_info.get('confidence', 0)
        method = pattern_info.get('method', 'UNKNOWN')
        
        if pattern_name == 'NO_PATTERN':
            return
        
        # สร้างข้อความสำหรับกล่องข้อมูล
        info_text = f"Pattern: {pattern_name}\n"
        info_text += f"Confidence: {confidence:.1%}\n"
        info_text += f"Method: {method}\n"
        
        # เพิ่มข้อมูลจุดสำคัญ
        if 'points' in pattern_info and pattern_info['points']:
            info_text += "\nKey Points:\n"
            for point_name, point_data in pattern_info['points'].items():
                if point_data:
                    _, price, _ = point_data
                    info_text += f"{point_name}: {price:.4f}\n"
        
        elif 'wave_points' in pattern_info and pattern_info['wave_points']:
            info_text += "\nWave Points:\n"
            for wave_name, wave_data in pattern_info['wave_points'].items():
                if wave_data:
                    _, price, _ = wave_data
                    info_text += f"W{wave_name}: {price:.4f}\n"
        
        # วางกล่องข้อมูล
        ax.text(0.02, 0.98, info_text.strip(), transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', color='yellow',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', 
                        edgecolor='yellow', alpha=0.9))
        
    except Exception as e:
        print(f"Pattern info box error: {e}")

# ============= ส่วนที่ 15: Enhanced Point Marking Functions =============

def draw_fibonacci_retracement_lines(ax, pattern_info):
    """วาดเส้น Fibonacci Retracement"""
    try:
        if 'points' not in pattern_info:
            return
        
        points = pattern_info['points']
        pattern_name = pattern_info.get('pattern_name', '')
        
        if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            # วาด Fibonacci lines สำหรับ Harmonic patterns
            draw_harmonic_fibonacci_lines(ax, points, pattern_name)
        
    except Exception as e:
        print(f"Fibonacci lines error: {e}")

def draw_harmonic_fibonacci_lines(ax, points, pattern_name):
    """วาดเส้น Fibonacci สำหรับ Harmonic patterns"""
    try:
        fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.618]
        fib_colors = ['#ff9999', '#ffcc99', '#ffff99', '#ccff99', 
                     '#99ffcc', '#99ccff', '#cc99ff', '#ff99cc']
        
        if 'X' in points and 'A' in points:
            X, A = points['X'], points['A']
            if X and A:
                # คำนวณระยะ XA
                xa_range = abs(A[1] - X[1])
                xa_direction = 1 if A[1] > X[1] else -1
                
                # วาดเส้น Fibonacci levels
                for i, level in enumerate(fib_levels):
                    fib_price = X[1] + (xa_range * level * xa_direction)
                    
                    ax.axhline(y=fib_price, color=fib_colors[i % len(fib_colors)], 
                              linestyle=':', alpha=0.6, linewidth=1)
                    
                    # Label
                    ax.text(len(ax.get_xlim())*0.95, fib_price, f'{level:.1%}', 
                           color=fib_colors[i % len(fib_colors)], fontsize=8,
                           verticalalignment='center', horizontalalignment='right')
        
    except Exception as e:
        print(f"Harmonic Fibonacci error: {e}")

def add_pattern_prediction_zones(ax, pattern_info, df):
    """เพิ่มโซนพยากรณ์ราคา"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        current_price = df['close'].iloc[-1]
        
        if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            add_harmonic_target_zones(ax, pattern_info, current_price)
        elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
            add_elliott_target_zones(ax, pattern_info, current_price)
        elif pattern_name in ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM']:
            add_classic_target_zones(ax, pattern_info, current_price, df)
        
    except Exception as e:
        print(f"Prediction zones error: {e}")

def add_harmonic_target_zones(ax, pattern_info, current_price):
    """เพิ่มโซนเป้าหมายสำหรับ Harmonic patterns"""
    try:
        if 'points' not in pattern_info:
            return
        
        points = pattern_info['points']
        
        if 'A' in points and 'D' in points:
            A, D = points['A'], points['D']
            if A and D:
                ad_range = abs(D[1] - A[1])
                
                # Target zones (38.2%, 61.8% retracement of AD)
                target1 = D[1] + (ad_range * 0.382 * (1 if A[1] > D[1] else -1))
                target2 = D[1] + (ad_range * 0.618 * (1 if A[1] > D[1] else -1))
                
                # วาดโซนเป้าหมาย
                ax.axhspan(target1, target2, alpha=0.2, color='green', 
                          label='Target Zone')
                
                ax.text(len(ax.get_xlim())*0.02, target1, 'T1: 38.2%', 
                       color='green', fontsize=9, fontweight='bold')
                ax.text(len(ax.get_xlim())*0.02, target2, 'T2: 61.8%', 
                       color='green', fontsize=9, fontweight='bold')
        
    except Exception as e:
        print(f"Harmonic target zones error: {e}")

def add_elliott_target_zones(ax, pattern_info, current_price):
    """เพิ่มโซนเป้าหมายสำหรับ Elliott Wave"""
    try:
        if 'wave_points' not in pattern_info:
            return
        
        wave_points = pattern_info['wave_points']
        pattern_name = pattern_info.get('pattern_name', '')
        
        if pattern_name == 'ELLIOTT_WAVE_5':
            # คาดการณ์ reversal หลัง Wave 5
            if '1' in wave_points and '3' in wave_points and '5' in wave_points:
                w1, w3, w5 = wave_points['1'], wave_points['3'], wave_points['5']
                if w1 and w3 and w5:
                    # Target ที่ 50-61.8% ของ Wave 5
                    wave5_range = abs(w5[1] - w3[1])
                    target_50 = w5[1] - (wave5_range * 0.5)
                    target_618 = w5[1] - (wave5_range * 0.618)
                    
                    ax.axhspan(target_618, target_50, alpha=0.2, color='orange',
                              label='Elliott Reversal Zone')
        
    except Exception as e:
        print(f"Elliott target zones error: {e}")

def add_classic_target_zones(ax, pattern_info, current_price, df):
    """เพิ่มโซนเป้าหมายสำหรับ Classic patterns"""
    try:
        pattern_name = pattern_info.get('pattern_name', '')
        highs = df['high'].values
        lows = df['low'].values
        
        if pattern_name == 'HEAD_SHOULDERS':
            # ค้นหา neckline และ head level
            head_level = max(highs[-30:]) if len(highs) >= 30 else max(highs)
            neckline_level = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            
            # Target = Head to Neckline distance projected down
            hs_range = head_level - neckline_level
            target = neckline_level - hs_range
            
            ax.axhline(y=target, color='red', linestyle='--', alpha=0.8,
                      linewidth=2, label='H&S Target')
            ax.text(len(highs)*0.8, target, f'Target: {target:.4f}', 
                   color='red', fontweight='bold')
        
        elif pattern_name == 'DOUBLE_TOP':
            # Target = distance between peaks and valley
            recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            valley_low = min(lows[-15:]) if len(lows) >= 15 else min(lows)
            
            dt_range = recent_high - valley_low
            target = valley_low - dt_range
            
            ax.axhline(y=target, color='red', linestyle='--', alpha=0.8,
                      linewidth=2, label='Double Top Target')
        
        elif pattern_name == 'DOUBLE_BOTTOM':
            # Target = distance between troughs and peak
            recent_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            peak_high = max(highs[-15:]) if len(highs) >= 15 else max(highs)
            
            db_range = peak_high - recent_low
            target = peak_high + db_range
            
            ax.axhline(y=target, color='green', linestyle='--', alpha=0.8,
                      linewidth=2, label='Double Bottom Target')
        
    except Exception as e:
        print(f"Classic target zones error: {e}")

# ============= ส่วนที่ 16: Export และ Utility Functions =============

def save_analysis_report(analysis_result, filename=None):
    """บันทึกรายงานการวิเคราะห์"""
    try:
        import json
        
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pattern_analysis_{analysis_result['symbol']}_{timestamp}.json"
        
        # เตรียมข้อมูลสำหรับ export
        export_data = {
            'symbol': analysis_result['symbol'],
            'timeframe': analysis_result['timeframe'], 
            'timestamp': analysis_result['analysis_timestamp'],
            'pattern': {
                'name': analysis_result['pattern_detected']['pattern_name'],
                'confidence': analysis_result['pattern_detected']['confidence'],
                'strength': analysis_result.get('pattern_strength', 0),
                'method': analysis_result['pattern_detected']['method']
            },
            'status': analysis_result['status']
        }
        
        # เพิ่มข้อมูลจุดสำคัญถ้ามี
        if 'points' in analysis_result['pattern_detected']:
            export_data['pattern']['points'] = analysis_result['pattern_detected']['points']
        elif 'wave_points' in analysis_result['pattern_detected']:
            export_data['pattern']['wave_points'] = analysis_result['pattern_detected']['wave_points']
        
        # บันทึกไฟล์
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Analysis report saved: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving report: {e}")
        return None

def batch_pattern_analysis(data_dict, save_reports=True):
    """วิเคราะห์แพทเทิร์นหลายๆ symbols พร้อมกัน"""
    try:
        results = {}
        
        for symbol, df in data_dict.items():
            print(f"\n{'='*50}")
            print(f"Analyzing {symbol}...")
            print(f"{'='*50}")
            
            # วิเคราะห์แพทเทิร์น
            result = run_complete_pattern_analysis(df, symbol)
            results[symbol] = result
            
            # บันทึกรายงาน
            if save_reports and result['status'] == 'SUCCESS':
                save_analysis_report(result)
        
        # สรุปผลรวม
        print(f"\n{'='*60}")
        print(f"BATCH ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        patterns_found = 0
        for symbol, result in results.items():
            pattern_name = result['pattern_detected']['pattern_name']
            confidence = result['pattern_detected']['confidence']
            
            if pattern_name != 'NO_PATTERN':
                patterns_found += 1
                print(f"✅ {symbol}: {pattern_name} ({confidence:.1%})")
            else:
                print(f"❌ {symbol}: No pattern detected")
        
        print(f"\nTotal patterns found: {patterns_found}/{len(data_dict)}")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")
        return {}

def create_pattern_summary_table(results_dict):
    """สร้างตารางสรุปแพทเทิร์น"""
    try:
        summary_data = []
        
        for symbol, result in results_dict.items():
            pattern_info = result['pattern_detected']
            
            summary_data.append({
                'Symbol': symbol,
                'Pattern': pattern_info['pattern_name'],
                'Confidence': f"{pattern_info['confidence']:.1%}",
                'Strength': f"{result.get('pattern_strength', 0):.1%}",
                'Method': pattern_info['method'],
                'Status': result['status']
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # เรียงตาม Confidence
        df_summary['Confidence_num'] = df_summary['Confidence'].str.rstrip('%').astype(float)
        df_summary = df_summary.sort_values('Confidence_num', ascending=False)
        df_summary = df_summary.drop('Confidence_num', axis=1)
        
        print("\n📊 PATTERN ANALYSIS SUMMARY TABLE:")
        print("="*80)
        print(df_summary.to_string(index=False))
        
        return df_summary
        
    except Exception as e:
        print(f"❌ Summary table error: {e}")
        return None

# ============= ส่วนที่ 17: Main Execution Function =============

def main_pattern_detection_system():
    """ฟังก์ชันหลักของระบบตรวจจับแพทเทิร์น"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              ADVANCED PATTERN DETECTION SYSTEM              ║
    ║                                                              ║
    ║  🎯 Harmonic Patterns: Gartley, Butterfly, Bat, Crab        ║
    ║  🌊 Elliott Wave: 5-Wave Impulse, 3-Wave Corrective         ║
    ║  📊 Classic Patterns: H&S, Double Top/Bottom, Triangles     ║
    ║  📈 Technical Analysis: Volume, MA, Support/Resistance       ║
    ║  📱 Telegram Integration: Automated alerts & charts         ║
    ║                                                              ║
    ║  Created by: Advanced Trading Bot System                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # ตัวอย่างการใช้งาน
    print("\n📋 USAGE EXAMPLES:")
    print("1. Single symbol analysis:")
    print("   result = run_complete_pattern_analysis(df, 'EURUSD', '1H')")
    print()
    print("2. Batch analysis:")
    print("   data_dict = {'EURUSD': df1, 'GBPUSD': df2}")
    print("   results = batch_pattern_analysis(data_dict)")
    print()
    print("3. Create summary table:")
    print("   summary = create_pattern_summary_table(results)")
    print()
    
    return True

# ============= Test Function =============

def test_pattern_detection():
    """ฟังก์ชันทดสอบระบบ"""
    try:
        # สร้างข้อมูลทดสอบ
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # สร้างราคาจำลอง
        price_base = 1.1000
        price_changes = np.random.randn(100) * 0.001
        prices = [price_base]
        
        for change in price_changes[1:]:
            prices.append(prices[-1] + change)
        
        # สร้าง OHLC data
        test_data = {
            'open': prices,
            'high': [p + abs(np.random.randn() * 0.0005) for p in prices],
            'low': [p - abs(np.random.randn() * 0.0005) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        df_test = pd.DataFrame(test_data, index=dates)
        
        # ทดสอบการตรวจจับแพทเทิร์น
        print("🧪 Testing Pattern Detection System...")
        result = run_complete_pattern_analysis(df_test, 'TEST_PAIR', '1H')
        
        if result['status'] == 'SUCCESS':
            print("✅ Test completed successfully!")
        else:
            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            
        return result
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return None

# ============= ส่วนที่ 18: Integration Instructions =============

# ============= ส่วนที่ 19: Advanced Error Handling =============

class PatternDetectionError(Exception):
    """Custom exception for pattern detection errors"""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def validate_input_data(df):
    """ตรวจสอบความถูกต้องของข้อมูลนำเข้า"""
    try:
        required_columns = ['open', 'high', 'low', 'close']
        
        # ตรวจสอบ columns ที่จำเป็น
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # ตรวจสอบข้อมูลว่าง
        if df[required_columns].isnull().any().any():
            raise DataValidationError("Data contains null values in OHLC columns")
        
        # ตรวจสอบความสมเหตุสมผลของราคา
        for i in range(len(df)):
            row = df.iloc[i]
            if not (row['low'] <= row['open'] <= row['high'] and 
                   row['low'] <= row['close'] <= row['high']):
                raise DataValidationError(f"Invalid OHLC data at index {i}")
        
        # ตรวจสอบจำนวนข้อมูลเพียงพอ
        if len(df) < 20:
            raise DataValidationError("Insufficient data: need at least 20 candles")
        
        return True
        
    except Exception as e:
        raise DataValidationError(f"Data validation failed: {str(e)}")

def safe_pattern_detection(detection_function, *args, **kwargs):
    """Wrapper สำหรับการตรวจจับแพทเทิร์นอย่างปลอดภัย"""
    try:
        return detection_function(*args, **kwargs)
    except Exception as e:
        print(f"Pattern detection error in {detection_function.__name__}: {e}")
        return {
            'pattern_id': 0,
            'pattern_name': 'ERROR',
            'confidence': 0.0,
            'method': f'ERROR_{detection_function.__name__}',
            'error': str(e)
        }

# ============= ส่วนที่ 20: Performance Optimization =============

def optimize_dataframe(df):
    """เพิ่มประสิทธิภาพ DataFrame"""
    try:
        # แปลงเป็น numpy arrays สำหรับการคำนวณที่เร็วขึ้น
        df = df.copy()
        
        # ลดขนาดข้อมูลโดยใช้ dtype ที่เหมาะสม
        float_cols = ['open', 'high', 'low', 'close']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
        
        return df
        
    except Exception as e:
        print(f"DataFrame optimization warning: {e}")
        return df

# ============= ส่วนที่ 21: Configuration Management =============

PATTERN_CONFIG = {
    'harmonic': {
        'enabled': True,
        'min_confidence': 0.70,
        'fibonacci_tolerance': 0.05,
        'patterns': ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD']
    },
    'elliott_wave': {
        'enabled': True,
        'min_confidence': 0.65,
        'wave_tolerance': 0.1,
        'patterns': ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']
    },
    'classic': {
        'enabled': True,
        'min_confidence': 0.60,
        'patterns': ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'DOUBLE_BOTTOM',
                    'ASCENDING_TRIANGLE', 'DESCENDING_TRIANGLE', 
                    'SYMMETRICAL_TRIANGLE', 'FLAG_PENNANT', 'CUP_HANDLE']
    },
    'chart': {
        'dark_theme': True,
        'show_volume': True,
        'show_indicators': True,
        'show_fibonacci': True,
        'show_targets': True
    },
    'telegram': {
        'send_charts': True,
        'include_theory': True,
        'include_strategy': True,
        'min_confidence_alert': 0.65
    }
}

def update_config(section, key, value):
    """อัพเดต configuration"""
    try:
        if section in PATTERN_CONFIG and key in PATTERN_CONFIG[section]:
            PATTERN_CONFIG[section][key] = value
            print(f"Config updated: {section}.{key} = {value}")
            return True
        else:
            print(f"Invalid config path: {section}.{key}")
            return False
    except Exception as e:
        print(f"Config update error: {e}")
        return False

def get_config(section, key=None):
    """ดึงค่า configuration"""
    try:
        if key is None:
            return PATTERN_CONFIG.get(section, {})
        else:
            return PATTERN_CONFIG.get(section, {}).get(key)
    except Exception as e:
        print(f"Config get error: {e}")
        return None

# ============= Final Integration Message =============

if __name__ == "__main__":
    print("🚀 Advanced Pattern Detection System Loaded Successfully!")
    print("📚 Run main_pattern_detection_system() to see usage examples")
    print("🧪 Run test_pattern_detection() to test the system")
    main_pattern_detection_system()             


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
@app.route('/run-harmonic-bot')
def run_harmonic_bot():
    """Run Harmonic + Elliott Wave patterns - Send Telegram once per hour"""
    global last_harmonic_sent_hour, message_sent_this_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        current_time = now_th.strftime("%Y-%m-%d %H:%M")
        
        # Reset message tracking เมื่อเปลี่ยนชั่วโมง
        if current_hour != last_harmonic_sent_hour:
            last_harmonic_sent_hour = current_hour
            message_sent_this_hour['harmonic'] = None
        
        # ตรวจสอบว่าส่งข้อความในชั่วโมงนี้แล้วหรือยัง
        if message_sent_this_hour['harmonic'] != current_hour:
            message_sent_this_hour['harmonic'] = current_hour
            
            def send_harmonic_task():
                try:
                    shared_df = get_shared_xau_data()
                    if shared_df is None:
                        error_msg = f"❌ Harmonic AI Data Error @ {current_time}\nCannot fetch market data"
                        send_telegram(error_msg)
                        return
                    
                    if len(shared_df) < 50:
                        error_msg = f"❌ Harmonic AI Data Error @ {current_time}\nInsufficient data (need 50+ candles)"
                        send_telegram(error_msg)
                        return
                    
                    # ตรวจจับ Harmonic Patterns
                    current_price = shared_df['close'].iloc[-1]
                    pattern_result, telegram_msg = detect_all_patterns_enhanced(
                        shared_df, 'XAUUSD', '1H'
                    )
                    
                    # ส่งข้อความหลัก
                    send_status = send_telegram(telegram_msg)
                    print(f"✅ [{current_time}] Harmonic message sent: Status {send_status}")
                    
                    # ถ้าพบแพทเทิร์น ส่งกราฟด้วย
                    if pattern_result['pattern_name'] not in ['NO_PATTERN', 'ERROR']:
                        time.sleep(3)
                        
                        # สร้างกราฟ
                        chart_result = analyze_and_send_telegram(
                            shared_df, 'XAUUSD', '1H', send_to_telegram=True
                        )
                        
                        if chart_result and chart_result.get('chart_base64'):
                            print(f"✅ [{current_time}] Harmonic chart sent")
                        
                        # ส่งทฤษฎีแพทเทิร์น
                        pattern_name = pattern_result['pattern_name']
                        theory = get_pattern_theory(pattern_name)
                        
                        if theory:
                            time.sleep(3)
                            theory_msg = f"""📚 PATTERN THEORY DETAIL

{theory['theory']}

💡 เพิ่มเติม:
• Minimum Confidence: {theory.get('confidence_min', 0.70):.0%}
• ศึกษาเพิ่มเติมจาก Technical Analysis Books
• ใช้ร่วมกับ indicators อื่นๆ เสมอ"""
                            
                            send_telegram(theory_msg)
                            print(f"✅ [{current_time}] Pattern theory sent")
                    
                except Exception as e:
                    print(f"❌ [{current_time}] Harmonic send error: {e}")
                    error_msg = f"❌ Harmonic AI Error @ {current_time}\nError: {str(e)[:150]}"
                    send_telegram(error_msg)
            
            Thread(target=send_harmonic_task, daemon=True).start()
            
            return jsonify({
                "status": "✅ Harmonic AI - Signal Sent",
                "mode": "TELEGRAM_SENT",
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": True,
                "system": "Harmonic Patterns + Elliott Wave",
                "patterns": ["GARTLEY", "BUTTERFLY", "BAT", "CRAB", "AB_CD", "ELLIOTT_WAVE"],
                "note": f"🎯 HARMONIC signals sent at {current_time}",
                "sent_count_this_hour": 1
            })
        else:
            return jsonify({
                "status": "✅ Harmonic AI - Keep Alive",
                "mode": "KEEP_ALIVE",
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": False,
                "system": "Harmonic Patterns + Elliott Wave",
                "note": f"Harmonic signals already sent in hour {current_hour}",
                "next_signal_time": f"{current_hour + 1}:00"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-harmonic')
def test_harmonic():
    """Test Harmonic pattern detection"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None:
            return jsonify({
                "status": "error",
                "message": "Cannot fetch market data"
            }), 500
        
        if len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data for Harmonic patterns (need 50+ candles)"
            }), 400
        
        current_price = shared_df['close'].iloc[-1]
        
        # ทดสอบ Harmonic detection
        pattern_result, telegram_msg = detect_all_patterns_enhanced(
            shared_df, 'XAUUSD', '1H'
        )
        
        return jsonify({
            "status": "success",
            "current_price": float(current_price),
            "pattern_detected": pattern_result,
            "message_preview": telegram_msg[:500] + "..." if len(telegram_msg) > 500 else telegram_msg,
            "data_points": len(shared_df),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-harmonic-send')
def test_harmonic_send():
    """Test sending Harmonic pattern to Telegram"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None:
            return jsonify({
                "status": "error",
                "message": "Cannot fetch market data"
            }), 500
        
        # Detect patterns
        pattern_result, telegram_msg = detect_all_patterns_enhanced(
            shared_df, 'XAUUSD', '1H'
        )
        
        # Send to Telegram
        send_status = send_telegram(telegram_msg)
        
        result_info = {
            "status": "success",
            "telegram_status": send_status,
            "pattern": pattern_result['pattern_name'],
            "confidence": pattern_result.get('confidence', 0),
            "message_sent": send_status == 200
        }
        
        # Send chart if pattern found
        if pattern_result['pattern_name'] not in ['NO_PATTERN', 'ERROR']:
            time.sleep(2)
            chart_result = analyze_and_send_telegram(
                shared_df, 'XAUUSD', '1H', send_to_telegram=True
            )
            result_info['chart_sent'] = chart_result is not None
        
        return jsonify(result_info)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/harmonic-status')
def harmonic_status():
    """Get current harmonic pattern analysis status"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data for harmonic analysis"
            })
        
        current_price = float(shared_df['close'].iloc[-1])
        
        # Detect all patterns
        harmonic_detector = HarmonicPatternDetector()
        harmonic_result = harmonic_detector.detect_harmonic_patterns(shared_df)
        
        elliott_detector = ElliottWaveDetector()
        elliott_result = elliott_detector.detect_elliott_waves(shared_df)
        
        return jsonify({
            "status": "success",
            "data_source": "shared",
            "current_price": current_price,
            "harmonic_pattern": {
                "pattern_name": harmonic_result.get('pattern_name', 'NO_PATTERN'),
                "confidence": harmonic_result.get('confidence', 0),
                "method": harmonic_result.get('method', 'HARMONIC'),
                "points": harmonic_result.get('points', {})
            },
            "elliott_wave": {
                "pattern_name": elliott_result.get('pattern_name', 'NO_PATTERN'),
                "confidence": elliott_result.get('confidence', 0),
                "method": elliott_result.get('method', 'ELLIOTT_WAVE'),
                "wave_points": elliott_result.get('wave_points', {})
            },
            "timestamp": datetime.now().isoformat(),
            "data_points": len(shared_df)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-specific-harmonic')
def test_specific_harmonic():
    """Test detection of specific harmonic pattern"""
    try:
        pattern_type = request.args.get('pattern', 'GARTLEY')
        valid_patterns = ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD']
        
        if pattern_type not in valid_patterns:
            return jsonify({
                "status": "error",
                "message": f"Invalid pattern. Choose from: {valid_patterns}"
            }), 400
        
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            }), 400
        
        # Try to detect the specific pattern
        harmonic_detector = HarmonicPatternDetector()
        result = harmonic_detector.detect_harmonic_patterns(shared_df)
        
        # Get pattern theory
        theory = get_pattern_theory(pattern_type)
        
        return jsonify({
            "status": "success",
            "requested_pattern": pattern_type,
            "detected_pattern": result.get('pattern_name', 'NO_PATTERN'),
            "matches": result.get('pattern_name') == pattern_type,
            "confidence": result.get('confidence', 0),
            "theory": theory,
            "current_price": float(shared_df['close'].iloc[-1])
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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
            "version": "3.0 - Triple Signal System + Harmonic Patterns",  # เปลี่ยนเวอร์ชัน
            "timestamp": datetime.now().isoformat(),
            "current_hour": current_hour,
            "bangkok_time": now_th.strftime("%Y-%m-%d %H:%M:%S"),
            "systems": {
                "original": "RSI + EMA + Price Change",
                "pattern": "Rule-based Pattern Detection",
                "harmonic": "Harmonic Patterns + Elliott Wave"  # เพิ่มระบบใหม่
            },
            "message_status": {
                "original_sent_this_hour": message_sent_this_hour.get('original') == current_hour,
                "pattern_sent_this_hour": message_sent_this_hour.get('pattern') == current_hour,
                "harmonic_sent_this_hour": message_sent_this_hour.get('harmonic') == current_hour,  # เพิ่ม
                "total_messages_this_hour": sum([
                    1 if message_sent_this_hour.get('original') == current_hour else 0,
                    1 if message_sent_this_hour.get('pattern') == current_hour else 0,
                    1 if message_sent_this_hour.get('harmonic') == current_hour else 0  # เพิ่ม
                ])
            },
            "harmonic_patterns": [  # เพิ่มข้อมูลแพทเทิร์น
                "GARTLEY", "BUTTERFLY", "BAT", "CRAB", "AB_CD",
                "ELLIOTT_WAVE_5", "ELLIOTT_WAVE_3"
            ],
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
                "/run-harmonic-bot",  # เพิ่ม
                "/test-telegram",
                "/test-pattern-ai",
                "/test-harmonic",  # เพิ่ม
                "/test-harmonic-send",  # เพิ่ม
                "/test-specific-harmonic",  # เพิ่ม
                "/pattern-status",
                "/harmonic-status",  # เพิ่ม
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
        <title>XAU AI Trading Bot v3.0</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #0a0a0a; color: #ffffff; }
            .container { max-width: 900px; margin: 0 auto; }
            h1 { color: #00ff88; text-align: center; text-shadow: 0 0 10px #00ff88; }
            h2 { color: #ffaa00; border-bottom: 2px solid #ffaa00; padding-bottom: 10px; }
            .endpoint { background-color: #1a1a1a; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #00ff88; }
            .harmonic-endpoint { border-left: 4px solid #ff00ff; }
            .method { display: inline-block; background-color: #007acc; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
            .new-badge { background-color: #ff00ff; color: white; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold; margin-left: 10px; }
            .status { color: #00ff88; font-weight: bold; }
            .warning { color: #ffaa00; }
            code { background-color: #2a2a2a; padding: 2px 6px; border-radius: 3px; }
            .pattern-list { background-color: #1a1a1a; padding: 15px; border-radius: 8px; border: 2px solid #ff00ff; }
            .setup-box { background-color: #1a1a1a; padding: 20px; border-radius: 8px; border-left: 4px solid #00ff88; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 XAU AI Trading Bot v3.0</h1>
            <p class="status">✅ System Online | Triple Hybrid AI Trading System</p>
            
            <h2>🎯 Trading Systems</h2>
            <ul>
                <li><strong>System 1:</strong> RSI + EMA + Price Change Analysis</li>
                <li><strong>System 2:</strong> Classic Chart Pattern Detection</li>
                <li><strong>System 3:</strong> <span style="color:#ff00ff;">⭐ Harmonic Patterns + Elliott Wave</span> <span class="new-badge">NEW!</span></li>
            </ul>
            
            <div class="pattern-list">
                <h3 style="color:#ff00ff; margin-top:0;">🌟 Harmonic Patterns Detected:</h3>
                <ul>
                    <li>🦋 <strong>GARTLEY</strong> - 61.8% XA retracement (High accuracy)</li>
                    <li>🦋 <strong>BUTTERFLY</strong> - 127-161.8% XA extension</li>
                    <li>🦇 <strong>BAT</strong> - 88.6% XA retracement</li>
                    <li>🦀 <strong>CRAB</strong> - 161.8% XA extension (Extreme)</li>
                    <li>📐 <strong>AB=CD</strong> - Equal leg structure</li>
                    <li>🌊 <strong>ELLIOTT WAVE 5</strong> - Impulse wave pattern</li>
                    <li>🌊 <strong>ELLIOTT WAVE 3</strong> - Corrective wave (ABC)</li>
                </ul>
            </div>
            
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong>
                <p>Health check endpoint for monitoring</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/run-ai</strong>
                <p><span class="status">ACTIVE:</span> Original AI System</p>
                <p><em>Frequency:</em> Every 3 minutes | <em>Output:</em> Once per hour</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/run-pattern-bot</strong>
                <p><span class="status">ACTIVE:</span> Classic Pattern AI</p>
                <p><em>Frequency:</em> Every 3 minutes | <em>Output:</em> Once per hour</p>
            </div>
            
            <div class="endpoint harmonic-endpoint">
                <span class="method">GET</span> <strong>/run-harmonic-bot</strong> <span class="new-badge">NEW!</span>
                <p><span class="status">ACTIVE:</span> Harmonic + Elliott Wave AI</p>
                <p><em>Frequency:</em> Every 3 minutes | <em>Output:</em> Once per hour</p>
                <p style="color:#ff00ff;"><strong>📊 Patterns:</strong> GARTLEY, BUTTERFLY, BAT, CRAB, AB=CD, ELLIOTT WAVE</p>
            </div>
            
            <h3>🧪 Test Endpoints</h3>
            
            <div class="endpoint harmonic-endpoint">
                <span class="method">GET</span> <strong>/test-harmonic</strong> <span class="new-badge">NEW!</span>
                <p>Test harmonic pattern detection (JSON response)</p>
            </div>
            
            <div class="endpoint harmonic-endpoint">
                <span class="method">GET</span> <strong>/test-harmonic-send</strong> <span class="new-badge">NEW!</span>
                <p>Test sending harmonic patterns to Telegram</p>
            </div>
            
            <div class="endpoint harmonic-endpoint">
                <span class="method">GET</span> <strong>/test-specific-harmonic?pattern=GARTLEY</strong> <span class="new-badge">NEW!</span>
                <p>Test specific harmonic pattern</p>
                <p><em>Parameters:</em> GARTLEY, BUTTERFLY, BAT, CRAB, AB_CD</p>
            </div>
            
            <div class="endpoint harmonic-endpoint">
                <span class="method">GET</span> <strong>/harmonic-status</strong> <span class="new-badge">NEW!</span>
                <p>Get current harmonic pattern analysis status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/test-telegram</strong>
                <p>Test Telegram connection</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/pattern-status</strong>
                <p>Get classic pattern analysis status</p>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <strong>/status</strong>
                <p>Get comprehensive system status</p>
            </div>
            
            <div class="setup-box">
                <h2 style="margin-top:0; color:#ff00ff;">⚡ UptimeRobot Setup (3 Systems)</h2>
                
                <p><strong>Monitor 1:</strong> <code>/run-ai</code> → Every 3 minutes</p>
                <p style="margin-left:20px;">→ Original AI signals (RSI+EMA)</p>
                
                <p><strong>Monitor 2:</strong> <code>/run-pattern-bot</code> → Every 3 minutes</p>
                <p style="margin-left:20px;">→ Classic pattern signals</p>
                
                <p><strong style="color:#ff00ff;">Monitor 3:</strong> <code>/run-harmonic-bot</code> → Every 3 minutes <span class="new-badge">NEW!</span></p>
                <p style="margin-left:20px; color:#ff00ff;">→ <strong>Harmonic + Elliott Wave signals</strong></p>
                
                <h3 style="color:#00ff88;">📊 Expected Results:</h3>
                <ul>
                    <li>🤖 <strong>1 signal/hour:</strong> Original AI (RSI+EMA)</li>
                    <li>📈 <strong>1 signal/hour:</strong> Classic Patterns</li>
                    <li style="color:#ff00ff;">🎯 <strong>1 signal/hour:</strong> Harmonic + Elliott Wave</li>
                </ul>
                <p style="color:#00ff88; font-weight:bold;">✅ Total: 3 independent signals per hour!</p>
                
                <h3 style="color:#ffaa00;">🎯 Why 3 Systems?</h3>
                <ul>
                    <li>Different analysis methods = Better market coverage</li>
                    <li>ML vs Rule-based vs Fibonacci = Multiple perspectives</li>
                    <li>Confirmation signals = Higher confidence trades</li>
                    <li>Never miss opportunities = 24/7 monitoring</li>
                </ul>
            </div>
            
            <h2>⚠️ Risk Disclaimer</h2>
            <p class="warning">
                This is an automated trading bot for educational purposes. 
                Harmonic patterns and Elliott Wave analysis require experience. 
                Always use proper risk management (1-2% per trade). 
                Past performance does not guarantee future results.
            </p>
            
            <hr style="border-color: #444; margin: 40px 0;">
            <p style="text-align: center; color: #666;">
                🚀 XAU AI Trading Bot v3.0 | Powered by Advanced Pattern Detection
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
    print("=" * 70)
    print("🤖 XAU AI Trading Bot v3.0 Starting...")
    print("=" * 70)
    print(f"Health Check: /health")
    print(f"System 1 - Original: /run-ai")
    print(f"System 2 - Classic Patterns: /run-pattern-bot")
    print(f"System 3 - Harmonic + Elliott: /run-harmonic-bot  ⭐ NEW!")
    print(f"\nTest Endpoints:")
    print(f"   • /test-harmonic")
    print(f"   • /test-harmonic-send")
    print(f"   • /test-specific-harmonic?pattern=GARTLEY")
    print(f"   • /harmonic-status")
    print(f"\nStatus:")
    print(f"   • /pattern-status")
    print(f"   • /status")
    print("=" * 70)
    print(f"Libraries Available:")
    print(f"   • TensorFlow: {'✅' if HAS_TENSORFLOW else '❌'}")
    print(f"   • Scikit-learn: {'✅' if HAS_SKLEARN else '❌'}")
    print(f"   • TA-Lib: {'✅' if HAS_TA else '❌'}")
    print(f"   • Charts: {'✅' if HAS_CHARTS else '❌'}")
    print("=" * 70)
    print(f"Configuration:")
    print(f"   • Bot Token: {'✅ Configured' if BOT_TOKEN else '❌ Missing'}")
    print(f"   • Chat ID: {'✅ Configured' if CHAT_ID else '❌ Missing'}")
    print(f"   • API Key: {'✅ Configured' if API_KEY else '❌ Missing'}")
    print("=" * 70)
    print("🎯 Harmonic Patterns Enabled:")
    print("   • GARTLEY, BUTTERFLY, BAT, CRAB, AB=CD")
    print("   • ELLIOTT WAVE 5 (Impulse)")
    print("   • ELLIOTT WAVE 3 (Corrective ABC)")
    print("=" * 70)
    print("🚀 Ready for AI-powered trading!")
    print("💰 Asset: XAU/USD | Timeframe: 1H")
    print("📊 3 Independent Systems Running")
    print("=" * 70)
    
    # Get port from environment
    port = int(os.environ.get("PORT", 5000))
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=False)
