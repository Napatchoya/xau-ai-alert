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
        import matplotlib.patches as patches
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
    """Get detailed pattern description"""
    descriptions = {
        'HEAD_SHOULDERS': """📊 HEAD & SHOULDERS PATTERN:

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

        'DOUBLE_TOP': """📊 DOUBLE TOP PATTERN:

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

        'DOUBLE_BOTTOM': """📊 DOUBLE BOTTOM PATTERN:

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

        'ASCENDING_TRIANGLE': """📊 ASCENDING TRIANGLE:

🔍 คุณสมบัติ:
• รูปแบบ Continuation แบบ Bullish
• แนวต้านแนวนอน (Horizontal Resistance)
• แนวรับทะยานขึ้น (Ascending Support)  
• ปริมาณการซื้อขายค่อยๆ ลดลง

📈 สั�ญาณ:
• เมื่อราคาทะลุ Resistance = สัญญาณ BUY
• Target = ความสูงของรูปสามเหลี่ยม
• Stop Loss ใต้แนวรับล่าสุด

⚠️ ความเสี่ยง: อาจ False Breakout ได้""",

        'BULL_FLAG': """📊 BULL FLAG PATTERN:

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

        'NO_PATTERN': """📊 NO CLEAR PATTERN:

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
    
    return descriptions.get(pattern_name, "ไม่มีข้อมูลแพทเทิร์นนี้")

def create_pattern_theory_diagram(pattern_name):
    """Create theoretical diagram explaining pattern characteristics"""
    try:
        import matplotlib.pyplot as plt
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

def send_pattern_theory_explanation(pattern_name, pattern_description):
    """Send pattern theory diagram and detailed explanation to Telegram"""
    try:
        # Create theory diagram
        theory_diagram = create_pattern_theory_diagram(pattern_name)
        
        if theory_diagram:
            # Prepare theory explanation message
            theory_message = f"""📚 PATTERN THEORY MASTER CLASS

🎯 {pattern_name} - ทฤษฎีและหลักการ

{pattern_description}

💡 หลักการสำคัญ:
• ศึกษาโครงสร้างแพทเทิร์นก่อนเทรดจริง
• รอการยืนยันก่อนเข้าออเดอร์
• ใช้ Risk Management ที่เข้มงวด
• เฝ้าดูปริมาณการซื้อขาย (Volume)

📖 แหล่งอ้างอิง: Technical Analysis Theory"""
            
            # Send diagram with theory explanation
            send_status = send_telegram_with_chart(theory_message, theory_diagram)
            print(f"Pattern theory diagram sent: Status {send_status}")
            return send_status
        else:
            # Fallback: send text-only theory
            theory_text = f"📚 PATTERN THEORY: {pattern_name}\n\n{pattern_description}"
            return send_telegram(theory_text)
            
    except Exception as e:
        print(f"Pattern theory explanation error: {e}")
        return 500

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
            return "❌ ไม่สามารถดึงข้อมูลสำหรับ Pattern Detection ได้", None, None
        
        detector = SimplePatternDetector()
        pattern_info = detector.detect_pattern(shared_df.tail(50))
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

        return message, chart_buffer, pattern_description
        
    except Exception as e:
        return f"❌ PATTERN AI ERROR: {str(e)}", None, None

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
    message, _, _ = run_pattern_ai_shared_with_chart(shared_df)
    return message
    try:
        if shared_df is None or len(shared_df) < 20:
            return "❌ ไม่สามารถดึงข้อมูลสำหรับ Pattern Detection ได้"
        
        detector = SimplePatternDetector()
        pattern_info = detector.detect_pattern(shared_df.tail(50))
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
                    # ใช้ shared data
                    shared_df = get_shared_xau_data()
                    if shared_df is not None:
                        # ใช้ฟังก์ชันใหม่ที่สร้างกราฟ
                        result, chart_buffer, pattern_description = run_pattern_ai_shared_with_chart(shared_df)
            
                        # ส่งข้อความพร้อมกราฟ
                        send_status = send_telegram_with_chart(result, chart_buffer)
            
                        # ส่งคำอธิบายแพทเทิร์นแยกต่างหาก (ถ้ามี)
                        if pattern_description and pattern_description != "ไม่มีข้อมูลแพทเทิร์นนี้":
                            time.sleep(2)  # รอ 2 วินาทีก่อนส่งข้อความต่อไป
                            send_telegram(f"📚 รายละเอียดแพทเทิร์น:\n{pattern_description}")
            
                        print(f"✅ [{current_time}] Pattern AI with chart sent to Telegram: Status {send_status}")
                        print(f"Pattern message preview: {result[:150]}...")
                    else:
                        error_msg = f"❌ Pattern AI Data Error @ {current_time}\nCannot fetch market data"
                        send_telegram(error_msg)
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

@app.route('/test-pattern-chart')
def test_pattern_chart():
    """Test pattern AI with chart generation"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is not None:
            result, chart_buffer, pattern_description = run_pattern_ai_shared_with_chart(shared_df)
            
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
        
        detector = SimplePatternDetector()
        pattern_info = detector.detect_pattern(shared_df.tail(50))
        
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
