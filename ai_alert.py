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
from scipy import interpolate  # สำหรับ smooth curves
import base64
import json
import asyncio

# Try importing optional ML libraries
try:
    from scipy import interpolate
    HAS_SCIPY = True
except ImportError:
    print("⚠️ SciPy not available, some features may be limited")
    HAS_SCIPY = False

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

# ========== NEW: LLM APIs ==========
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    print("⚠️ OpenAI not available")
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    print("⚠️ Gemini not available")
    HAS_GEMINI = False

load_dotenv()

# Environment Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")

# NEW: LLM API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Configure LLMs
if OPENAI_API_KEY and HAS_OPENAI:
    openai.api_key = OPENAI_API_KEY
    print("✅ OpenAI configured")

if GEMINI_API_KEY and HAS_GEMINI:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured")

app = Flask(__name__)

# Global variables
last_signal = None
last_original_sent_hour = None  # สำหรับระบบเก่า
last_pattern_sent_hour = None   # สำหรับระบบ Pattern AI
last_harmonic_sent_hour = None  # เพิ่มบรรทัดนี้ที่ส่วน Global variables
message_sent_this_hour = {      # เพิ่มตัวตรวจสอบว่าส่งข้อความไหนแล้วบ้าง
    'original': None,
    'pattern': None,
    'harmonic': None,  # เพิ่มบรรทัดนี้
    'ai_analysis': None  # NEW: for AI signals
}
# NEW: AI Analysis history
ai_analysis_history = []

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

# NEW: Additional indicators for AI
def calculate_macd(prices):
    """Calculate MACD"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calculate_atr(df, period=14):
    """Calculate ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(period).mean()
    
    return atr

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

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
        df["ema_50"] = calculate_ema(df["close"], 50)
        df["macd"], df["macd_signal"] = calculate_macd(df["close"])
        df["atr"] = calculate_atr(df)
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(df["close"])
        df["price_change"] = df["close"].pct_change()
        
        return df
        
    except Exception as e:
        print(f"Error fetching shared XAU data: {e}")
        return None

# ====================== NEW: News & Economic Data ======================

def get_market_news():
    """Fetch recent gold market news"""
    try:
        news_api_key = os.getenv("NEWS_API_KEY")
        if not news_api_key:
            return []
        
        url = f"https://newsapi.org/v2/everything?q=gold+trading+federal+reserve&apiKey={news_api_key}&pageSize=5&sortBy=publishedAt"
        response = requests.get(url, timeout=15)
        articles = response.json().get('articles', [])
        
        return [
            {
                "title": a.get('title', ''),
                "description": a.get('description', '')[:200],
                "source": a.get('source', {}).get('name', '')
            } 
            for a in articles[:3]
        ]
    except Exception as e:
        print(f"⚠️ News fetch error: {e}")
        return []

def get_economic_indicators():
    """Get economic indicators affecting gold"""
    return {
        "dxy_index": "DXY weakening = Gold bullish",
        "interest_rates": "Fed at 5.25% - watch for cuts",
        "inflation": "CPI above target supports gold",
        "sentiment": "Risk-on/off environment"
    }

# ====================== NEW: AI ANALYST CLASSES ======================

class AIAnalyst:
    """Base class for AI analysts"""
    
    def __init__(self, name):
        self.name = name
    
    def create_analysis_prompt(self, market_data):
        """Create comprehensive prompt for AI"""
        current_price = market_data['price']
        indicators = market_data['indicators']
        pattern = market_data['pattern']
        news = market_data.get('news', [])
        economics = market_data.get('economics', {})
        
        news_text = "\n".join([f"- {n['title']}" for n in news[:3]]) if news else "No recent news"
        
        prompt = f"""You are a professional gold (XAU/USD) trader with 20 years of experience.

CURRENT MARKET DATA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Price: ${current_price:,.2f}

📈 Technical Indicators:
• RSI(14): {indicators['rsi']:.2f}
• MACD: {indicators['macd']:.2f} (Signal: {indicators['macd_signal']:.2f})
• EMA 10/21/50: ${indicators['ema']:.2f} / ${indicators['ema_21']:.2f} / ${indicators['ema_50']:.2f}
• ATR: {indicators['atr']:.2f}
• Bollinger Bands: ${indicators['bb_lower']:.2f} - ${indicators['bb_upper']:.2f}

🎯 Chart Pattern: {pattern['name']} (Confidence: {pattern['confidence']}%)

🌍 Economic Factors:
{chr(10).join([f"• {k}: {v}" for k, v in economics.items()])}

📰 Recent News:
{news_text}

TASK: Analyze and decide BUY, SELL, or HOLD.

Respond ONLY in this JSON format (no other text):
{{
    "action": "BUY/SELL/HOLD",
    "confidence": 85,
    "stop_loss_pips": 50,
    "take_profit_pips": 150,
    "reasoning": "Brief explanation (max 100 words)"
}}
"""
        return prompt
    
    async def analyze(self, market_data):
        """Analyze market - to be overridden"""
        raise NotImplementedError
    
    def _fallback_analysis(self, market_data):
        """Simple fallback when API unavailable"""
        indicators = market_data['indicators']
        rsi = indicators['rsi']
        
        if rsi < 30:
            action = "BUY"
            conf = 70
        elif rsi > 70:
            action = "SELL"
            conf = 70
        else:
            action = "HOLD"
            conf = 50
        
        return {
            "analyst": self.name,
            "action": action,
            "confidence": conf,
            "stop_loss_pips": 50,
            "take_profit_pips": 100,
            "reasoning": f"RSI-based decision: {rsi:.1f}"
        }

class OpenAIAnalyst(AIAnalyst):
    """OpenAI GPT-4 Analyst"""
    
    def __init__(self):
        super().__init__("OpenAI GPT-4")
    
    async def analyze(self, market_data):
        if not OPENAI_API_KEY or not HAS_OPENAI:
            return self._fallback_analysis(market_data)
        
        try:
            prompt = self.create_analysis_prompt(market_data)
            
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a professional XAU/USD trader. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            analysis_text = response.choices[0].message.content
            
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                analysis['analyst'] = self.name
                return analysis
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
        
        return self._fallback_analysis(market_data)

class GeminiAnalyst(AIAnalyst):
    """Google Gemini Analyst"""
    
    def __init__(self):
        super().__init__("Google Gemini")
        if GEMINI_API_KEY and HAS_GEMINI:
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
    
    async def analyze(self, market_data):
        if not self.model:
            return self._fallback_analysis(market_data)
        
        try:
            prompt = self.create_analysis_prompt(market_data)
            response = self.model.generate_content(prompt)
            analysis_text = response.text
            
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                analysis['analyst'] = self.name
                return analysis
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
        
        return self._fallback_analysis(market_data)

class DeepSeekAnalyst(AIAnalyst):
    """DeepSeek AI Analyst"""
    
    def __init__(self):
        super().__init__("DeepSeek")
    
    async def analyze(self, market_data):
        if not DEEPSEEK_API_KEY:
            return self._fallback_analysis(market_data)
        
        try:
            prompt = self.create_analysis_prompt(market_data)
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis_text = data['choices'][0]['message']['content']
                
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    analysis['analyst'] = self.name
                    return analysis
            
        except Exception as e:
            print(f"❌ {self.name} error: {e}")
        
        return self._fallback_analysis(market_data)

class GrokAnalyst(AIAnalyst):
    """Grok (xAI) Analyst"""
    
    def __init__(self):
        super().__init__("Grok (xAI)")
    
    async def analyze(self, market_data):
        # Pattern-based fallback (Grok API not fully public yet)
        pattern = market_data['pattern']
        indicators = market_data['indicators']
        
        bullish_patterns = ['BULL_FLAG', 'DOUBLE_BOTTOM', 'INVERSE_HEAD_SHOULDERS', 'CUP_AND_HANDLE']
        bearish_patterns = ['BEAR_FLAG', 'DOUBLE_TOP', 'HEAD_SHOULDERS']
        
        if pattern['name'] in bullish_patterns:
            action = "BUY"
            conf = pattern['confidence']
        elif pattern['name'] in bearish_patterns:
            action = "SELL"
            conf = pattern['confidence']
        else:
            action = "HOLD" if indicators['ema'] < indicators['ema_21'] else "BUY"
            conf = 65
        
        return {
            "analyst": self.name,
            "action": action,
            "confidence": conf,
            "stop_loss_pips": 55,
            "take_profit_pips": 165,
            "reasoning": f"Pattern {pattern['name']} + EMA trend analysis"
        }

# ====================== NEW: AI Consensus System ======================

async def get_ai_consensus(market_data):
    """Get analysis from all AI agents"""
    analysts = [
        OpenAIAnalyst(),
        GeminiAnalyst(),
        DeepSeekAnalyst(),
        GrokAnalyst()
    ]
    
    analyses = []
    
    for analyst in analysts:
        try:
            analysis = await analyst.analyze(market_data)
            analyses.append(analysis)
            print(f"  ✓ {analyst.name}: {analysis['action']} ({analysis['confidence']}%)")
        except Exception as e:
            print(f"  ✗ {analyst.name} failed: {e}")
    
    return analyses

def make_consensus_decision(analyses):
    """Aggregate AI decisions"""
    if not analyses:
        return None
    
    buy_votes = sum(1 for a in analyses if a['action'] == 'BUY')
    sell_votes = sum(1 for a in analyses if a['action'] == 'SELL')
    hold_votes = sum(1 for a in analyses if a['action'] == 'HOLD')
    
    buy_conf = np.mean([a['confidence'] for a in analyses if a['action'] == 'BUY']) if buy_votes > 0 else 0
    sell_conf = np.mean([a['confidence'] for a in analyses if a['action'] == 'SELL']) if sell_votes > 0 else 0
    
    if buy_votes > sell_votes and buy_votes > hold_votes:
        final_action = "BUY"
        final_conf = buy_conf
    elif sell_votes > buy_votes and sell_votes > hold_votes:
        final_action = "SELL"
        final_conf = sell_conf
    else:
        final_action = "HOLD"
        final_conf = 50
    
    avg_sl = np.mean([a['stop_loss_pips'] for a in analyses])
    avg_tp = np.mean([a['take_profit_pips'] for a in analyses])
    
    return {
        "action": final_action,
        "confidence": round(final_conf, 1),
        "votes": {"BUY": buy_votes, "SELL": sell_votes, "HOLD": hold_votes},
        "stop_loss_pips": round(avg_sl, 0),
        "take_profit_pips": round(avg_tp, 0),
        "individual_analyses": analyses
    }

# ====================== NEW: Enhanced Chart with AI ======================

def create_ai_enhanced_chart(df, consensus, pattern_info):
    """Create chart with AI consensus overlay"""
    try:
        chart_df = df.tail(50).copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        fig.patch.set_facecolor('#0a0a0a')
        ax1.set_facecolor('#0a0a0a')
        ax2.set_facecolor('#0a0a0a')
        
        # Candlesticks
        for i, (idx, row) in enumerate(chart_df.iterrows()):
            color = '#00ff88' if row['close'] >= row['open'] else '#ff4444'
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            ax1.add_patch(patches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, edgecolor=color, alpha=0.9
            ))
            ax1.plot([i, i], [row['low'], row['high']], 
                    color=color, linewidth=1.5, alpha=0.8)
        
        # EMAs
        ax1.plot(range(len(chart_df)), chart_df['ema'].values, 
                color='#ffaa00', linewidth=2.5, label='EMA 10', alpha=0.9)
        ax1.plot(range(len(chart_df)), chart_df['ema_21'].values, 
                color='#ff6600', linewidth=2.5, label='EMA 21', alpha=0.9)
        
        # AI Consensus Box
        votes_text = f"BUY:{consensus['votes']['BUY']} | SELL:{consensus['votes']['SELL']} | HOLD:{consensus['votes']['HOLD']}"
        
        consensus_color = {
            'BUY': '#00ff88',
            'SELL': '#ff4444',
            'HOLD': '#ffaa00'
        }.get(consensus['action'], '#ffffff')
        
        ai_box = f"🤖 AI CONSENSUS\n{consensus['action']} | {consensus['confidence']}%\n{votes_text}"
        
        ax1.text(0.02, 0.98, ai_box,
                transform=ax1.transAxes, va='top', ha='left',
                color=consensus_color, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a1a', 
                         edgecolor=consensus_color, alpha=0.95, linewidth=3))
        
        # Pattern info
        pattern_text = f"Pattern: {pattern_info['name']}\n{pattern_info['confidence']}%"
        ax1.text(0.98, 0.98, pattern_text,
                transform=ax1.transAxes, va='top', ha='right',
                color='#ff00ff', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a1a', 
                         edgecolor='#ff00ff', alpha=0.9, linewidth=2))
        
        # Trading levels
        current_price = chart_df['close'].iloc[-1]
        if consensus['action'] != 'HOLD':
            sl_pips = consensus['stop_loss_pips']
            tp_pips = consensus['take_profit_pips']
            
            if consensus['action'] == 'BUY':
                sl = current_price - sl_pips
                tp = current_price + tp_pips
            else:
                sl = current_price + sl_pips
                tp = current_price - tp_pips
            
            ax1.axhline(y=current_price, color='#ffffff', linestyle='--', 
                       linewidth=2.5, alpha=0.95, label=f'Entry: ${current_price:.2f}')
            ax1.axhline(y=tp, color='#00ff88', linestyle='-', 
                       linewidth=2, alpha=0.85, label=f'TP: ${tp:.2f}')
            ax1.axhline(y=sl, color='#ff4444', linestyle='-', 
                       linewidth=2, alpha=0.85, label=f'SL: ${sl:.2f}')
        
        title = f'XAU/USD - {consensus["action"]} | AI Multi-Agent Analysis'
        ax1.set_title(title, color=consensus_color, fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Price ($)', color='#ffffff', fontsize=13)
        ax1.tick_params(colors='#ffffff')
        ax1.grid(True, alpha=0.2, color='#444444')
        ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#444444', 
                  labelcolor='#ffffff', fontsize=9)
        
        # RSI
        if 'rsi' in chart_df.columns:
            rsi_values = chart_df['rsi'].dropna()
            ax2.plot(range(len(rsi_values)), rsi_values.values, 
                    color='#00ddff', linewidth=2.5, label='RSI')
            ax2.axhline(y=70, color='#ff4444', linestyle='--', alpha=0.8, linewidth=1.5)
            ax2.axhline(y=30, color='#00ff88', linestyle='--', alpha=0.8, linewidth=1.5)
            ax2.axhline(y=50, color='#888888', linestyle='-', alpha=0.5)
            
            ax2.set_ylabel('RSI', color='#ffffff', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.tick_params(colors='#ffffff')
            ax2.grid(True, alpha=0.2, color='#444444')
        
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        fig.text(0.5, 0.01, f"🤖 AI Analysis | {timestamp} | OpenAI • Gemini • DeepSeek • Grok", 
                ha='center', color='#888888', fontsize=9)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#0a0a0a', dpi=120, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"❌ AI chart error: {e}")
        return None

# ====================== NEW: Telegram with AI ======================

def send_ai_telegram_alert(consensus, market_data, pattern_info, chart_buffer=None):
    """Send AI analysis to Telegram"""
    try:
        action_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}
        emoji = action_emoji.get(consensus['action'], '⚪')
        
        price = market_data['price']
        indicators = market_data['indicators']
        
        ai_votes = "\n".join([
            f"  • {a['analyst']}: {a['action']} ({a['confidence']}%)"
            for a in consensus['individual_analyses']
        ])
        
        top_analyst = max(consensus['individual_analyses'], key=lambda x: x['confidence'])
        reasoning = top_analyst['reasoning'][:120]
        
        message = f"""
🤖 <b>XAU/USD AI SIGNAL</b> 🤖

{emoji} <b>CONSENSUS: {consensus['action']}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Market:</b>
  • Price: ${price:,.2f}
  • RSI: {indicators['rsi']:.1f}
  • MACD: {indicators['macd']:.2f}

🎯 <b>Pattern:</b> {pattern_info['name']} ({pattern_info['confidence']}%)

🤖 <b>AI Votes:</b>
{ai_votes}

📈 <b>Decision:</b>
  • {consensus['action']} | Confidence: {consensus['confidence']}%
  • Votes: 🟢{consensus['votes']['BUY']} | 🔴{consensus['votes']['SELL']} | 🟡{consensus['votes']['HOLD']}

💡 <b>Analysis:</b>
"{reasoning}"

📌 <b>Risk:</b>
  • SL: {consensus['stop_loss_pips']} pips
  • TP: {consensus['take_profit_pips']} pips
  • R:R = {consensus['take_profit_pips']/consensus['stop_loss_pips']:.1f}:1

⏰ {datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")}
━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ <i>AI-generated. Trade responsibly.</i>
"""
        
        if not BOT_TOKEN or not CHAT_ID:
            print("⚠️ Telegram not configured")
            return False
        
        if chart_buffer:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            chart_buffer.seek(0)
            files = {'photo': chart_buffer}
            data = {'chat_id': CHAT_ID, 'caption': message, 'parse_mode': 'HTML'}
            response = requests.post(url, files=files, data=data, timeout=30)
        else:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            data = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
            response = requests.post(url, data=data, timeout=30)
        
        if response.status_code == 200:
            print("✅ AI Telegram alert sent")
            return True
        else:
            print(f"❌ Telegram error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Telegram AI alert error: {e}")
        return False

# ====================== NEW: Main AI Analysis Function ======================

async def run_ai_analysis():
    """Main AI analysis - NEW SYSTEM"""
    global message_sent_this_hour
    
    try:
        current_hour = datetime.now(ZoneInfo("Asia/Bangkok")).hour
        
        # Check if already sent this hour
        if message_sent_this_hour.get('ai_analysis') == current_hour:
            print("⏭️ AI analysis already sent this hour")
            return
        
        print("\n" + "="*70)
        print(f"🤖 AI MULTI-LLM ANALYSIS - {datetime.now(ZoneInfo('Asia/Bangkok'))}")
        print("="*70)
        
        # 1. Fetch data
        print("📊 Fetching market data...")
        df = get_shared_xau_data()
        
        if df is None or len(df) < 50:
            print("❌ Insufficient data")
            return
        
        current_price = df['close'].iloc[-1]
        print(f"✅ Price: ${current_price:,.2f}")
        
        # 2. Detect pattern (use existing function)
        print("🔍 Detecting patterns...")
        pattern_name, pattern_confidence, is_priority = detect_pattern(df)
        
        pattern_info = {
            'name': pattern_name,
            'confidence': pattern_confidence,
            'priority': is_priority
        }
        print(f"✅ Pattern: {pattern_name} ({pattern_confidence}%)")
        
        # 3. Get news and economics
        print("📰 Fetching news...")
        news = get_market_news()
        economics = get_economic_indicators()
        
        # 4. Prepare market data
        market_data = {
            'price': current_price,
            'indicators': {
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'macd_signal': df['macd_signal'].iloc[-1],
                'ema': df['ema'].iloc[-1],
                'ema_21': df['ema_21'].iloc[-1],
                'ema_50': df['ema_50'].iloc[-1],
                'atr': df['atr'].iloc[-1],
                'bb_upper': df['bb_upper'].iloc[-1],
                'bb_lower': df['bb_lower'].iloc[-1]
            },
            'pattern': pattern_info,
            'news': news,
            'economics': economics
        }
        
        # 5. Get AI consensus
        print("\n🤖 Querying AI agents...")
        analyses = await get_ai_consensus(market_data)
        
        if not analyses:
            print("❌ No AI responses")
            return
        
        # 6. Make consensus
        print("\n📊 Building consensus...")
        consensus = make_consensus_decision(analyses)
        
        print(f"\n{'='*70}")
        print(f"🎯 CONSENSUS: {consensus['action']} | Confidence: {consensus['confidence']}%")
        print(f"   Votes: BUY={consensus['votes']['BUY']}, SELL={consensus['votes']['SELL']}, HOLD={consensus['votes']['HOLD']}")
        print(f"{'='*70}\n")
        
        # 7. Create chart
        print("📈 Creating AI chart...")
        chart_buffer = create_ai_enhanced_chart(df, consensus, pattern_info)
        
        # 8. Send to Telegram (only if actionable)
        should_send = (
            consensus['action'] != 'HOLD' or 
            consensus['confidence'] >= 75 or 
            pattern_info['priority']
        )
        
        if should_send:
            print("📤 Sending AI signal to Telegram...")
            success = send_ai_telegram_alert(consensus, market_data, pattern_info, chart_buffer)
            
            if success:
                message_sent_this_hour['ai_analysis'] = current_hour
                
                # Store in history
                ai_analysis_history.append({
                    'timestamp': datetime.now(ZoneInfo('Asia/Bangkok')),
                    'consensus': consensus,
                    'pattern': pattern_info,
                    'price': current_price
                })
        else:
            print("⏭️ Signal not strong enough - skipping")
        
        print("\n✅ AI Analysis complete!\n")
        
    except Exception as e:
        print(f"❌ AI analysis error: {e}")
        import traceback
        traceback.print_exc()

# ====================== Pattern Detection (Keep Original) ======================



# ====================== Chart Generation Functions ======================
def draw_enhanced_pattern_lines(ax, df, pattern_info):
    """Enhanced pattern line drawing with specific point marking"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        
        
        # ✅ Priority: Harmonic Patterns
        if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            if 'points' in pattern_info and pattern_info['points']:
                print(f"📊 Drawing Harmonic pattern: {pattern_name}")
                draw_harmonic_on_chart(ax, df, pattern_info['points'], pattern_name)
                return
        
        # ✅ AB=CD Pattern
        elif pattern_name == 'AB_CD':
            if 'points' in pattern_info and pattern_info['points']:
                print(f"📊 Drawing AB=CD pattern")
                draw_abcd_on_chart(ax, df, pattern_info['points'])
                return
        
        # ✅ Elliott Wave Patterns
        elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
            if 'wave_points' in pattern_info and pattern_info['wave_points']:
                print(f"📊 Drawing Elliott Wave: {pattern_name}")
                draw_elliott_wave_on_chart(ax, df, pattern_info['wave_points'], pattern_name)
                return
        
        # Classic Chart Patterns
        elif pattern_name == 'DOUBLE_TOP':
            draw_double_top_on_chart(ax, df)
        elif pattern_name == 'HEAD_SHOULDERS':
            draw_head_shoulders_on_chart(ax, df)
        elif pattern_name == 'DOUBLE_BOTTOM':
            draw_double_bottom_on_chart(ax, df)
        elif pattern_name == 'BULL_FLAG':
            draw_bull_flag_on_chart(ax, df)
        elif pattern_name == 'BEAR_FLAG':
            draw_bear_flag_on_chart(ax, df)
        elif pattern_name == 'SYMMETRICAL_TRIANGLE':
            draw_symmetrical_triangle_on_chart(ax, df)
        elif pattern_name == 'ASCENDING_TRIANGLE':
            draw_ascending_triangle_on_chart(ax, df)
        elif pattern_name == 'DESCENDING_TRIANGLE':
            draw_descending_triangle_on_chart(ax, df)
        elif pattern_name == 'PENNANT':
            draw_pennant_on_chart(ax, df)
        elif pattern_name == 'WEDGE_RISING':
            draw_wedge_rising_on_chart(ax, df)
        elif pattern_name == 'WEDGE_FALLING':
            draw_wedge_falling_on_chart(ax, df)
        elif pattern_name == 'CUP_AND_HANDLE':
            draw_cup_and_handle_on_chart(ax, df)
        elif pattern_name == 'INVERSE_HEAD_SHOULDERS':
            # ✅ ส่ง chart_df แทน df
            draw_inverse_head_shoulders_on_chart(ax, chart_df)
        elif pattern_name == 'RECTANGLE':
            draw_rectangle_on_chart(ax, df)
        elif pattern_name == 'DIAMOND':
            draw_diamond_on_chart(ax, df)
        
        # 🆕 Candlestick Patterns - ใช้ฟังก์ชันใหม่
        elif pattern_name in [
            'DOJI', 'HAMMER', 'SHOOTING_STAR', 'HANGING_MAN', 
            'INVERTED_HAMMER', 'MARUBOZU', 'SPINNING_TOP',
            'ENGULFING_BULLISH', 'ENGULFING_BEARISH',
            'PIERCING_LINE', 'DARK_CLOUD_COVER',
            'HARAMI_BULLISH', 'HARAMI_BEARISH',
            'TWEEZER_TOP', 'TWEEZER_BOTTOM',
            'MORNING_STAR', 'EVENING_STAR',
            'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS'
        ]:
            # วาด marker
            draw_candlestick_pattern_markers(ax, df, pattern_name)
            
            # Highlight candles ตามจำนวน
            if pattern_name in ['ENGULFING_BULLISH', 'ENGULFING_BEARISH',
                               'PIERCING_LINE', 'DARK_CLOUD_COVER',
                               'HARAMI_BULLISH', 'HARAMI_BEARISH',
                               'TWEEZER_TOP', 'TWEEZER_BOTTOM']:
                draw_two_candlestick_highlight(ax, df)
            
            elif pattern_name in ['MORNING_STAR', 'EVENING_STAR',
                                 'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS']:
                draw_three_candlestick_highlight(ax, df)
        
        else:
            print(f"⚠️ No specific drawing function for pattern: {pattern_name}")
            
    except Exception as e:
        print(f"❌ Enhanced pattern marking error: {e}")
        import traceback
        traceback.print_exc()

def draw_candlestick_pattern_markers(ax, df, pattern_name):
    """วาด markers สำหรับ Candlestick patterns ทั้งหมด"""
    try:
        if len(df) < 1:
            return
        
        last_idx = len(df) - 1
        last_candle = df.iloc[-1]
        
        # กำหนดสีและ emoji สำหรับแต่ละ pattern
        pattern_styles = {
            # Single Candlestick
            'DOJI': {'color': '#ffff00', 'marker': '*', 'size': 250, 'emoji': '⭐', 'label': 'DOJI'},
            'HAMMER': {'color': '#00ff88', 'marker': '^', 'size': 250, 'emoji': '🔨', 'label': 'HAMMER'},
            'SHOOTING_STAR': {'color': '#ff4444', 'marker': 'v', 'size': 250, 'emoji': '💫', 'label': 'SHOOTING STAR'},
            'HANGING_MAN': {'color': '#ff6600', 'marker': 'v', 'size': 250, 'emoji': '👨', 'label': 'HANGING MAN'},
            'INVERTED_HAMMER': {'color': '#00ffff', 'marker': '^', 'size': 250, 'emoji': '🔨', 'label': 'INV HAMMER'},
            'MARUBOZU': {'color': '#ff00ff', 'marker': 's', 'size': 250, 'emoji': '📊', 'label': 'MARUBOZU'},
            'SPINNING_TOP': {'color': '#ffaa00', 'marker': 'o', 'size': 250, 'emoji': '🌀', 'label': 'SPINNING TOP'},
            
            # Two Candlestick
            'ENGULFING_BULLISH': {'color': '#00ff00', 'marker': 'D', 'size': 280, 'emoji': '🟢', 'label': 'ENGULFING BULL'},
            'ENGULFING_BEARISH': {'color': '#ff0000', 'marker': 'D', 'size': 280, 'emoji': '🔴', 'label': 'ENGULFING BEAR'},
            'PIERCING_LINE': {'color': '#00dd88', 'marker': '^', 'size': 280, 'emoji': '⬆️', 'label': 'PIERCING LINE'},
            'DARK_CLOUD_COVER': {'color': '#dd0088', 'marker': 'v', 'size': 280, 'emoji': '⬇️', 'label': 'DARK CLOUD'},
            'HARAMI_BULLISH': {'color': '#88ff88', 'marker': 'p', 'size': 280, 'emoji': '🤰', 'label': 'HARAMI BULL'},
            'HARAMI_BEARISH': {'color': '#ff8888', 'marker': 'p', 'size': 280, 'emoji': '🤰', 'label': 'HARAMI BEAR'},
            'TWEEZER_TOP': {'color': '#ff66ff', 'marker': 'X', 'size': 280, 'emoji': '🔝', 'label': 'TWEEZER TOP'},
            'TWEEZER_BOTTOM': {'color': '#66ff66', 'marker': 'X', 'size': 280, 'emoji': '🔻', 'label': 'TWEEZER BOT'},
            
            # Three Candlestick
            'MORNING_STAR': {'color': '#ffff00', 'marker': '*', 'size': 300, 'emoji': '🌟', 'label': 'MORNING STAR'},
            'EVENING_STAR': {'color': '#ff00ff', 'marker': '*', 'size': 300, 'emoji': '🌙', 'label': 'EVENING STAR'},
            'THREE_WHITE_SOLDIERS': {'color': '#00ff00', 'marker': '^', 'size': 300, 'emoji': '⚔️', 'label': '3 SOLDIERS'},
            'THREE_BLACK_CROWS': {'color': '#000000', 'marker': 'v', 'size': 300, 'emoji': '🦅', 'label': '3 CROWS'}
        }
        
        if pattern_name not in pattern_styles:
            print(f"⚠️ No style defined for pattern: {pattern_name}")
            return
        
        style = pattern_styles[pattern_name]
        
        # กำหนดตำแหน่ง marker
        y_position = last_candle['high'] if pattern_name in [
            'SHOOTING_STAR', 'HANGING_MAN', 'EVENING_STAR', 
            'DARK_CLOUD_COVER', 'TWEEZER_TOP', 'THREE_BLACK_CROWS',
            'ENGULFING_BEARISH', 'HARAMI_BEARISH'
        ] else last_candle['low']
        
        # วาด marker หลัก
        ax.scatter([last_idx], [y_position], 
                  color=style['color'], 
                  s=style['size'], 
                  marker=style['marker'],
                  label=style['label'], 
                  zorder=20,
                  edgecolors='white',
                  linewidths=3,
                  alpha=0.95)
        
        # เพิ่ม emoji label
        y_offset = 15 if y_position == last_candle['high'] else -15
        va = 'bottom' if y_position == last_candle['low'] else 'top'
        
        ax.text(last_idx, y_position + y_offset, 
               f"{style['emoji']} {style['label']}", 
               ha='center', 
               va=va, 
               color=style['color'], 
               fontweight='bold', 
               fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', 
                        edgecolor=style['color'],
                        alpha=0.9, 
                        linewidth=2))
        
        # เพิ่ม info box สำหรับ Two/Three candlestick patterns
        if pattern_name in ['ENGULFING_BULLISH', 'ENGULFING_BEARISH', 
                           'PIERCING_LINE', 'DARK_CLOUD_COVER',
                           'MORNING_STAR', 'EVENING_STAR',
                           'THREE_WHITE_SOLDIERS', 'THREE_BLACK_CROWS']:
            draw_candlestick_pattern_info(ax, df, pattern_name)
        
        print(f"✅ Candlestick marker drawn: {pattern_name}")
        
    except Exception as e:
        print(f"❌ Candlestick marker error for {pattern_name}: {e}")
        import traceback
        traceback.print_exc()


def draw_candlestick_pattern_info(ax, df, pattern_name):
    """วาดกล่องข้อมูลเพิ่มเติมสำหรับ Candlestick patterns"""
    try:
        if len(df) < 3:
            return
        
        pattern_info = {
            'ENGULFING_BULLISH': {
                'candles': 2,
                'signal': '🟢 Bullish Reversal',
                'description': 'Large white candle engulfs previous black'
            },
            'ENGULFING_BEARISH': {
                'candles': 2,
                'signal': '🔴 Bearish Reversal',
                'description': 'Large black candle engulfs previous white'
            },
            'PIERCING_LINE': {
                'candles': 2,
                'signal': '🟢 Bullish Reversal',
                'description': 'White candle closes above midpoint'
            },
            'DARK_CLOUD_COVER': {
                'candles': 2,
                'signal': '🔴 Bearish Reversal',
                'description': 'Black candle opens above, closes below midpoint'
            },
            'MORNING_STAR': {
                'candles': 3,
                'signal': '🟢 Strong Bullish Reversal',
                'description': 'Black → Small → Large White'
            },
            'EVENING_STAR': {
                'candles': 3,
                'signal': '🔴 Strong Bearish Reversal',
                'description': 'White → Small → Large Black'
            },
            'THREE_WHITE_SOLDIERS': {
                'candles': 3,
                'signal': '🟢 Strong Bullish Continuation',
                'description': 'Three consecutive rising white candles'
            },
            'THREE_BLACK_CROWS': {
                'candles': 3,
                'signal': '🔴 Strong Bearish Continuation',
                'description': 'Three consecutive falling black candles'
            }
        }
        
        if pattern_name not in pattern_info:
            return
        
        info = pattern_info[pattern_name]
        last_idx = len(df) - 1
        
        # สร้างกล่องข้อมูล
        info_text = f"{info['signal']}\n{info['description']}\nCandles: {info['candles']}"
        
        # วางกล่องที่มุมขวาบน
        ax.text(0.98, 0.85, info_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               color='white',
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.7', 
                        facecolor='#1a1a1a', 
                        edgecolor='#ffaa00',
                        alpha=0.95, 
                        linewidth=2))
        
    except Exception as e:
        print(f"❌ Pattern info box error: {e}")


def draw_two_candlestick_highlight(ax, df):
    """Highlight 2 แท่งเทียนล่าสุดสำหรับ Two-Candlestick patterns"""
    try:
        if len(df) < 2:
            return
        
        last_idx = len(df) - 1
        
        # วาดกรอบรอบ 2 แท่งล่าสุด
        first_candle = df.iloc[-2]
        second_candle = df.iloc[-1]
        
        box_high = max(first_candle['high'], second_candle['high'])
        box_low = min(first_candle['low'], second_candle['low'])
        
        # วาดกรอบสีเหลือง
        rect = patches.Rectangle((last_idx - 1.5, box_low), 
                                 2, 
                                 box_high - box_low,
                                 linewidth=2, 
                                 edgecolor='#ffaa00', 
                                 facecolor='none',
                                 linestyle='--',
                                 alpha=0.8,
                                 zorder=15)
        ax.add_patch(rect)
        
    except Exception as e:
        print(f"❌ Two candlestick highlight error: {e}")


def draw_three_candlestick_highlight(ax, df):
    """Highlight 3 แท่งเทียนล่าสุดสำหรับ Three-Candlestick patterns"""
    try:
        if len(df) < 3:
            return
        
        last_idx = len(df) - 1
        
        # วาดกรอบรอบ 3 แท่งล่าสุด
        candles = df.iloc[-3:]
        
        box_high = candles['high'].max()
        box_low = candles['low'].min()
        
        # วาดกรอบสีชมพู
        rect = patches.Rectangle((last_idx - 2.5, box_low), 
                                 3, 
                                 box_high - box_low,
                                 linewidth=2, 
                                 edgecolor='#ff00ff', 
                                 facecolor='none',
                                 linestyle='-',
                                 alpha=0.8,
                                 zorder=15)
        ax.add_patch(rect)
        
    except Exception as e:
        print(f"❌ Three candlestick highlight error: {e}")

def create_candlestick_chart(df, trading_signals, pattern_info):
    """Create candlestick chart with pattern lines and trading levels"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        import matplotlib.dates as mdates
        from scipy import interpolate
        
        chart_df = df.tail(50).copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        fig.patch.set_facecolor('#1a1a1a')
        ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('#1a1a1a')
        
        # Main candlestick chart
        for i, (idx, row) in enumerate(chart_df.iterrows()):
            color = '#00ff88' if row['close'] >= row['open'] else '#ff4444'
            
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            ax1.add_patch(patches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height,
                facecolor=color, edgecolor=color, alpha=0.8
            ))
            
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
        
        # 🔥 Get pattern name and priority status
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        is_priority = pattern_info.get('priority', False)
        
        # 🌟 Add priority badge to title
        priority_badge = '⭐ PRIORITY ⭐' if is_priority else ''
        
        # Draw enhanced pattern lines
        draw_enhanced_pattern_lines(ax1, chart_df, pattern_info)
        
        # Add support/resistance levels
        draw_support_resistance(ax1, chart_df)
        
        # 🎨 Style main chart with priority badge
        title_color = '#ff00ff' if is_priority else '#ffaa00'
        ax1.set_title(
            f'{priority_badge}\nXAU/USD - Pattern: {pattern_name} | Signal: {trading_signals["action"]}', 
            color=title_color, fontsize=14, fontweight='bold'
        )
        ax1.set_ylabel('Price ($)', color='#ffffff', fontsize=12)
        ax1.tick_params(colors='#ffffff')
        ax1.grid(True, alpha=0.3, color='#444444')
        ax1.legend(loc='upper left', facecolor='#2a2a2a', edgecolor='#444444', 
                  labelcolor='#ffffff', fontsize=9)
        
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
        
        ax1.set_xlim(-1, len(chart_df))
        ax2.set_xlim(-1, len(chart_df))
        
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        
        # 🌟 Add priority watermark
        watermark_text = f"Generated: {timestamp} (Bangkok)"
        if is_priority:
            watermark_text += " | ⭐ PRIORITY PATTERN"
        
        fig.text(0.02, 0.02, watermark_text, 
                color='#888888', fontsize=10)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', facecolor='#1a1a1a', 
                   edgecolor='none', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
        
    except Exception as e:
        print(f"Chart creation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def draw_double_top_on_chart(ax, df):
    """วาด Double Top pattern บนกราฟ"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        # หาจุดสูงสุด 2 จุด
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
               highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            # เรียงตามความสูง แล้วเอา 2 อันดับแรก
            peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
            # เรียงตามเวลา
            peaks = sorted(peaks, key=lambda x: x[0])
            
            top1_idx, top1_price = peaks[0]
            top2_idx, top2_price = peaks[1]
            
            # 🔴 วาดจุด TOP 1 และ TOP 2
            ax.scatter([top1_idx], [top1_price], color='#ff6600', s=200, 
                      marker='v', label='TOP 1', zorder=10, edgecolors='white', linewidths=2)
            ax.scatter([top2_idx], [top2_price], color='#ff3300', s=200, 
                      marker='v', label='TOP 2', zorder=10, edgecolors='white', linewidths=2)
            
            # 📝 เพิ่ม Label
            ax.text(top1_idx, top1_price + 10, '🔴 TOP1', 
                   ha='center', va='bottom', color='#ff6600', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            ax.text(top2_idx, top2_price + 10, '🔴 TOP2', 
                   ha='center', va='bottom', color='#ff3300', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            # 🟢 หา Valley (จุดต่ำระหว่าง 2 tops)
            valley_start = min(top1_idx, top2_idx)
            valley_end = max(top1_idx, top2_idx)
            
            if valley_end > valley_start:
                valley_idx = valley_start + np.argmin(lows[valley_start:valley_end])
                valley_price = lows[valley_idx]
                
                # วาดจุด Valley
                ax.scatter([valley_idx], [valley_price], color='#00ff88', s=180, 
                          marker='^', label='Valley (Support)', zorder=10, 
                          edgecolors='white', linewidths=2)
                
                ax.text(valley_idx, valley_price - 10, '🟢 VALLEY', 
                       ha='center', va='top', color='#00ff88', 
                       fontweight='bold', fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
                
                # 📏 วาดเส้น Resistance (เชื่อม 2 tops)
                ax.plot([top1_idx, top2_idx], [top1_price, top2_price], 
                       color='#ff6600', linestyle='--', linewidth=3, 
                       alpha=0.9, label='Resistance Line')
                
                # 📏 วาดเส้น Support (ที่ Valley)
                ax.axhline(y=valley_price, xmin=valley_start/len(df), xmax=valley_end/len(df),
                          color='#00ff88', linestyle='-', linewidth=3, 
                          alpha=0.8, label='Support Level')
                
                # 🎯 คำนวณและแสดง Target
                target_distance = (top1_price + top2_price) / 2 - valley_price
                target_price = valley_price - target_distance
                
                ax.axhline(y=target_price, color='#ff0000', linestyle=':', 
                          linewidth=2, alpha=0.7, label=f'Target: ${target_price:.2f}')
                
                ax.text(len(df) - 5, target_price, f'🎯 Target\n${target_price:.2f}', 
                       ha='right', va='center', color='#ff0000', 
                       fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
                
                print(f"✅ Double Top drawn: TOP1={top1_price:.2f}, TOP2={top2_price:.2f}, Valley={valley_price:.2f}")
            else:
                print(f"⚠️ Valley calculation error: valley_end ({valley_end}) <= valley_start ({valley_start})")
        else:
            print(f"⚠️ Not enough peaks found for Double Top (found {len(peaks)})")
            
    except Exception as e:
        print(f"❌ Draw Double Top error: {e}")
        import traceback
        traceback.print_exc()
        
def draw_double_bottom_on_chart(ax, df):
    """วาด Double Bottom pattern บนกราฟ"""
    try:
        lows = df['low'].values
        highs = df['high'].values
        
        # หาจุดต่ำสุด 2 จุด
        troughs = []
        for i in range(2, len(lows)-2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and \
               lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                troughs.append((i, lows[i]))
        
        if len(troughs) >= 2:
            troughs = sorted(troughs, key=lambda x: x[1])[:2]
            troughs = sorted(troughs, key=lambda x: x[0])
            
            bot1_idx, bot1_price = troughs[0]
            bot2_idx, bot2_price = troughs[1]
            
            # 🟢 วาดจุด BOTTOM 1 และ BOTTOM 2
            ax.scatter([bot1_idx], [bot1_price], color='#00ff88', s=200, 
                      marker='^', label='BOTTOM 1', zorder=10, edgecolors='white', linewidths=2)
            ax.scatter([bot2_idx], [bot2_price], color='#00dd66', s=200, 
                      marker='^', label='BOTTOM 2', zorder=10, edgecolors='white', linewidths=2)
            
            ax.text(bot1_idx, bot1_price - 10, '🟢 BOT1', 
                   ha='center', va='top', color='#00ff88', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            ax.text(bot2_idx, bot2_price - 10, '🟢 BOT2', 
                   ha='center', va='top', color='#00dd66', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            # 🔴 หา Peak
            peak_start = min(bot1_idx, bot2_idx)
            peak_end = max(bot1_idx, bot2_idx)
            
            if peak_end > peak_start:
                peak_idx = peak_start + np.argmax(highs[peak_start:peak_end])
                peak_price = highs[peak_idx]
                
                ax.scatter([peak_idx], [peak_price], color='#ff4444', s=180, 
                          marker='v', label='Peak (Resistance)', zorder=10,
                          edgecolors='white', linewidths=2)
                
                ax.text(peak_idx, peak_price + 10, '🔴 PEAK', 
                       ha='center', va='bottom', color='#ff4444', 
                       fontweight='bold', fontsize=12,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
                
                # วาดเส้น Support และ Resistance
                ax.plot([bot1_idx, bot2_idx], [bot1_price, bot2_price], 
                       color='#00ff88', linestyle='--', linewidth=3, 
                       alpha=0.9, label='Support Line')
                
                ax.axhline(y=peak_price, xmin=peak_start/len(df), xmax=peak_end/len(df),
                          color='#ff4444', linestyle='-', linewidth=3, 
                          alpha=0.8, label='Resistance Level')
                
                # Target
                target_distance = peak_price - (bot1_price + bot2_price) / 2
                target_price = peak_price + target_distance
                
                ax.axhline(y=target_price, color='#00ff00', linestyle=':', 
                          linewidth=2, alpha=0.7, label=f'Target: ${target_price:.2f}')
                
                print(f"✅ Double Bottom drawn: BOT1={bot1_price:.2f}, BOT2={bot2_price:.2f}, Peak={peak_price:.2f}")
                
    except Exception as e:
        print(f"❌ Draw Double Bottom error: {e}")

def draw_head_shoulders_on_chart(ax, df):
    """วาด Head & Shoulders pattern บนกราฟ"""
    try:
        highs = df['high'].values
        lows = df['low'].values 
        
        if len(highs) >= 20:
            mid_point = len(highs) // 2
            
            # หาจุด Left Shoulder, Head, Right Shoulder
            left_shoulder_idx = max(0, mid_point - 10) + np.argmax(highs[max(0, mid_point-10):mid_point])
            head_idx = mid_point - 5 + np.argmax(highs[mid_point-5:mid_point+5])
            right_shoulder_idx = mid_point + np.argmax(highs[mid_point:min(len(highs), mid_point+10)])
            
            ls_price = highs[left_shoulder_idx]
            head_price = highs[head_idx]
            rs_price = highs[right_shoulder_idx]
            
            # วาดจุด
            ax.scatter([left_shoulder_idx], [ls_price], color='#ff00ff', s=180, 
                      marker='^', label='Left Shoulder', zorder=10, edgecolors='white', linewidths=2)
            ax.scatter([head_idx], [head_price], color='#ff0000', s=220, 
                      marker='^', label='Head', zorder=10, edgecolors='white', linewidths=2)
            ax.scatter([right_shoulder_idx], [rs_price], color='#ff00ff', s=180, 
                      marker='^', label='Right Shoulder', zorder=10, edgecolors='white', linewidths=2)
            
            # Labels
            ax.text(left_shoulder_idx, ls_price + 8, '💜 LS', 
                   ha='center', va='bottom', color='#ff00ff', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            ax.text(head_idx, head_price + 8, '🔴 HEAD', 
                   ha='center', va='bottom', color='#ff0000', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            ax.text(right_shoulder_idx, rs_price + 8, '💜 RS', 
                   ha='center', va='bottom', color='#ff00ff', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
            
            # Neckline
            neckline_y = (ls_price + rs_price) / 2
            ax.axhline(y=neckline_y, color='#00ffff', linestyle='--', 
                      linewidth=2, alpha=0.8, label='Neckline')
            
            # Target
            target_price = neckline_y - (head_price - neckline_y)
            ax.axhline(y=target_price, color='#ff0000', linestyle=':', 
                      linewidth=2, alpha=0.7, label=f'Target: ${target_price:.2f}')
            
            print(f"✅ Head & Shoulders drawn: LS={ls_price:.2f}, Head={head_price:.2f}, RS={rs_price:.2f}")
            
    except Exception as e:
        print(f"❌ Draw Head & Shoulders error: {e}")

def draw_bull_flag_on_chart(ax, df):
    """วาด Bull Flag Pattern บนกราฟ"""
    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 25:
            print("⚠️ Not enough data for Bull Flag")
            return
        
        # 📊 Flagpole (การขึ้นแรง)
        flagpole_start_idx = len(closes) - 25
        flagpole_end_idx = len(closes) - 15
        
        flagpole_start_price = closes[flagpole_start_idx]
        flagpole_end_price = closes[flagpole_end_idx]
        
        # ตรวจสอบว่าเป็นการขึ้นแรงจริง
        if flagpole_end_price <= flagpole_start_price:
            print("⚠️ No strong upward move for Bull Flag")
            return
        
        # 🚩 Flag (consolidation)
        flag_start_idx = flagpole_end_idx
        flag_end_idx = len(closes) - 1
        
        flag_highs = highs[flag_start_idx:flag_end_idx+1]
        flag_lows = lows[flag_start_idx:flag_end_idx+1]
        
        flag_top = np.max(flag_highs)
        flag_bottom = np.min(flag_lows)
        
        # 📏 วาด Flagpole
        ax.plot([flagpole_start_idx, flagpole_end_idx], 
               [flagpole_start_price, flagpole_end_price],
               color='#00ff88', linewidth=5, alpha=0.9, 
               label='Flagpole (Strong Move)', zorder=10)
        
        # วาดจุดเริ่มต้นและจุดสิ้นสุด Flagpole
        ax.scatter([flagpole_start_idx], [flagpole_start_price], 
                  color='#00ff88', s=200, marker='o', 
                  edgecolors='white', linewidths=3, zorder=15)
        
        ax.scatter([flagpole_end_idx], [flagpole_end_price], 
                  color='#00ff88', s=200, marker='o',
                  edgecolors='white', linewidths=3, zorder=15)
        
        ax.text(flagpole_start_idx, flagpole_start_price - 10, 
               '🟢 START', ha='center', va='top',
               color='#00ff88', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        # 🚩 วาดขอบบน-ล่างของ Flag
        ax.axhline(y=flag_top, xmin=flag_start_idx/len(closes), 
                  xmax=flag_end_idx/len(closes),
                  color='#ffaa00', linestyle='--', linewidth=2.5, 
                  alpha=0.8, label='Flag Top')
        
        ax.axhline(y=flag_bottom, xmin=flag_start_idx/len(closes),
                  xmax=flag_end_idx/len(closes),
                  color='#ffaa00', linestyle='--', linewidth=2.5,
                  alpha=0.8, label='Flag Bottom')
        
        # 📐 คำนวณและวาด Target
        flagpole_height = flagpole_end_price - flagpole_start_price
        target_price = flagpole_end_price + flagpole_height
        
        ax.axhline(y=target_price, color='#00ff00', linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Target: ${target_price:.2f}')
        
        ax.text(len(closes) - 2, target_price, 
               f'🎯 Target\n${target_price:.2f}', 
               ha='right', va='center',
               color='#00ff00', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#00ff00',
                        alpha=0.9, linewidth=2))
        
        # 🚩 Label
        flag_mid_y = (flag_top + flag_bottom) / 2
        ax.text(flag_start_idx + (flag_end_idx - flag_start_idx)/2, 
               flag_mid_y,
               '🚩 FLAG', ha='center', va='center',
               color='#ffaa00', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#ffaa00',
                        alpha=0.9, linewidth=2))
        
        print(f"✅ Bull Flag drawn: Flagpole={flagpole_height:.2f}, Target={target_price:.2f}")
        
    except Exception as e:
        print(f"❌ Draw Bull Flag error: {e}")
        import traceback
        traceback.print_exc()

def draw_bear_flag_on_chart(ax, df):
    """วาด Bear Flag Pattern บนกราฟ"""
    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 25:
            print("⚠️ Not enough data for Bear Flag")
            return
        
        # 📊 Flagpole (การลงแรง)
        flagpole_start_idx = len(closes) - 25
        flagpole_end_idx = len(closes) - 15
        
        flagpole_start_price = closes[flagpole_start_idx]
        flagpole_end_price = closes[flagpole_end_idx]
        
        # ตรวจสอบว่าเป็นการลงแรงจริง
        if flagpole_end_price >= flagpole_start_price:
            print("⚠️ No strong downward move for Bear Flag")
            return
        
        # 🚩 Flag (consolidation)
        flag_start_idx = flagpole_end_idx
        flag_end_idx = len(closes) - 1
        
        flag_highs = highs[flag_start_idx:flag_end_idx+1]
        flag_lows = lows[flag_start_idx:flag_end_idx+1]
        
        flag_top = np.max(flag_highs)
        flag_bottom = np.min(flag_lows)
        
        # 📏 วาด Flagpole
        ax.plot([flagpole_start_idx, flagpole_end_idx], 
               [flagpole_start_price, flagpole_end_price],
               color='#ff4444', linewidth=5, alpha=0.9, 
               label='Flagpole (Strong Drop)', zorder=10)
        
        # วาดจุดเริ่มต้นและจุดสิ้นสุด Flagpole
        ax.scatter([flagpole_start_idx], [flagpole_start_price], 
                  color='#ff4444', s=200, marker='o', 
                  edgecolors='white', linewidths=3, zorder=15)
        
        ax.scatter([flagpole_end_idx], [flagpole_end_price], 
                  color='#ff4444', s=200, marker='o',
                  edgecolors='white', linewidths=3, zorder=15)
        
        ax.text(flagpole_start_idx, flagpole_start_price + 10, 
               '🔴 START', ha='center', va='bottom',
               color='#ff4444', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        # 🚩 วาดขอบบน-ล่างของ Flag
        ax.axhline(y=flag_top, xmin=flag_start_idx/len(closes), 
                  xmax=flag_end_idx/len(closes),
                  color='#ff9900', linestyle='--', linewidth=2.5, 
                  alpha=0.8, label='Flag Top')
        
        ax.axhline(y=flag_bottom, xmin=flag_start_idx/len(closes),
                  xmax=flag_end_idx/len(closes),
                  color='#ff9900', linestyle='--', linewidth=2.5,
                  alpha=0.8, label='Flag Bottom')
        
        # 📐 คำนวณและวาด Target
        flagpole_height = flagpole_start_price - flagpole_end_price
        target_price = flagpole_end_price - flagpole_height
        
        ax.axhline(y=target_price, color='#ff0000', linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Target: ${target_price:.2f}')
        
        ax.text(len(closes) - 2, target_price, 
               f'🎯 Target\n${target_price:.2f}', 
               ha='right', va='center',
               color='#ff0000', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#ff0000',
                        alpha=0.9, linewidth=2))
        
        # 🚩 Label
        flag_mid_y = (flag_top + flag_bottom) / 2
        ax.text(flag_start_idx + (flag_end_idx - flag_start_idx)/2, 
               flag_mid_y,
               '🚩 FLAG', ha='center', va='center',
               color='#ff9900', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#ff9900',
                        alpha=0.9, linewidth=2))
        
        print(f"✅ Bear Flag drawn: Flagpole={flagpole_height:.2f}, Target={target_price:.2f}")
        
    except Exception as e:
        print(f"❌ Draw Bear Flag error: {e}")
        import traceback
        traceback.print_exc()

def draw_symmetrical_triangle_on_chart(ax, df):
    """วาด Symmetrical Triangle บนกราฟ"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) < 30:
            print("⚠️ Not enough data for Symmetrical Triangle")
            return
        
        # หาจุดสูงและต่ำ
        high_points = []
        low_points = []
        
        for i in range(5, len(highs) - 5):
            # Swing high
            if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
               all(highs[i] >= highs[i+j] for j in range(1, 6)):
                high_points.append((i, highs[i]))
            
            # Swing low
            if all(lows[i] <= lows[i-j] for j in range(1, 6)) and \
               all(lows[i] <= lows[i+j] for j in range(1, 6)):
                low_points.append((i, lows[i]))
        
        # ต้องมีอย่างน้อย 2 จุดสูงและ 2 จุดต่ำ
        if len(high_points) < 2 or len(low_points) < 2:
            print(f"⚠️ Not enough swing points (highs={len(high_points)}, lows={len(low_points)})")
            return
        
        # เอาจุดล่าสุด
        recent_highs = high_points[-3:] if len(high_points) >= 3 else high_points[-2:]
        recent_lows = low_points[-3:] if len(low_points) >= 3 else low_points[-2:]
        
        # 📉 วาดเส้น Descending Resistance
        if len(recent_highs) >= 2:
            h1, h2 = recent_highs[0], recent_highs[-1]
            
            # วาดเส้นแนวโน้มขาลง
            ax.plot([h1[0], h2[0]], [h1[1], h2[1]], 
                   color='#ff6600', linestyle='-', linewidth=3,
                   alpha=0.9, label='Descending Resistance', zorder=10)
            
            # วาดจุด
            for idx, price in recent_highs:
                ax.scatter([idx], [price], color='#ff6600', s=150, 
                          marker='v', edgecolors='white', linewidths=2, zorder=15)
        
        # 📈 วาดเส้น Ascending Support
        if len(recent_lows) >= 2:
            l1, l2 = recent_lows[0], recent_lows[-1]
            
            # วาดเส้นแนวโน้มขาขึ้น
            ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 
                   color='#00ff88', linestyle='-', linewidth=3,
                   alpha=0.9, label='Ascending Support', zorder=10)
            
            # วาดจุด
            for idx, price in recent_lows:
                ax.scatter([idx], [price], color='#00ff88', s=150, 
                          marker='^', edgecolors='white', linewidths=2, zorder=15)
        
        # 🎯 วาดจุด Apex (จุดบรรจบ)
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # คำนวณจุดบรรจบโดยประมาณ
            h_slope = (recent_highs[-1][1] - recent_highs[0][1]) / (recent_highs[-1][0] - recent_highs[0][0])
            l_slope = (recent_lows[-1][1] - recent_lows[0][1]) / (recent_lows[-1][0] - recent_lows[0][0])
            
            # ถ้า slopes ลู่เข้าหากัน
            if h_slope < 0 and l_slope > 0:
                # ประมาณจุด apex
                apex_x = len(df) + 10  # คาดว่าจะบรรจบในอนาคต
                apex_price = (recent_highs[-1][1] + recent_lows[-1][1]) / 2
                
                # วาดเส้นประไปยัง apex
                ax.plot([recent_highs[-1][0], apex_x], 
                       [recent_highs[-1][1], apex_price],
                       color='#ff6600', linestyle=':', linewidth=2, alpha=0.6)
                
                ax.plot([recent_lows[-1][0], apex_x], 
                       [recent_lows[-1][1], apex_price],
                       color='#00ff88', linestyle=':', linewidth=2, alpha=0.6)
                
                ax.scatter([apex_x], [apex_price], color='#ffff00', 
                          s=250, marker='*', edgecolors='white', 
                          linewidths=2, label='Apex (Breakout Point)', zorder=15)
        
        # 📊 Label
        mid_x = len(df) - 10
        mid_y = (highs[mid_x] + lows[mid_x]) / 2
        
        ax.text(mid_x, mid_y, 
               '⚖️ SYMMETRICAL\nTRIANGLE', 
               ha='center', va='center',
               color='#ffaa00', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.6', 
                        facecolor='black', edgecolor='#ffaa00',
                        alpha=0.9, linewidth=2))
        
        print("✅ Symmetrical Triangle drawn")
        
    except Exception as e:
        print(f"❌ Draw Symmetrical Triangle error: {e}")
        import traceback
        traceback.print_exc()

def draw_pennant_on_chart(ax, df):
    """วาด Pennant Pattern บนกราฟ"""
    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 25:
            print("⚠️ Not enough data for Pennant")
            return
        
        # 📊 Flagpole (การเคลื่อนไหวแรง)
        flagpole_start_idx = len(closes) - 25
        flagpole_end_idx = len(closes) - 12
        
        flagpole_start_price = closes[flagpole_start_idx]
        flagpole_end_price = closes[flagpole_end_idx]
        
        flagpole_move = abs(flagpole_end_price - flagpole_start_price)
        
        # ตรวจสอบว่ามีการเคลื่อนไหวแรงพอ
        if flagpole_move < flagpole_start_price * 0.03:
            print("⚠️ Not enough strong move for Pennant")
            return
        
        # 🎏 Pennant (รูปสามเหลี่ยมเล็ก)
        pennant_start_idx = flagpole_end_idx
        pennant_end_idx = len(closes) - 1
        
        pennant_highs = highs[pennant_start_idx:pennant_end_idx+1]
        pennant_lows = lows[pennant_start_idx:pennant_end_idx+1]
        
        # หาแนวโน้มของ pennant
        pennant_high_trend = []
        pennant_low_trend = []
        
        for i in range(len(pennant_highs)):
            if i > 0 and pennant_highs[i] > pennant_highs[i-1]:
                pennant_high_trend.append((pennant_start_idx + i, pennant_highs[i]))
            if i > 0 and pennant_lows[i] < pennant_lows[i-1]:
                pennant_low_trend.append((pennant_start_idx + i, pennant_lows[i]))
        
        # 📏 วาด Flagpole
        pole_color = '#00ff88' if flagpole_end_price > flagpole_start_price else '#ff4444'
        
        ax.plot([flagpole_start_idx, flagpole_end_idx], 
               [flagpole_start_price, flagpole_end_price],
               color=pole_color, linewidth=5, alpha=0.9, 
               label='Strong Move', zorder=10)
        
        ax.scatter([flagpole_start_idx], [flagpole_start_price], 
                  color=pole_color, s=200, marker='o', 
                  edgecolors='white', linewidths=3, zorder=15)
        
        # 🎏 วาด Pennant (converging lines)
        # Upper line
        pennant_top_start = np.max(pennant_highs[:3]) if len(pennant_highs) >= 3 else pennant_highs[0]
        pennant_top_end = np.max(pennant_highs[-3:]) if len(pennant_highs) >= 3 else pennant_highs[-1]
        
        ax.plot([pennant_start_idx, pennant_end_idx], 
               [pennant_top_start, pennant_top_end],
               color='#ffaa00', linestyle='-', linewidth=2.5,
               alpha=0.8, label='Pennant Upper', zorder=10)
        
        # Lower line
        pennant_bottom_start = np.min(pennant_lows[:3]) if len(pennant_lows) >= 3 else pennant_lows[0]
        pennant_bottom_end = np.min(pennant_lows[-3:]) if len(pennant_lows) >= 3 else pennant_lows[-1]
        
        ax.plot([pennant_start_idx, pennant_end_idx], 
               [pennant_bottom_start, pennant_bottom_end],
               color='#ffaa00', linestyle='-', linewidth=2.5,
               alpha=0.8, label='Pennant Lower', zorder=10)
        
        # 📐 คำนวณ Target
        target_price = flagpole_end_price + (flagpole_move if pole_color == '#00ff88' else -flagpole_move)
        
        ax.axhline(y=target_price, color=pole_color, linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Target: ${target_price:.2f}')
        
        # 🎏 Label
        pennant_mid_x = pennant_start_idx + (pennant_end_idx - pennant_start_idx) / 2
        pennant_mid_y = (pennant_top_start + pennant_bottom_start) / 2
        
        ax.text(pennant_mid_x, pennant_mid_y, 
               '🎏 PENNANT', ha='center', va='center',
               color='#ffaa00', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#ffaa00',
                        alpha=0.9, linewidth=2))
        
        print(f"✅ Pennant drawn: Move={flagpole_move:.2f}, Target={target_price:.2f}")
        
    except Exception as e:
        print(f"❌ Draw Pennant error: {e}")
        import traceback
        traceback.print_exc()

def draw_wedge_rising_on_chart(ax, df):
    """วาด Rising Wedge Pattern (Bearish) บนกราฟ - IMPROVED VERSION"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) < 20:  # ✅ ลด requirement
            print("⚠️ Not enough data for Rising Wedge")
            return
        
        # ✅ ใช้ข้อมูลทั้งหมดที่มี แทนที่จะจำกัดแค่ 30 bars
        lookback = min(40, len(highs))
        start_idx = len(highs) - lookback
        
        # ✅ หาจุดสูงและต่ำที่สำคัญ (ผ่อนเงื่อนไข)
        high_points = []
        low_points = []
        
        for i in range(start_idx + 3, len(highs) - 3):
            # ✅ Swing high - ผ่อนเงื่อนไข (ใช้ 1 แท่งข้างเคียงแทน 2)
            if highs[i] >= highs[i-1] and highs[i] >= highs[i+1]:
                # ตรวจสอบเพิ่มเติมว่าเป็น local maximum
                is_high = True
                for j in range(max(0, i-3), min(len(highs), i+4)):
                    if j != i and highs[j] > highs[i]:
                        is_high = False
                        break
                
                if is_high:
                    high_points.append((i, highs[i]))
            
            # ✅ Swing low - ผ่อนเงื่อนไข
            if lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                # ตรวจสอบเพิ่มเติมว่าเป็น local minimum
                is_low = True
                for j in range(max(0, i-3), min(len(lows), i+4)):
                    if j != i and lows[j] < lows[i]:
                        is_low = False
                        break
                
                if is_low:
                    low_points.append((i, lows[i]))
        
        print(f"🔍 Rising Wedge: Found {len(high_points)} highs, {len(low_points)} lows")
        
        if len(high_points) < 2 or len(low_points) < 2:
            print(f"⚠️ Not enough swing points for Rising Wedge")
            return
        
        # ✅ เอา 2-3 จุดล่าสุดของแต่ละด้าน
        recent_highs = high_points[-3:] if len(high_points) >= 3 else high_points[-2:]
        recent_lows = low_points[-3:] if len(low_points) >= 3 else low_points[-2:]
        
        # ✅ คำนวณ Slope (ผ่อนเงื่อนไข)
        h1, h_last = recent_highs[0], recent_highs[-1]
        l1, l_last = recent_lows[0], recent_lows[-1]
        
        h_slope = (h_last[1] - h1[1]) / (h_last[0] - h1[0]) if h_last[0] != h1[0] else 0
        l_slope = (l_last[1] - l1[1]) / (l_last[0] - l1[0]) if l_last[0] != l1[0] else 0
        
        print(f"   Slopes: h_slope={h_slope:.4f}, l_slope={l_slope:.4f}")
        
        # ✅ ตรวจสอบว่าเป็น Rising Wedge (ทั้งสองขาขึ้น แต่ support ขึ้นชันกว่า)
        # ผ่อนเงื่อนไข: อนุญาตให้ slope เล็กน้อย
        if h_slope < -0.1 or l_slope < -0.1:  # ถ้าลงชันมาก = ไม่ใช่ rising
            print(f"⚠️ Not a valid Rising Wedge (slopes descending)")
            return
        
        # ✅ ตรวจสอบว่า support ขึ้นชันกว่า resistance (หรือใกล้เคียงกัน)
        # Rising Wedge: ทั้งสองเส้นขาขึ้น และลู่เข้าหากัน
        if not (l_slope >= h_slope * 0.7):  # support ต้องขึ้นชันกว่า หรือใกล้เคียง
            print(f"⚠️ Support not steeper than resistance")
            return
        
        # ✅ ตรวจสอบว่าเส้นลู่เข้าหากัน
        start_width = h1[1] - l1[1]
        end_width = h_last[1] - l_last[1]
        
        if end_width >= start_width:  # ถ้าช่วงห่างเพิ่มขึ้น = ไม่ได้ลู่เข้า
            print(f"⚠️ Lines not converging (start={start_width:.2f}, end={end_width:.2f})")
            return
        
        # 📈 วาดเส้น Resistance (ขาบน)
        ax.plot([h1[0], h_last[0]], [h1[1], h_last[1]], 
               color='#ff6600', linestyle='-', linewidth=3,
               alpha=0.9, label='Rising Resistance', zorder=10)
        
        # ✅ วาดจุด Resistance พร้อม marker ชัดเจน
        for idx, price in recent_highs:
            ax.scatter([idx], [price], color='#ff6600', s=200, 
                      marker='v', edgecolors='white', linewidths=3, zorder=15)
            
            # เพิ่ม label ที่แต่ละจุด
            ax.text(idx, price + 8, 'R', 
                   ha='center', va='bottom',
                   color='#ff6600', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='circle,pad=0.3', 
                            facecolor='#ff6600', 
                            edgecolor='white',
                            alpha=0.9, linewidth=2))
        
        # 📈 วาดเส้น Support (ขาล่าง - ขึ้นชัน)
        ax.plot([l1[0], l_last[0]], [l1[1], l_last[1]], 
               color='#00ff88', linestyle='-', linewidth=3,
               alpha=0.9, label='Rising Support (Steeper)', zorder=10)
        
        # ✅ วาดจุด Support พร้อม marker ชัดเจน
        for idx, price in recent_lows:
            ax.scatter([idx], [price], color='#00ff88', s=200, 
                      marker='^', edgecolors='white', linewidths=3, zorder=15)
            
            # เพิ่ม label ที่แต่ละจุด
            ax.text(idx, price - 8, 'S', 
                   ha='center', va='top',
                   color='#00ff88', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='circle,pad=0.3', 
                            facecolor='#00ff88',
                            edgecolor='white',
                            alpha=0.9, linewidth=2))
        
        # 🎯 คำนวณ Target (Bearish breakdown)
        wedge_height = h_last[1] - l_last[1]
        target_price = l_last[1] - wedge_height
        
        ax.axhline(y=target_price, color='#ff0000', linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Breakdown Target: ${target_price:.2f}')
        
        ax.text(len(df) - 2, target_price, 
               f'🎯 Target\n${target_price:.2f}', 
               ha='right', va='center',
               color='#ff0000', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#ff0000',
                        alpha=0.9, linewidth=2))
        
        # 📊 Main Label พร้อมข้อมูล Slope
        mid_x = (h1[0] + h_last[0]) / 2
        mid_y = (h_last[1] + l_last[1]) / 2
        
        convergence_info = f"Converging: {start_width:.2f}→{end_width:.2f}"
        
        ax.text(mid_x, mid_y, 
               f'📐 RISING WEDGE\n(Bearish)\n{convergence_info}', 
               ha='center', va='center',
               color='#ff6600', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.6', 
                        facecolor='black', edgecolor='#ff6600',
                        alpha=0.95, linewidth=2))
        
        print(f"✅ Rising Wedge drawn successfully!")
        print(f"   Resistance points: {len(recent_highs)}, Support points: {len(recent_lows)}")
        print(f"   Convergence: {start_width:.2f} → {end_width:.2f} ({((end_width/start_width-1)*100):.1f}%)")
        
    except Exception as e:
        print(f"❌ Draw Rising Wedge error: {e}")
        import traceback
        traceback.print_exc()

def draw_wedge_falling_on_chart(ax, df):
    """วาด Falling Wedge Pattern (Bullish) บนกราฟ - IMPROVED VERSION"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) < 20:  # ✅ ลด requirement
            print("⚠️ Not enough data for Falling Wedge")
            return
        
        # ✅ ใช้ข้อมูลทั้งหมดที่มี แทนที่จะจำกัดแค่ 30 bars
        lookback = min(40, len(highs))
        start_idx = len(highs) - lookback
        
        # ✅ หาจุดสูงและต่ำที่สำคัญ (ผ่อนเงื่อนไข)
        high_points = []
        low_points = []
        
        for i in range(start_idx + 3, len(highs) - 3):
            # ✅ Swing high - ผ่อนเงื่อนไข (ใช้ 1 แท่งข้างเคียงแทน 2)
            if highs[i] >= highs[i-1] and highs[i] >= highs[i+1]:
                # ตรวจสอบเพิ่มเติมว่าเป็น local maximum
                is_high = True
                for j in range(max(0, i-3), min(len(highs), i+4)):
                    if j != i and highs[j] > highs[i]:
                        is_high = False
                        break
                
                if is_high:
                    high_points.append((i, highs[i]))
            
            # ✅ Swing low - ผ่อนเงื่อนไข
            if lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                # ตรวจสอบเพิ่มเติมว่าเป็น local minimum
                is_low = True
                for j in range(max(0, i-3), min(len(lows), i+4)):
                    if j != i and lows[j] < lows[i]:
                        is_low = False
                        break
                
                if is_low:
                    low_points.append((i, lows[i]))
        
        print(f"🔍 Falling Wedge: Found {len(high_points)} highs, {len(low_points)} lows")
        
        if len(high_points) < 2 or len(low_points) < 2:
            print(f"⚠️ Not enough swing points for Falling Wedge")
            return
        
        # ✅ เอา 2-3 จุดล่าสุดของแต่ละด้าน
        recent_highs = high_points[-3:] if len(high_points) >= 3 else high_points[-2:]
        recent_lows = low_points[-3:] if len(low_points) >= 3 else low_points[-2:]
        
        # ✅ คำนวณ Slope (ผ่อนเงื่อนไข)
        h1, h_last = recent_highs[0], recent_highs[-1]
        l1, l_last = recent_lows[0], recent_lows[-1]
        
        h_slope = (h_last[1] - h1[1]) / (h_last[0] - h1[0]) if h_last[0] != h1[0] else 0
        l_slope = (l_last[1] - l1[1]) / (l_last[0] - l1[0]) if l_last[0] != l1[0] else 0
        
        print(f"   Slopes: h_slope={h_slope:.4f}, l_slope={l_slope:.4f}")
        
        # ✅ ตรวจสอบว่าเป็น Falling Wedge (ทั้งสองขาลง แต่ resistance ลงชันกว่า)
        # ผ่อนเงื่อนไข: อนุญาตให้ slope เล็กน้อย
        if h_slope > 0.1 or l_slope > 0.1:  # ถ้าขึ้นชันมาก = ไม่ใช่ falling
            print(f"⚠️ Not a valid Falling Wedge (slopes ascending)")
            return
        
        # ✅ ตรวจสอบว่า resistance ลงชันกว่า support (หรือใกล้เคียงกัน)
        # Falling Wedge: ทั้งสองเส้นขาลง และลู่เข้าหากัน
        if not (h_slope <= l_slope * 0.7):  # resistance ต้องลงชันกว่า หรือใกล้เคียง
            print(f"⚠️ Resistance not steeper than support")
            return
        
        # ✅ ตรวจสอบว่าเส้นลู่เข้าหากัน
        start_width = h1[1] - l1[1]
        end_width = h_last[1] - l_last[1]
        
        if end_width >= start_width:  # ถ้าช่วงห่างเพิ่มขึ้น = ไม่ได้ลู่เข้า
            print(f"⚠️ Lines not converging (start={start_width:.2f}, end={end_width:.2f})")
            return
        
        # 📉 วาดเส้น Resistance (ขาบน - ลงชัน)
        ax.plot([h1[0], h_last[0]], [h1[1], h_last[1]], 
               color='#ff6600', linestyle='-', linewidth=3,
               alpha=0.9, label='Falling Resistance (Steeper)', zorder=10)
        
        # ✅ วาดจุด Resistance พร้อม marker ชัดเจน
        for idx, price in recent_highs:
            ax.scatter([idx], [price], color='#ff6600', s=200, 
                      marker='v', edgecolors='white', linewidths=3, zorder=15)
            
            # เพิ่ม label ที่แต่ละจุด
            ax.text(idx, price + 8, 'R', 
                   ha='center', va='bottom',
                   color='#ff6600', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='circle,pad=0.3', 
                            facecolor='#ff6600', 
                            edgecolor='white',
                            alpha=0.9, linewidth=2))
        
        # 📉 วาดเส้น Support (ขาล่าง - ลงช้า)
        ax.plot([l1[0], l_last[0]], [l1[1], l_last[1]], 
               color='#00ff88', linestyle='-', linewidth=3,
               alpha=0.9, label='Falling Support', zorder=10)
        
        # ✅ วาดจุด Support พร้อม marker ชัดเจน
        for idx, price in recent_lows:
            ax.scatter([idx], [price], color='#00ff88', s=200, 
                      marker='^', edgecolors='white', linewidths=3, zorder=15)
            
            # เพิ่ม label ที่แต่ละจุด
            ax.text(idx, price - 8, 'S', 
                   ha='center', va='top',
                   color='#00ff88', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='circle,pad=0.3', 
                            facecolor='#00ff88',
                            edgecolor='white',
                            alpha=0.9, linewidth=2))
        
        # 🎯 คำนวณ Target (Bullish breakout)
        wedge_height = h_last[1] - l_last[1]
        target_price = h_last[1] + wedge_height
        
        ax.axhline(y=target_price, color='#00ff00', linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Breakout Target: ${target_price:.2f}')
        
        ax.text(len(df) - 2, target_price, 
               f'🎯 Target\n${target_price:.2f}', 
               ha='right', va='center',
               color='#00ff00', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#00ff00',
                        alpha=0.9, linewidth=2))
        
        # 📊 Main Label พร้อมข้อมูล Slope
        mid_x = (h1[0] + h_last[0]) / 2
        mid_y = (h_last[1] + l_last[1]) / 2
        
        convergence_info = f"Converging: {start_width:.2f}→{end_width:.2f}"
        
        ax.text(mid_x, mid_y, 
               f'📐 FALLING WEDGE\n(Bullish)\n{convergence_info}', 
               ha='center', va='center',
               color='#00ff88', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.6', 
                        facecolor='black', edgecolor='#00ff88',
                        alpha=0.95, linewidth=2))
        
        print(f"✅ Falling Wedge drawn successfully!")
        print(f"   Resistance points: {len(recent_highs)}, Support points: {len(recent_lows)}")
        print(f"   Convergence: {start_width:.2f} → {end_width:.2f} ({((end_width/start_width-1)*100):.1f}%)")
        
    except Exception as e:
        print(f"❌ Draw Falling Wedge error: {e}")
        import traceback
        traceback.print_exc()

def draw_cup_and_handle_on_chart(ax, df):
    """วาด Cup and Handle Pattern บนกราฟ"""
    try:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 50:
            print("⚠️ Not enough data for Cup and Handle (need 50+)")
            return
        
        # ☕ Cup formation (ใช้ 40 แท่งสำหรับ cup)
        cup_start_idx = len(closes) - 50
        cup_end_idx = len(closes) - 10
        
        # Left rim
        left_rim_idx = cup_start_idx + np.argmax(highs[cup_start_idx:cup_start_idx+10])
        left_rim_price = highs[left_rim_idx]
        
        # Cup bottom
        cup_bottom_idx = cup_start_idx + 10 + np.argmin(lows[cup_start_idx+10:cup_end_idx-10])
        cup_bottom_price = lows[cup_bottom_idx]
        
        # Right rim
        right_rim_idx = cup_end_idx - 10 + np.argmax(highs[cup_end_idx-10:cup_end_idx])
        right_rim_price = highs[right_rim_idx]
        
        # ตรวจสอบว่า rims ใกล้เคียงกัน
        rim_diff = abs(left_rim_price - right_rim_price) / left_rim_price
        if rim_diff > 0.05:  # ต้องไม่ต่างกันเกิน 5%
            print(f"⚠️ Rims not similar enough (diff={rim_diff:.2%})")
            return
        
        # ☕ วาด Cup
        # วาดจุด Left Rim
        ax.scatter([left_rim_idx], [left_rim_price], 
                  color='#00aaff', s=220, marker='o',
                  edgecolors='white', linewidths=3, label='Left Rim', zorder=15)
        
        ax.text(left_rim_idx, left_rim_price + 10, 
               '☕ LEFT', ha='center', va='bottom',
               color='#00aaff', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        # วาดจุด Cup Bottom
        ax.scatter([cup_bottom_idx], [cup_bottom_price], 
                  color='#0066ff', s=220, marker='o',
                  edgecolors='white', linewidths=3, label='Cup Bottom', zorder=15)
        
        ax.text(cup_bottom_idx, cup_bottom_price - 10, 
               '☕ BOTTOM', ha='center', va='top',
               color='#0066ff', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        # วาดจุด Right Rim
        ax.scatter([right_rim_idx], [right_rim_price], 
                  color='#00aaff', s=220, marker='o',
                  edgecolors='white', linewidths=3, label='Right Rim', zorder=15)
        
        ax.text(right_rim_idx, right_rim_price + 10, 
               '☕ RIGHT', ha='center', va='bottom',
               color='#00aaff', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.9))
        
        # วาดเส้นโค้ง Cup (ใช้เส้นประ)
        cup_x = [left_rim_idx, cup_bottom_idx, right_rim_idx]
        cup_y = [left_rim_price, cup_bottom_price, right_rim_price]
        
        # สร้างเส้นโค้งแบบ smooth
        from scipy import interpolate
        try:
            x_smooth = np.linspace(left_rim_idx, right_rim_idx, 100)
            spl = interpolate.make_interp_spline(cup_x, cup_y, k=2)
            y_smooth = spl(x_smooth)
            
            ax.plot(x_smooth, y_smooth, color='#00aaff', 
                   linewidth=3, alpha=0.8, linestyle='-', zorder=10)
        except:
            # Fallback: ใช้เส้นตรง
            ax.plot(cup_x, cup_y, color='#00aaff', 
                   linewidth=3, alpha=0.8, linestyle='-', zorder=10)
        
        # 🍵 Handle formation (10 แท่งสุดท้าย)
        handle_start_idx = cup_end_idx
        handle_end_idx = len(closes) - 1
        
        handle_highs = highs[handle_start_idx:handle_end_idx+1]
        handle_lows = lows[handle_start_idx:handle_end_idx+1]
        
        handle_high = np.max(handle_highs)
        handle_low = np.min(handle_lows)
        
        # วาดขอบบน-ล่างของ Handle
        ax.axhline(y=handle_high, xmin=handle_start_idx/len(closes), 
                  xmax=handle_end_idx/len(closes),
                  color='#ff9900', linestyle='--', linewidth=2, 
                  alpha=0.8, label='Handle Top')
        
        ax.axhline(y=handle_low, xmin=handle_start_idx/len(closes),
                  xmax=handle_end_idx/len(closes),
                  color='#ff9900', linestyle='--', linewidth=2,
                  alpha=0.8, label='Handle Bottom')
        
        # 🍵 Label Handle
        handle_mid_x = handle_start_idx + (handle_end_idx - handle_start_idx) / 2
        handle_mid_y = (handle_high + handle_low) / 2
        
        ax.text(handle_mid_x, handle_mid_y, 
               '🍵', ha='center', va='center',
               fontsize=30, zorder=15)
        
        # 🎯 Target (Cup depth + breakout point)
        cup_depth = left_rim_price - cup_bottom_price
        target_price = right_rim_price + cup_depth
        
        ax.axhline(y=target_price, color='#00ff00', linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Target: ${target_price:.2f}')
        
        ax.text(len(closes) - 2, target_price, 
               f'🎯 Target\n${target_price:.2f}', 
               ha='right', va='center',
               color='#00ff00', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#00ff00',
                        alpha=0.9, linewidth=2))
        
        # 📊 Main Label
        cup_label_x = (left_rim_idx + right_rim_idx) / 2
        cup_label_y = (left_rim_price + cup_bottom_price) / 2
        
        ax.text(cup_label_x, cup_label_y, 
               '☕ CUP & HANDLE', 
               ha='center', va='center',
               color='#00aaff', fontweight='bold', fontsize=14,
               bbox=dict(boxstyle='round,pad=0.7', 
                        facecolor='black', edgecolor='#00aaff',
                        alpha=0.9, linewidth=2))
        
        print(f"✅ Cup and Handle drawn: Depth={cup_depth:.2f}, Target={target_price:.2f}")
        
    except Exception as e:
        print(f"❌ Draw Cup and Handle error: {e}")
        import traceback
        traceback.print_exc()


def draw_inverse_head_shoulders_on_chart(ax, df):
    """วาด Inverse Head & Shoulders Pattern บนกราฟ - IMPROVED VERSION"""
    try:
        lows = df['low'].values
        highs = df['high'].values
        
        if len(lows) < 20:  # ✅ ลด requirement
            print("⚠️ Not enough data for Inverse H&S")
            return
        
        # ✅ หาจุดต่ำสุด (valleys) ที่สำคัญ
        valleys = []
        lookback = min(40, len(lows))
        start_idx = len(lows) - lookback
        
        for i in range(start_idx + 3, len(lows) - 3):
            # Swing low
            if lows[i] <= lows[i-1] and lows[i] <= lows[i+1]:
                is_valley = True
                for j in range(max(0, i-3), min(len(lows), i+4)):
                    if j != i and lows[j] < lows[i]:
                        is_valley = False
                        break
                
                if is_valley:
                    valleys.append((i, lows[i]))
        
        print(f"🔍 Inverse H&S: Found {len(valleys)} valleys")
        
        if len(valleys) < 3:
            print(f"⚠️ Need at least 3 valleys for Inverse H&S")
            return
        
        # ✅ เรียงตามความลึก (ต่ำสุดคือ Head)
        valleys_sorted_by_depth = sorted(valleys, key=lambda x: x[1])
        
        # เอา 3-5 จุดที่ต่ำที่สุด
        candidate_valleys = valleys_sorted_by_depth[:5] if len(valleys_sorted_by_depth) >= 5 else valleys_sorted_by_depth[:3]
        
        # เรียงตามตำแหน่ง (เวลา)
        candidate_valleys = sorted(candidate_valleys, key=lambda x: x[0])
        
        # หา Head (จุดที่ต่ำที่สุด)
        head_idx, head_price = min(candidate_valleys, key=lambda x: x[1])
        
        # หา Left และ Right Shoulders (จุดที่อยู่ซ้าย-ขวา Head)
        left_shoulders = [v for v in candidate_valleys if v[0] < head_idx]
        right_shoulders = [v for v in candidate_valleys if v[0] > head_idx]
        
        if not left_shoulders or not right_shoulders:
            print("⚠️ Cannot find both shoulders")
            return
        
        # เอาจุดที่ใกล้ Head ที่สุด
        ls_idx, ls_price = max(left_shoulders, key=lambda x: x[0])
        rs_idx, rs_price = min(right_shoulders, key=lambda x: x[0])
        
        # ✅ ตรวจสอบว่า Head ต่ำกว่า Shoulders
        if not (head_price < ls_price and head_price < rs_price):
            print(f"⚠️ Head ({head_price:.2f}) not lower than shoulders (LS:{ls_price:.2f}, RS:{rs_price:.2f})")
            return
        
        # ตรวจสอบว่า Shoulders ใกล้เคียงกัน (ไม่เกิน 5% diff)
        shoulder_diff = abs(ls_price - rs_price) / ls_price
        if shoulder_diff > 0.05:
            print(f"⚠️ Shoulders too different ({shoulder_diff:.1%})")
            return
        
        # 🟢 วาดจุด Left Shoulder
        ax.scatter([ls_idx], [ls_price], 
                  color='#00ff88', s=250, marker='^',
                  edgecolors='white', linewidths=3, label='Left Shoulder', zorder=15)
        
        ax.text(ls_idx, ls_price - 12, 'LS', 
               ha='center', va='top',
               color='#00ff88', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='circle,pad=0.4', 
                        facecolor='#00ff88', 
                        edgecolor='white',
                        alpha=0.95, linewidth=3))
        
        # 🔵 วาดจุด Head
        ax.scatter([head_idx], [head_price], 
                  color='#0066ff', s=300, marker='^',
                  edgecolors='white', linewidths=3, label='Head (Lowest)', zorder=15)
        
        ax.text(head_idx, head_price - 12, 'HEAD', 
               ha='center', va='top',
               color='#0066ff', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='#0066ff', 
                        edgecolor='white',
                        alpha=0.95, linewidth=3))
        
        # 🟢 วาดจุด Right Shoulder
        ax.scatter([rs_idx], [rs_price], 
                  color='#00ff88', s=250, marker='^',
                  edgecolors='white', linewidths=3, label='Right Shoulder', zorder=15)
        
        ax.text(rs_idx, rs_price - 12, 'RS', 
               ha='center', va='top',
               color='#00ff88', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='circle,pad=0.4', 
                        facecolor='#00ff88',
                        edgecolor='white',
                        alpha=0.95, linewidth=3))
        
        # 🔷 Neckline (เส้นต้าน - หา peaks ระหว่าง shoulders)
        left_peak_range = highs[ls_idx:head_idx]
        right_peak_range = highs[head_idx:rs_idx+1]
        
        if len(left_peak_range) > 0 and len(right_peak_range) > 0:
            left_peak_price = np.max(left_peak_range)
            right_peak_price = np.max(right_peak_range)
            
            neckline_y = (left_peak_price + right_peak_price) / 2
            
            ax.axhline(y=neckline_y, xmin=ls_idx/len(lows), 
                      xmax=rs_idx/len(lows),
                      color='#00ffff', linestyle='--', linewidth=3, 
                      alpha=0.9, label=f'Neckline: ${neckline_y:.2f}')
            
            ax.text(len(lows) - 3, neckline_y, 
                   'Neckline', ha='right', va='center',
                   color='#00ffff', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='black', 
                            edgecolor='#00ffff',
                            alpha=0.9, linewidth=2))
        else:
            neckline_y = (ls_price + rs_price) / 2
        
        # 🎯 Target (Bullish breakout)
        hs_height = neckline_y - head_price
        target_price = neckline_y + hs_height
        
        ax.axhline(y=target_price, color='#00ff00', linestyle=':', 
                  linewidth=3, alpha=0.8, label=f'Target: ${target_price:.2f}')
        
        ax.text(len(lows) - 2, target_price, 
               f'🎯 Target\n${target_price:.2f}', 
               ha='right', va='center',
               color='#00ff00', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='black', edgecolor='#00ff00',
                        alpha=0.9, linewidth=2))
        
        # 📊 Main Label
        label_x = head_idx
        label_y = (head_price + neckline_y) / 2
        
        pattern_info = f"Depth: {hs_height:.2f}"
        
        ax.text(label_x, label_y, 
               f'🔄 INVERSE H&S\n(Bullish)\n{pattern_info}', 
               ha='center', va='center',
               color='#00ffff', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.6', 
                        facecolor='black', edgecolor='#00ffff',
                        alpha=0.95, linewidth=2))
        
        print(f"✅ Inverse H&S drawn: Head={head_price:.2f}, LS={ls_price:.2f}, RS={rs_price:.2f}, Target={target_price:.2f}")
        
    except Exception as e:
        print(f"❌ Draw Inverse H&S error: {e}")
        import traceback
        traceback.print_exc()

def draw_rectangle_on_chart(ax, df):
    """วาด Rectangle Pattern (Trading Range) บนกราฟ"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) < 30:
            print("⚠️ Not enough data for Rectangle")
            return
        
        # หาแนว Resistance และ Support (horizontal lines)
        lookback = 30
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Resistance = ค่าเฉลี่ยของจุดสูงสุด
        high_threshold = np.percentile(recent_highs, 90)
        resistance_prices = recent_highs[recent_highs > high_threshold]
        resistance_level = np.mean(resistance_prices) if len(resistance_prices) > 0 else np.max(recent_highs)
        
        # Support = ค่าเฉลี่ยของจุดต่ำสุด
        low_threshold = np.percentile(recent_lows, 10)
        support_prices = recent_lows[recent_lows < low_threshold]
        support_level = np.mean(support_prices) if len(support_prices) > 0 else np.min(recent_lows)
        
        # ตรวจสอบว่ามี rectangle range
        range_size = resistance_level - support_level
        range_pct = range_size / support_level
        
        if not (0.02 < range_pct < 0.15):  # 2-15% range
            print(f"⚠️ Range too small or too large ({range_pct:.1%})")
            return
        
        # หาจุดที่สัมผัส Resistance และ Support
        resistance_touches = []
        support_touches = []
        
        start_idx = len(highs) - lookback
        for i in range(len(recent_highs)):
            actual_idx = start_idx + i
            if abs(recent_highs[i] - resistance_level) < resistance_level * 0.01:
                resistance_touches.append(actual_idx)
            if abs(recent_lows[i] - support_level) < support_level * 0.01:
                support_touches.append(actual_idx)
        
        if len(resistance_touches) < 2 or len(support_touches) < 2:
            print(f"⚠️ Not enough touches (R={len(resistance_touches)}, S={len(support_touches)})")
            return
        
        # 📊 วาดเส้น Resistance
        ax.axhline(y=resistance_level, color='#ff4444', linestyle='-', 
                  linewidth=3, alpha=0.9, label=f'Resistance: ${resistance_level:.2f}', 
                  zorder=10)
        
        # วาดจุดสัมผัส Resistance
        for idx in resistance_touches[:5]:  # แสดงไม่เกิน 5 จุด
            if 0 <= idx < len(highs):
                ax.scatter([idx], [resistance_level], color='#ff4444', 
                          s=120, marker='_', linewidths=3, zorder=15)
        
        # 📊 วาดเส้น Support
        ax.axhline(y=support_level, color='#00ff88', linestyle='-', 
                  linewidth=3, alpha=0.9, label=f'Support: ${support_level:.2f}',
                  zorder=10)
        
        # วาดจุดสัมผัส Support
        for idx in support_touches[:5]:
            if 0 <= idx < len(lows):
                ax.scatter([idx], [support_level], color='#00ff88', 
                          s=120, marker='_', linewidths=3, zorder=15)
        
        # 🔲 วาดกรอบ Rectangle
        rect_start_x = start_idx
        rect_end_x = len(highs) - 1
        rect_width = rect_end_x - rect_start_x
        rect_height = range_size
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((rect_start_x, support_level), rect_width, rect_height,
                        linewidth=2, edgecolor='#ffaa00', facecolor='#ffaa00',
                        alpha=0.15, zorder=5)
        ax.add_patch(rect)
        
        # 📊 Label
        mid_x = (rect_start_x + rect_end_x) / 2
        mid_y = (resistance_level + support_level) / 2
        
        range_info = f"Range: {range_size:.2f}\n({range_pct:.1%})"
        
        ax.text(mid_x, mid_y, 
               f'🔲 RECTANGLE\n{range_info}', 
               ha='center', va='center',
               color='#ffaa00', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.6', 
                        facecolor='black', edgecolor='#ffaa00',
                        alpha=0.9, linewidth=2))
        
        # 🎯 Breakout Targets
        target_up = resistance_level + range_size
        target_down = support_level - range_size
        
        ax.axhline(y=target_up, color='#00ff00', linestyle=':', 
                  linewidth=2, alpha=0.6, label=f'Breakout Up: ${target_up:.2f}')
        
        ax.axhline(y=target_down, color='#ff0000', linestyle=':', 
                  linewidth=2, alpha=0.6, label=f'Breakdown: ${target_down:.2f}')
        
        print(f"✅ Rectangle drawn: R={resistance_level:.2f}, S={support_level:.2f}, Range={range_pct:.1%}")
        
    except Exception as e:
        print(f"❌ Draw Rectangle error: {e}")
        import traceback
        traceback.print_exc()

def draw_diamond_on_chart(ax, df):
    """วาด Diamond Pattern บนกราฟ - IMPROVED VERSION"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        if len(highs) < 30:
            print("⚠️ Not enough data for Diamond")
            return
        
        # ✅ Diamond pattern: volatility expansion then contraction
        # แบ่งเป็น 4 ส่วน (quarters)
        lookback = min(40, len(highs))
        start_idx = len(highs) - lookback
        quarter_len = lookback // 4
        
        # คำนวณ volatility ของแต่ละส่วน
        quarters_volatility = []
        for q in range(4):
            q_start = start_idx + (q * quarter_len)
            q_end = start_idx + ((q + 1) * quarter_len) if q < 3 else len(highs)
            
            if q_end > q_start:
                q_range = np.max(highs[q_start:q_end]) - np.min(lows[q_start:q_end])
                quarters_volatility.append({
                    'quarter': q + 1,
                    'start': q_start,
                    'end': q_end,
                    'volatility': q_range
                })
        
        if len(quarters_volatility) < 4:
            print("⚠️ Cannot divide into 4 quarters")
            return
        
        q1, q2, q3, q4 = quarters_volatility
        
        print(f"🔍 Diamond volatility: Q1={q1['volatility']:.2f}, Q2={q2['volatility']:.2f}, Q3={q3['volatility']:.2f}, Q4={q4['volatility']:.2f}")
        
        # ✅ ตรวจสอบ pattern: Q2 > Q1, Q3 >= Q2 (expansion), Q4 < Q3 (contraction)
        is_expanding = q2['volatility'] > q1['volatility'] * 0.95 and q3['volatility'] >= q2['volatility'] * 0.95
        is_contracting = q4['volatility'] < q3['volatility'] * 0.85
        
        if not (is_expanding and is_contracting):
            print(f"⚠️ Not a Diamond pattern (expanding={is_expanding}, contracting={is_contracting})")
            return
        
        # 📍 หาจุดสำคัญของ Diamond
        # จุดซ้าย (start point)
        left_idx = q1['start']
        left_price = (highs[left_idx] + lows[left_idx]) / 2
        
        # จุดกลาง (widest point - Q3)
        mid_high_idx = q3['start'] + np.argmax(highs[q3['start']:q3['end']])
        mid_low_idx = q3['start'] + np.argmin(lows[q3['start']:q3['end']])
        mid_high = highs[mid_high_idx]
        mid_low = lows[mid_low_idx]
        
        # จุดขวา (apex - current)
        right_idx = len(highs) - 1
        right_price = closes[-1]
        
        # 💎 วาดรูปเพชร (4 เส้น)
        # Upper left to top
        ax.plot([left_idx, mid_high_idx], 
               [left_price, mid_high],
               color='#ff00ff', linestyle='-', linewidth=3,
               alpha=0.9, zorder=10)
        
        # Top to upper right
        ax.plot([mid_high_idx, right_idx], 
               [mid_high, right_price],
               color='#ff00ff', linestyle='-', linewidth=3,
               alpha=0.9, zorder=10)
        
        # Lower left to bottom
        ax.plot([left_idx, mid_low_idx], 
               [left_price, mid_low],
               color='#ff00ff', linestyle='-', linewidth=3,
               alpha=0.9, zorder=10)
        
        # Bottom to lower right
        ax.plot([mid_low_idx, right_idx], 
               [mid_low, right_price],
               color='#ff00ff', linestyle='-', linewidth=3,
               alpha=0.9, zorder=10)
        
        # 💎 วาดจุดสำคัญ
        # จุดซ้าย
        ax.scatter([left_idx], [left_price], 
                  color='#ff00ff', s=200, marker='D',
                  edgecolors='white', linewidths=3, zorder=15)
        
        ax.text(left_idx, left_price, 
               'START', ha='right', va='center',
               color='#ff00ff', fontweight='bold', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor='black', 
                        edgecolor='#ff00ff',
                        alpha=0.9, linewidth=2))
        
        # จุดบน (widest top)
        ax.scatter([mid_high_idx], [mid_high], 
                  color='#ff00ff', s=280, marker='D',
                  edgecolors='white', linewidths=3, label='Diamond Top', zorder=15)
        
        ax.text(mid_high_idx, mid_high + 10, 
               '💎 TOP', ha='center', va='bottom',
               color='#ff00ff', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='#ff00ff', 
                        edgecolor='white',
                        alpha=0.95, linewidth=3))
        
        # จุดล่าง (widest bottom)
        ax.scatter([mid_low_idx], [mid_low], 
                  color='#ff00ff', s=280, marker='D',
                  edgecolors='white', linewidths=3, label='Diamond Bottom', zorder=15)
        
        ax.text(mid_low_idx, mid_low - 10, 
               '💎 BOTTOM', ha='center', va='top',
               color='#ff00ff', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='#ff00ff', 
                        edgecolor='white',
                        alpha=0.95, linewidth=3))
        
        # จุดขวา (apex - current)
        ax.scatter([right_idx], [right_price], 
                  color='#ffff00', s=250, marker='*',
                  edgecolors='white', linewidths=3, label='Apex (Breakout)', zorder=15)
        
        ax.text(right_idx, right_price + 10, 
               '⭐ APEX', ha='center', va='bottom',
               color='#ffff00', fontweight='bold', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor='black', 
                        edgecolor='#ffff00',
                        alpha=0.9, linewidth=2))
        
        # 📊 Main Label
        mid_x = (left_idx + right_idx) / 2
        mid_y = (mid_high + mid_low) / 2
        
        volatility_info = f"Peak Vol: {q3['volatility']:.2f}\nCurrent: {q4['volatility']:.2f}"
        convergence_pct = ((q4['volatility'] / q3['volatility']) - 1) * 100
        
        ax.text(mid_x, mid_y, 
               f'💎 DIAMOND\n{volatility_info}\nConverging: {convergence_pct:.1f}%', 
               ha='center', va='center',
               color='#ff00ff', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle='round,pad=0.7', 
                        facecolor='black', edgecolor='#ff00ff',
                        alpha=0.95, linewidth=2))
        
        # 🎯 Breakout Targets
        diamond_height = mid_high - mid_low
        
        target_up = right_price + diamond_height
        target_down = right_price - diamond_height
        
        ax.axhline(y=target_up, color='#00ff00', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Up Target: ${target_up:.2f}')
        
        ax.axhline(y=target_down, color='#ff0000', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Down Target: ${target_down:.2f}')
        
        print(f"✅ Diamond drawn: Expansion Q1→Q3 ({q1['volatility']:.2f}→{q3['volatility']:.2f}), Contraction Q4 ({q4['volatility']:.2f})")
        
    except Exception as e:
        print(f"❌ Draw Diamond error: {e}")
        import traceback
        traceback.print_exc()

def draw_harmonic_on_chart(ax, df, points, pattern_name):
    """วาด Harmonic Pattern (XABCD) บนกราฟ - FIXED"""
    try:
        colors = {
            'X': '#ff0000', 'A': '#00ff00', 'B': '#0000ff', 
            'C': '#ffff00', 'D': '#ff00ff'
        }
        
        point_order = ['X', 'A', 'B', 'C', 'D']
        valid_points = []
        
        # ✅ FIX: คำนวณ offset ถูกต้อง
        full_df_length = len(df)  # ใช้ความยาวจริงของ df ที่ส่งเข้ามา
        chart_df_length = min(50, len(df))  # จำนวน bars ที่แสดงในกราฟ
        offset = full_df_length - chart_df_length
        
        print(f"🔍 Harmonic Debug: full_length={full_df_length}, chart_length={chart_df_length}, offset={offset}")
        
        for point_name in point_order:
            if point_name in points and points[point_name]:
                point_data = points[point_name]
                original_idx, price, ptype = point_data
                
                # คำนวณ index สำหรับกราฟ
                chart_idx = original_idx - offset
                
                print(f"  Point {point_name}: original_idx={original_idx}, chart_idx={chart_idx}, price={price:.2f}")
                
                # ตรวจสอบว่าอยู่ในช่วงที่แสดง
                if 0 <= chart_idx < chart_df_length:
                    valid_points.append({
                        'name': point_name,
                        'idx': chart_idx,
                        'price': price,
                        'color': colors[point_name]
                    })
                else:
                    print(f"  ⚠️ Point {point_name} out of range!")
        
        if len(valid_points) < 4:
            print(f"⚠️ Not enough valid points for {pattern_name}: {len(valid_points)}/5")
            return
        
        print(f"✅ Drawing {len(valid_points)} points for {pattern_name}")
        
        # 🎯 วาดจุด XABCD
        for point in valid_points:
            ax.scatter([point['idx']], [point['price']], 
                      color=point['color'], s=250, marker='D',  # ✅ เพิ่มขนาด
                      label=f"Point {point['name']}", zorder=15,
                      edgecolors='white', linewidths=3)
            
            # 📝 เพิ่ม Label ที่ชัดเจน
            ax.text(point['idx'], point['price'] + 10, 
                   f"🎯 {point['name']}", 
                   ha='center', va='bottom', 
                   color=point['color'], 
                   fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle='round,pad=0.6', 
                            facecolor='black', 
                            edgecolor=point['color'],
                            alpha=0.95, linewidth=2))
        
        # 📏 วาดเส้นเชื่อมจุด
        for i in range(len(valid_points) - 1):
            p1 = valid_points[i]
            p2 = valid_points[i + 1]
            
            ax.plot([p1['idx'], p2['idx']], 
                   [p1['price'], p2['price']], 
                   color='#ffffff', linestyle='-', 
                   linewidth=3, alpha=0.8, zorder=10)
            
            # แสดงชื่อขา
            mid_x = (p1['idx'] + p2['idx']) / 2
            mid_y = (p1['price'] + p2['price']) / 2
            leg_name = f"{p1['name']}{p2['name']}"
            
            ax.text(mid_x, mid_y, leg_name, 
                   ha='center', va='center',
                   color='#ffaa00', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='black', alpha=0.9))
        
        # 🎯 คำนวณและแสดง Fibonacci Ratios
        if len(valid_points) >= 5:  # XABCD complete
            X, A, B, C, D = valid_points[0], valid_points[1], valid_points[2], valid_points[3], valid_points[4]
            
            # คำนวณ Fibonacci Ratios
            XA = abs(A['price'] - X['price'])
            AB = abs(B['price'] - A['price'])
            BC = abs(C['price'] - B['price'])
            CD = abs(D['price'] - C['price'])
            
            # Ratios
            AB_XA_ratio = (AB / XA) if XA != 0 else 0
            BC_AB_ratio = (BC / AB) if AB != 0 else 0
            CD_BC_ratio = (CD / BC) if BC != 0 else 0
            AD_XA_ratio = (abs(D['price'] - A['price']) / XA) if XA != 0 else 0
            
            # 📊 แสดง Ratios บนกราฟ
            ratio_text = f"📐 Fibonacci Ratios:\n"
            ratio_text += f"AB/XA: {AB_XA_ratio:.3f}\n"
            ratio_text += f"BC/AB: {BC_AB_ratio:.3f}\n"
            ratio_text += f"CD/BC: {CD_BC_ratio:.3f}\n"
            ratio_text += f"AD/XA: {AD_XA_ratio:.3f}"
            
            # วางข้อความที่มุมขวาบน
            ax.text(0.98, 0.98, ratio_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   color='#00ffff',
                   fontsize=10,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', 
                            facecolor='black', 
                            edgecolor='#00ffff',
                            alpha=0.95, 
                            linewidth=2))
            
            # 🎯 คำนวณ PRZ (Potential Reversal Zone)
            prz_range = CD * 0.1  # 10% ของขา CD
            prz_high = D['price'] + prz_range
            prz_low = D['price'] - prz_range
            
            # วาดเขต PRZ
            ax.axhspan(prz_low, prz_high, 
                      alpha=0.2, 
                      color='#ff00ff', 
                      zorder=5,
                      label='PRZ (Reversal Zone)')
            
            ax.text(D['idx'] + 2, D['price'], 
                   '🎯 PRZ', 
                   ha='left', va='center',
                   color='#ff00ff', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='black', 
                            edgecolor='#ff00ff',
                            alpha=0.95, linewidth=2))
            
            print(f"✅ Fibonacci Ratios calculated: AB/XA={AB_XA_ratio:.3f}, BC/AB={BC_AB_ratio:.3f}, CD/BC={CD_BC_ratio:.3f}")
        
        # 🏷️ แสดงชื่อ Pattern
        pattern_label_x = valid_points[len(valid_points)//2]['idx']
        pattern_label_y = sum([p['price'] for p in valid_points]) / len(valid_points)
        
        ax.text(pattern_label_x, pattern_label_y, 
               f'✨ {pattern_name} ✨', 
               ha='center', va='center',
               color='#ff00ff', 
               fontweight='bold', 
               fontsize=15,
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='black', 
                        edgecolor='#ff00ff',
                        alpha=0.95, 
                        linewidth=3))
        
        print(f"✅ Harmonic pattern {pattern_name} drawn successfully with {len(valid_points)} points")
        
    except Exception as e:
        print(f"❌ Draw Harmonic error: {e}")
        import traceback
        traceback.print_exc()

def draw_abcd_on_chart(ax, df, points):
    """วาด AB=CD Pattern บนกราฟ"""
    try:
        colors = {
            'A': '#ff0000',  # Red
            'B': '#00ff00',  # Green
            'C': '#0000ff',  # Blue
            'D': '#ff00ff'   # Magenta
        }
        
        point_order = ['A', 'B', 'C', 'D']
        valid_points = []
        
        for point_name in point_order:
            if point_name in points and points[point_name]:
                point_data = points[point_name]
                original_idx, price, ptype = point_data
                
                chart_idx = len(df) - 50 + original_idx if original_idx >= len(df) - 50 else original_idx
                
                if 0 <= chart_idx < 50:
                    valid_points.append({
                        'name': point_name,
                        'idx': chart_idx,
                        'price': price,
                        'color': colors[point_name]
                    })
        
        if len(valid_points) < 4:
            print(f"⚠️ Not enough valid points for AB=CD (found {len(valid_points)})")
            return
        
        # วาดจุด A, B, C, D
        for point in valid_points:
            ax.scatter([point['idx']], [point['price']], 
                      color=point['color'], s=280, marker='D',  # Diamond shape
                      label=f"Point {point['name']}", zorder=15,
                      edgecolors='white', linewidths=3)
            
            emoji = {'A': '🔴', 'B': '🟢', 'C': '🔵', 'D': '💜'}
            ax.text(point['idx'], point['price'] + 8, 
                   f"{emoji[point['name']]} {point['name']}", 
                   ha='center', va='bottom', 
                   color=point['color'], 
                   fontweight='bold', fontsize=14,
                   bbox=dict(boxstyle='round,pad=0.6', 
                            facecolor='black', 
                            edgecolor=point['color'],
                            alpha=0.9, linewidth=2))
        
        A, B, C, D = valid_points
        
        # 📏 วาดขา AB (เส้นเขียว)
        ax.plot([A['idx'], B['idx']], 
               [A['price'], B['price']], 
               color='#00ff88', linestyle='-', 
               linewidth=4, alpha=0.9, label='AB Leg', zorder=10)
        
        # 📏 วาดขา CD (เส้นม่วง)
        ax.plot([C['idx'], D['idx']], 
               [C['price'], D['price']], 
               color='#ff00ff', linestyle='-', 
               linewidth=4, alpha=0.9, label='CD Leg', zorder=10)
        
        # วาดขา BC (เส้นเชื่อม)
        ax.plot([B['idx'], C['idx']], 
               [B['price'], C['price']], 
               color='#888888', linestyle='--', 
               linewidth=2, alpha=0.6, zorder=9)
        
        # คำนวณและแสดง ratios
        AB = abs(B['price'] - A['price'])
        CD = abs(D['price'] - C['price'])
        ratio = CD / AB if AB > 0 else 0
        
        ratio_text = f"""
📐 AB=CD PATTERN:
AB Length = {AB:.2f}
CD Length = {CD:.2f}
CD/AB Ratio = {ratio:.3f}
"""
        
        # Expected ratios
        if 0.95 <= ratio <= 1.05:
            ratio_status = "✅ Perfect 1:1"
        elif 1.20 <= ratio <= 1.30:
            ratio_status = "✅ Good 1.272"
        elif 1.55 <= ratio <= 1.65:
            ratio_status = "✅ Good 1.618"
        else:
            ratio_status = "⚠️ Non-standard"
        
        ratio_text += f"\n{ratio_status}"
        
        ax.text(0.98, 0.98, ratio_text.strip(), 
               transform=ax.transAxes,
               horizontalalignment='right',
               verticalalignment='top',
               color='#ffff00', fontsize=10,
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='#1a1a1a', 
                        edgecolor='#ffff00',
                        alpha=0.95, linewidth=2))
        
        # 🎯 Entry Zone at D
        ax.axhspan(D['price'] * 0.998, D['price'] * 1.002, 
                  alpha=0.2, color='#ff00ff', 
                  label='Entry Zone', zorder=5)
        
        print(f"✅ AB=CD Pattern drawn (Ratio: {ratio:.3f})")
        
    except Exception as e:
        print(f"❌ Draw AB=CD error: {e}")
        import traceback
        traceback.print_exc()

def draw_elliott_wave_on_chart(ax, df, wave_points, pattern_type):
    """วาด Elliott Wave บนกราฟ - FIXED"""
    try:
        if pattern_type == 'ELLIOTT_WAVE_5':
            point_order = ['start', '1', '2', '3', '4', '5']
            colors = {
                'start': '#ffffff',
                '1': '#ff0000',
                '2': '#00ff00',
                '3': '#0000ff',
                '4': '#ffff00',
                '5': '#ff00ff'
            }
        else:  # ELLIOTT_WAVE_3
            point_order = ['start', 'A', 'B', 'C']
            colors = {
                'start': '#ffffff',
                'A': '#ff0000',
                'B': '#00ff00',
                'C': '#0000ff'
            }
        
        valid_points = []
        
        # ✅ คำนวณ offset
        full_df_length = len(df)
        chart_df_length = min(50, len(df))
        offset = full_df_length - chart_df_length
        
        print(f"🌊 Elliott Debug: pattern={pattern_type}, offset={offset}")
        
        for wave_name in point_order:
            if wave_name in wave_points and wave_points[wave_name]:
                wave_data = wave_points[wave_name]
                original_idx, price, wtype = wave_data
                
                chart_idx = original_idx - offset
                
                print(f"  Wave {wave_name}: original_idx={original_idx}, chart_idx={chart_idx}, price={price:.2f}")
                
                if 0 <= chart_idx < chart_df_length:
                    valid_points.append({
                        'name': wave_name,
                        'idx': chart_idx,
                        'price': price,
                        'color': colors[wave_name]
                    })
        
        if len(valid_points) < 3:
            print(f"⚠️ Not enough waves for {pattern_type}: {len(valid_points)}")
            return
        
        print(f"✅ Drawing {len(valid_points)} waves for {pattern_type}")
        
        # 🌊 วาดจุด Wave
        for point in valid_points:
            ax.scatter([point['idx']], [point['price']], 
                      color=point['color'], s=240, marker='o',
                      label=f"Wave {point['name']}", zorder=15,
                      edgecolors='white', linewidths=3)
            
            ax.text(point['idx'], point['price'] + 10, 
                   f"🌊 W{point['name']}", 
                   ha='center', va='bottom', 
                   color=point['color'], 
                   fontweight='bold', fontsize=13,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='black', 
                            edgecolor=point['color'],
                            alpha=0.95, linewidth=2))
        
        # 📏 วาดเส้นเชื่อม
        for i in range(len(valid_points) - 1):
            p1 = valid_points[i]
            p2 = valid_points[i + 1]
            
            # สลับสีเส้น
            line_color = '#00ffcc' if i % 2 == 0 else '#ff6666'
            
            ax.plot([p1['idx'], p2['idx']], 
                   [p1['price'], p2['price']], 
                   color=line_color, linestyle='-', 
                   linewidth=3, alpha=0.8, zorder=10)
        
        # 📊 แสดง Wave Analysis
        if pattern_type == 'ELLIOTT_WAVE_5' and len(valid_points) == 6:
            start, w1, w2, w3, w4, w5 = valid_points
            
            wave1 = abs(w1['price'] - start['price'])
            wave3 = abs(w3['price'] - w2['price'])
            wave5 = abs(w5['price'] - w4['price'])
            
            analysis = f"""🌊 ELLIOTT 5:
W1: {wave1:.2f}
W3: {wave3:.2f}
W5: {wave5:.2f}
W3/W1: {(wave3/wave1):.2f}x"""
            
            ax.text(0.02, 0.70, analysis.strip(), 
                   transform=ax.transAxes,
                   verticalalignment='top',
                   color='#00ffcc', fontsize=10,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.8', 
                            facecolor='#1a1a1a', 
                            edgecolor='#00ffcc',
                            alpha=0.95, linewidth=2))
        
        print(f"✅ {pattern_type} drawn successfully!")
        
    except Exception as e:
        print(f"❌ Draw Elliott Wave error: {e}")
        import traceback
        traceback.print_exc()


def draw_ascending_triangle_on_chart(ax, df):
    """วาด Ascending Triangle บนกราฟ - Bullish Pattern"""
    try:
        import numpy as np  # ✅ เพิ่ม import
        
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) < 20:
            print("⚠️ Not enough data for Ascending Triangle")
            return
        
        # 🔍 หาจุดสูงและต่ำด้วยวิธีที่ง่ายกว่า
        lookback = min(30, len(highs) - 5)  # ใช้ช่วงที่มีข้อมูล
        
        high_points = []
        low_points = []
        
        # หาจุด swing high/low ด้วย window ที่เล็กกว่า
        window = 3  # ลดจาก 5 เป็น 3 เพื่อหาจุดได้มากขึ้น
        
        for i in range(window, len(highs) - window):
            # Swing high
            if highs[i] == max(highs[i-window:i+window+1]):
                high_points.append((i, highs[i]))
            
            # Swing low
            if lows[i] == min(lows[i-window:i+window+1]):
                low_points.append((i, lows[i]))
        
        print(f"🔍 Found {len(high_points)} highs, {len(low_points)} lows")
        
        if len(high_points) < 2 or len(low_points) < 2:
            print(f"⚠️ Not enough swing points for Ascending Triangle")
            return
        
        # เอาจุดล่าสุดที่มีนัยสำคัญ
        recent_highs = high_points[-4:] if len(high_points) >= 4 else high_points[-2:]
        recent_lows = low_points[-4:] if len(low_points) >= 4 else low_points[-2:]
        
        print(f"📊 Using {len(recent_highs)} highs, {len(recent_lows)} lows for pattern")
        
        # 🔴 วาดเส้น Horizontal Resistance (เส้นราบด้านบน)
        if len(recent_highs) >= 2:
            # คำนวณระดับ resistance (ค่าเฉลี่ยของจุดสูง)
            resistance_prices = [h[1] for h in recent_highs]
            resistance_level = np.mean(resistance_prices)
            
            # ✅ ผ่อนปรนเงื่อนไข - ให้แปรปรวนได้มากขึ้น
            high_std = np.std(resistance_prices)
            high_range = max(resistance_prices) - min(resistance_prices)
            
            print(f"📈 Resistance level: {resistance_level:.2f}, range: {high_range:.2f} ({high_range/resistance_level*100:.2f}%)")
            
            if high_range > resistance_level * 0.03:  # เพิ่มจาก 0.02 เป็น 0.03 (3%)
                print(f"⚠️ Highs not horizontal enough (range={high_range:.2f}, {high_range/resistance_level*100:.1f}%)")
                # ไม่ return - ลองวาดต่อไป
            
            # หาช่วงเวลาของ resistance line
            first_high_idx = recent_highs[0][0]
            last_high_idx = recent_highs[-1][0]
            
            # วาดเส้น Resistance แนวราบ
            ax.axhline(y=resistance_level, 
                      xmin=first_high_idx/len(df), 
                      xmax=1.0,  # ✅ ขยายไปจนสุดกราฟ
                      color='#ff4444', linestyle='-', linewidth=3,
                      alpha=0.9, label=f'Resistance: ${resistance_level:.2f}', 
                      zorder=10)
            
            # วาดจุด Resistance touches
            for idx, price in recent_highs:
                ax.scatter([idx], [resistance_level], color='#ff4444', s=200,  # ✅ ใช้ resistance_level แทน price
                          marker='v', edgecolors='white', linewidths=3, zorder=15)
                
                ax.text(idx, resistance_level + 5, '🔴', 
                       ha='center', va='bottom', fontsize=16, zorder=16)
            
            print(f"✅ Drew Resistance line at {resistance_level:.2f}")
        else:
            print("⚠️ Not enough high points for Resistance")
            return
        
        # 🟢 วาดเส้น Ascending Support (เส้นขาขึ้นด้านล่าง) - KEY FIX HERE
        if len(recent_lows) >= 2:
            # ✅ เรียงจุดต่ำตามเวลา
            sorted_lows = sorted(recent_lows, key=lambda x: x[0])
            
            # เอา 2-3 จุดที่ไม่ซ้ำกัน
            unique_lows = []
            for idx, price in sorted_lows:
                if not unique_lows or idx - unique_lows[-1][0] > 3:  # ห่างกันอย่างน้อย 3 bars
                    unique_lows.append((idx, price))
            
            if len(unique_lows) < 2:
                print("⚠️ Not enough distinct low points")
                return
            
            l1, l2 = unique_lows[0], unique_lows[-1]  # เอาจุดแรกและจุดสุดท้าย
            
            print(f"📊 Support points: L1=({l1[0]}, {l1[1]:.2f}), L2=({l2[0]}, {l2[1]:.2f})")
            
            # ตรวจสอบว่าเป็นแนวขาขึ้นหรือไม่
            if l2[1] <= l1[1]:
                print(f"⚠️ Support is not ascending: L1={l1[1]:.2f}, L2={l2[1]:.2f}")
                # ลองใช้จุดอื่นแทน
                if len(unique_lows) >= 3:
                    l1 = unique_lows[-2]
                    l2 = unique_lows[-1]
                    print(f"🔄 Trying alternate points: L1=({l1[0]}, {l1[1]:.2f}), L2=({l2[0]}, {l2[1]:.2f})")
                    if l2[1] <= l1[1]:
                        print("⚠️ Still not ascending, using best fit line")
                        # ใช้ linear regression สำหรับ best fit
                        from scipy import stats
                        low_indices = [p[0] for p in unique_lows]
                        low_prices = [p[1] for p in unique_lows]
                        slope, intercept, _, _, _ = stats.linregress(low_indices, low_prices)
                        
                        if slope <= 0:
                            print("❌ Cannot create ascending support line")
                            return
                        
                        # สร้างจุด l1, l2 จาก regression line
                        l1 = (unique_lows[0][0], intercept + slope * unique_lows[0][0])
                        l2 = (unique_lows[-1][0], intercept + slope * unique_lows[-1][0])
                        print(f"✅ Using regression: slope={slope:.4f}")
            
            # ✅ วาดเส้นแนวโน้มขาขึ้น (MAIN LINE)
            ax.plot([l1[0], l2[0]], [l1[1], l2[1]], 
                   color='#00ff88', linestyle='-', linewidth=4,  # ✅ เพิ่มความหนา
                   alpha=1.0, label='Ascending Support', zorder=12)
            
            print(f"✅ Drew Ascending Support from ({l1[0]}, {l1[1]:.2f}) to ({l2[0]}, {l2[1]:.2f})")
            
            # ✅ Extend เส้นไปข้างหน้า
            slope = (l2[1] - l1[1]) / (l2[0] - l1[0])
            extended_x = len(df) - 1
            extended_y = l2[1] + slope * (extended_x - l2[0])
            
            ax.plot([l2[0], extended_x], [l2[1], extended_y], 
                   color='#00ff88', linestyle='--', linewidth=2.5, 
                   alpha=0.8, zorder=12)
            
            print(f"✅ Extended support to x={extended_x}, y={extended_y:.2f}")
            
            # วาดจุด Support touches
            for idx, price in unique_lows:
                # คำนวณตำแหน่งบนเส้น support
                support_price_at_idx = l1[1] + slope * (idx - l1[0])
                
                ax.scatter([idx], [support_price_at_idx], color='#00ff88', s=200, 
                          marker='^', edgecolors='white', linewidths=3, zorder=15)
                
                ax.text(idx, support_price_at_idx - 5, '🟢', 
                       ha='center', va='top', fontsize=16, zorder=16)
        else:
            print("⚠️ Not enough low points for Support")
            return
        
        # 🎯 คำนวณ Apex และ Breakout Target
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            resistance_level = np.mean([h[1] for h in recent_highs])
            
            l1, l2 = recent_lows[0], recent_lows[-1]
            support_slope = (l2[1] - l1[1]) / (l2[0] - l1[0])
            
            # หาจุด Apex (จุดที่ support จะตัดกับ resistance)
            # resistance_level = l1[1] + slope * (apex_x - l1[0])
            apex_x = l1[0] + (resistance_level - l1[1]) / support_slope
            
            if apex_x > len(df) - 1:  # Apex อยู่ในอนาคต
                # วาดเส้นประไปยัง apex
                ax.plot([l2[0], apex_x], 
                       [l2[1], resistance_level],
                       color='#ffaa00', linestyle=':', linewidth=2, alpha=0.6)
                
                ax.scatter([apex_x], [resistance_level], 
                          color='#ffff00', s=250, marker='*', 
                          edgecolors='white', linewidths=2, 
                          label='Apex (Breakout Point)', zorder=15)
                
                ax.text(apex_x, resistance_level + 10, 
                       '⭐ APEX', ha='center', va='bottom',
                       color='#ffff00', fontweight='bold', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='black', alpha=0.9))
            
            # 🎯 Target (ความสูงของรูปสามเหลี่ยม + resistance)
            triangle_height = resistance_level - l1[1]
            target_price = resistance_level + triangle_height
            
            ax.axhline(y=target_price, color='#00ff00', linestyle=':', 
                      linewidth=3, alpha=0.8, 
                      label=f'Breakout Target: ${target_price:.2f}')
            
            ax.text(len(df) - 2, target_price, 
                   f'🎯 Target\n${target_price:.2f}', 
                   ha='right', va='center',
                   color='#00ff00', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='black', edgecolor='#00ff00',
                            alpha=0.9, linewidth=2))
        
        # 📊 Main Label
        mid_x = len(df) - 15
        mid_y = (resistance_level + recent_lows[-1][1]) / 2
        
        ax.text(mid_x, mid_y, 
               '📐 ASCENDING\nTRIANGLE\n(Bullish)', 
               ha='center', va='center',
               color='#00ff88', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.7', 
                        facecolor='black', edgecolor='#00ff88',
                        alpha=0.9, linewidth=2))
        
        print(f"✅ Ascending Triangle drawn: Resistance={resistance_level:.2f}, Support slope={support_slope:.4f}")
        
    except Exception as e:
        print(f"❌ Draw Ascending Triangle error: {e}")
        import traceback
        traceback.print_exc()


def draw_descending_triangle_on_chart(ax, df):
    """วาด Descending Triangle บนกราฟ - Bearish Pattern"""
    try:
        import numpy as np  # ✅ เพิ่ม import
        
        highs = df['high'].values
        lows = df['low'].values
        
        if len(highs) < 20:
            print("⚠️ Not enough data for Descending Triangle")
            return
        
        # หาจุดสูงและต่ำ
        high_points = []
        low_points = []
        
        window = 3
        for i in range(window, len(highs) - window):
            # Swing high
            if highs[i] == max(highs[i-window:i+window+1]):
                high_points.append((i, highs[i]))
            
            # Swing low
            if lows[i] == min(lows[i-window:i+window+1]):
                low_points.append((i, lows[i]))
        
        print(f"🔍 Found {len(high_points)} highs, {len(low_points)} lows")
        
        if len(high_points) < 2 or len(low_points) < 2:
            print(f"⚠️ Not enough swing points for Descending Triangle")
            return
        
        recent_highs = high_points[-4:] if len(high_points) >= 4 else high_points[-2:]
        recent_lows = low_points[-4:] if len(low_points) >= 4 else low_points[-2:]
        
        print(f"📊 Using {len(recent_highs)} highs, {len(recent_lows)} lows for pattern")
        
        # 🟢 วาดเส้น Horizontal Support (เส้นราบด้านล่าง)
        if len(recent_lows) >= 2:
            support_prices = [l[1] for l in recent_lows]
            support_level = np.mean(support_prices)
            
            low_range = max(support_prices) - min(support_prices)
            print(f"📈 Support level: {support_level:.2f}, range: {low_range:.2f} ({low_range/support_level*100:.2f}%)")
            
            if low_range > support_level * 0.03:
                print(f"⚠️ Lows not horizontal enough")
            
            first_low_idx = recent_lows[0][0]
            
            # วาดเส้น Support
            ax.axhline(y=support_level, 
                      xmin=first_low_idx/len(df), 
                      xmax=1.0,
                      color='#00ff88', linestyle='-', linewidth=3,
                      alpha=0.9, label=f'Support: ${support_level:.2f}', 
                      zorder=10)
            
            # วาดจุด Support touches
            for idx, price in recent_lows:
                ax.scatter([idx], [support_level], color='#00ff88', s=200, 
                          marker='^', edgecolors='white', linewidths=3, zorder=15)
                
                ax.text(idx, support_level - 5, '🟢', 
                       ha='center', va='top', fontsize=16, zorder=16)
            
            print(f"✅ Drew Support line at {support_level:.2f}")
        else:
            print("⚠️ Not enough low points for Support")
            return
        
        # 🔴 วาดเส้น Descending Resistance (เส้นขาลงด้านบน)
        if len(recent_highs) >= 2:
            sorted_highs = sorted(recent_highs, key=lambda x: x[0])
            
            unique_highs = []
            for idx, price in sorted_highs:
                if not unique_highs or idx - unique_highs[-1][0] > 3:
                    unique_highs.append((idx, price))
            
            if len(unique_highs) < 2:
                print("⚠️ Not enough distinct high points")
                return
            
            h1, h2 = unique_highs[0], unique_highs[-1]
            
            print(f"📊 Resistance points: H1=({h1[0]}, {h1[1]:.2f}), H2=({h2[0]}, {h2[1]:.2f})")
            
            # ตรวจสอบว่าเป็นแนวขาลง
            if h2[1] >= h1[1]:
                print(f"⚠️ Resistance is not descending")
                if len(unique_highs) >= 3:
                    h1 = unique_highs[-2]
                    h2 = unique_highs[-1]
                    if h2[1] >= h1[1]:
                        from scipy import stats
                        high_indices = [p[0] for p in unique_highs]
                        high_prices = [p[1] for p in unique_highs]
                        slope, intercept, _, _, _ = stats.linregress(high_indices, high_prices)
                        
                        if slope >= 0:
                            print("❌ Cannot create descending resistance line")
                            return
                        
                        h1 = (unique_highs[0][0], intercept + slope * unique_highs[0][0])
                        h2 = (unique_highs[-1][0], intercept + slope * unique_highs[-1][0])
            
            # วาดเส้นแนวโน้มขาลง
            ax.plot([h1[0], h2[0]], [h1[1], h2[1]], 
                   color='#ff4444', linestyle='-', linewidth=4,
                   alpha=1.0, label='Descending Resistance', zorder=12)
            
            print(f"✅ Drew Descending Resistance from ({h1[0]}, {h1[1]:.2f}) to ({h2[0]}, {h2[1]:.2f})")
            
            # Extend เส้นไปข้างหน้า
            slope = (h2[1] - h1[1]) / (h2[0] - h1[0])
            extended_x = len(df) - 1
            extended_y = h2[1] + slope * (extended_x - h2[0])
            
            ax.plot([h2[0], extended_x], [h2[1], extended_y], 
                   color='#ff4444', linestyle='--', linewidth=2.5, 
                   alpha=0.8, zorder=12)
            
            # วาดจุด Resistance touches
            for idx, price in unique_highs:
                resistance_price_at_idx = h1[1] + slope * (idx - h1[0])
                
                ax.scatter([idx], [resistance_price_at_idx], color='#ff4444', s=200, 
                          marker='v', edgecolors='white', linewidths=3, zorder=15)
                
                ax.text(idx, resistance_price_at_idx + 5, '🔴', 
                       ha='center', va='bottom', fontsize=16, zorder=16)
        else:
            print("⚠️ Not enough high points for Resistance")
            return
        
        # 🎯 คำนวณ Apex และ Breakdown Target
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            support_level = np.mean([l[1] for l in recent_lows])
            
            h1, h2 = recent_highs[0], recent_highs[-1]
            resistance_slope = (h2[1] - h1[1]) / (h2[0] - h1[0])
            
            # หาจุด Apex
            apex_x = h1[0] + (support_level - h1[1]) / resistance_slope
            
            if apex_x > len(df) - 1:
                ax.plot([h2[0], apex_x], 
                       [h2[1], support_level],
                       color='#ffaa00', linestyle=':', linewidth=2, alpha=0.6)
                
                ax.scatter([apex_x], [support_level], 
                          color='#ffff00', s=250, marker='*', 
                          edgecolors='white', linewidths=2, 
                          label='Apex (Breakdown Point)', zorder=15)
                
                ax.text(apex_x, support_level - 10, 
                       '⭐ APEX', ha='center', va='top',
                       color='#ffff00', fontweight='bold', fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='black', alpha=0.9))
            
            # 🎯 Target (ความสูงของรูปสามเหลี่ยม - support)
            triangle_height = h1[1] - support_level
            target_price = support_level - triangle_height
            
            ax.axhline(y=target_price, color='#ff0000', linestyle=':', 
                      linewidth=3, alpha=0.8, 
                      label=f'Breakdown Target: ${target_price:.2f}')
            
            ax.text(len(df) - 2, target_price, 
                   f'🎯 Target\n${target_price:.2f}', 
                   ha='right', va='center',
                   color='#ff0000', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='black', edgecolor='#ff0000',
                            alpha=0.9, linewidth=2))
        
        # 📊 Main Label
        mid_x = len(df) - 15
        mid_y = (recent_highs[-1][1] + support_level) / 2
        
        ax.text(mid_x, mid_y, 
               '📐 DESCENDING\nTRIANGLE\n(Bearish)', 
               ha='center', va='center',
               color='#ff4444', fontweight='bold', fontsize=13,
               bbox=dict(boxstyle='round,pad=0.7', 
                        facecolor='black', edgecolor='#ff4444',
                        alpha=0.9, linewidth=2))
        
        print(f"✅ Descending Triangle drawn: Support={support_level:.2f}, Resistance slope={resistance_slope:.4f}")
        
    except Exception as e:
        print(f"❌ Draw Descending Triangle error: {e}")
        import traceback
        traceback.print_exc()                  

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


def create_mock_harmonic_pattern(df, pattern_type):
    """สร้าง mock Harmonic pattern - FIXED VERSION"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        # ✅ ใช้ indices จริงจาก df
        total_len = len(df)
        
        print(f"🔧 Creating mock {pattern_type}: total_len={total_len}")
        
        # สร้างจุด XABCD จาก swing points จริง
        swing_highs = []
        swing_lows = []
        
        for i in range(5, total_len - 5):
            if all(highs[i] >= highs[i-j] for j in range(1, 6)) and \
               all(highs[i] >= highs[i+j] for j in range(1, 6)):
                swing_highs.append((i, highs[i]))
            
            if all(lows[i] <= lows[i-j] for j in range(1, 6)) and \
               all(lows[i] <= lows[i+j] for j in range(1, 6)):
                swing_lows.append((i, lows[i]))
        
        # เลือก 5 จุดสำหรับ XABCD
        all_swings = swing_highs + swing_lows
        all_swings.sort(key=lambda x: x[0])
        
        if len(all_swings) >= 5:
            # ✅ เลือก swing points ที่กระจายดี
            step = len(all_swings) // 5
            selected = [
                all_swings[step * 0],
                all_swings[step * 1],
                all_swings[step * 2],
                all_swings[step * 3],
                all_swings[step * 4]
            ]
            
            points = {
                'X': (selected[0][0], selected[0][1], 'swing'),
                'A': (selected[1][0], selected[1][1], 'swing'),
                'B': (selected[2][0], selected[2][1], 'swing'),
                'C': (selected[3][0], selected[3][1], 'swing'),
                'D': (selected[4][0], selected[4][1], 'swing')
            }
            
            print(f"✅ Mock points created:")
            for name, (idx, price, _) in points.items():
                print(f"  {name}: idx={idx}, price={price:.2f}")
        else:
            # Fallback
            print(f"⚠️ Not enough swings, using fallback")
            points = {
                'X': (total_len - 40, float(df['low'].iloc[-40]), 'low'),
                'A': (total_len - 30, float(df['high'].iloc[-30]), 'high'),
                'B': (total_len - 20, float(df['low'].iloc[-20]), 'low'),
                'C': (total_len - 10, float(df['high'].iloc[-10]), 'high'),
                'D': (total_len - 1, float(df['close'].iloc[-1]), 'close')
            }
        
        return {
            'pattern_id': 35,
            'pattern_name': pattern_type,
            'confidence': 0.75,
            'method': 'MOCK_HARMONIC',
            'points': points
        }
        
    except Exception as e:
        print(f"❌ Mock harmonic error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pattern_id': 0,
            'pattern_name': 'NO_PATTERN',
            'confidence': 0,
            'method': 'ERROR'
        }


def create_mock_abcd_pattern(df):
    """สร้าง mock AB=CD pattern"""
    try:
        last_idx = len(df) - 1
        
        points = {
            'A': (last_idx - 30, float(df['high'].iloc[-30]), 'high'),
            'B': (last_idx - 20, float(df['low'].iloc[-20]), 'low'),
            'C': (last_idx - 10, float(df['high'].iloc[-10]), 'high'),
            'D': (last_idx, float(df['close'].iloc[-1]), 'close')
        }
        
        return {
            'pattern_id': 39,
            'pattern_name': 'AB_CD',
            'confidence': 0.70,
            'method': 'MOCK_ABCD',
            'points': points
        }
        
    except Exception as e:
        print(f"Mock AB=CD error: {e}")
        return {
            'pattern_id': 0,
            'pattern_name': 'NO_PATTERN',
            'confidence': 0,
            'method': 'ERROR'
        }


def create_mock_elliott_wave(df, wave_type):
    """สร้าง mock Elliott Wave pattern"""
    try:
        last_idx = len(df) - 1
        
        if wave_type == '5':
            wave_points = {
                'start': (last_idx - 40, float(df['close'].iloc[-40]), 'start'),
                '1': (last_idx - 32, float(df['high'].iloc[-32]), 'high'),
                '2': (last_idx - 24, float(df['low'].iloc[-24]), 'low'),
                '3': (last_idx - 16, float(df['high'].iloc[-16]), 'high'),
                '4': (last_idx - 8, float(df['low'].iloc[-8]), 'low'),
                '5': (last_idx, float(df['close'].iloc[-1]), 'close')
            }
        else:  # wave_type == '3'
            wave_points = {
                'start': (last_idx - 30, float(df['close'].iloc[-30]), 'start'),
                'A': (last_idx - 20, float(df['low'].iloc[-20]), 'low'),
                'B': (last_idx - 10, float(df['high'].iloc[-10]), 'high'),
                'C': (last_idx, float(df['close'].iloc[-1]), 'close')
            }
        
        return {
            'pattern_id': 40 if wave_type == '5' else 41,
            'pattern_name': f'ELLIOTT_WAVE_{wave_type}',
            'confidence': 0.70,
            'method': 'MOCK_ELLIOTT',
            'wave_points': wave_points
        }
        
    except Exception as e:
        print(f"Mock Elliott Wave error: {e}")
        return {
            'pattern_id': 0,
            'pattern_name': 'NO_PATTERN',
            'confidence': 0,
            'method': 'ERROR'
        }

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
    """ส่งข้อความรูปแบบใหม่แยกตาม method - Enhanced Version with Top 5 Charts"""
    try:
        current_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        current_data = shared_df.iloc[-1]
        
        # สร้าง detector instance
        detector = AdvancedPatternDetector()
        trading_signals = detector.predict_signals(shared_df)
        
        # 🔥 Priority Patterns (ไม่สนใจ confidence)
        PRIORITY_PATTERNS = [
            'GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD',
            'ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3'
        ]
        
        # แยก priority patterns ออกมา
        priority_found = [p for p in all_patterns if p['pattern_name'] in PRIORITY_PATTERNS]
        
        # กรอง patterns ปกติที่มีคุณภาพ
        quality_patterns = [
            p for p in all_patterns 
            if p['pattern_name'] != 'NO_PATTERN' 
            and p['pattern_name'] not in PRIORITY_PATTERNS
            and p['confidence'] > 0.60
        ]
        
        # 🎯 รวม priority patterns + quality patterns
        combined_patterns = priority_found + quality_patterns
        
        # เรียงตาม confidence (priority patterns จะอยู่ข้างหน้าอยู่แล้ว)
        combined_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        # เอาแค่ top 5
        top_5_patterns = combined_patterns[:5]
        
        if not top_5_patterns:
            no_pattern_msg = f"""📊 XAU/USD Pattern Analysis
⏰ {current_time}

❌ No quality patterns detected
Current Price: ${current_data['close']:,.2f}

Waiting for clear pattern formation..."""
            send_telegram(no_pattern_msg)
            return 200
        
        # จำนวน priority patterns ที่พบ
        priority_count = len(priority_found)
        
        # ========================================
        # 1) ข้อความที่ 1: สรุป Top 5 Patterns
        # ========================================
        
        # จำแนกประเภท patterns สำหรับแสดง
        reversal_patterns = []
        continuation_patterns = []
        bearish_patterns = []
        bullish_patterns = []
        neutral_patterns = []
        harmonic_patterns = []
        elliott_patterns = []
        
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
        
        for pattern in top_5_patterns:
            pattern_name = pattern['pattern_name']
            
            # จำแนกตามประเภท
            if pattern_name in PRIORITY_PATTERNS:
                if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD']:
                    harmonic_patterns.append(pattern)
                else:
                    elliott_patterns.append(pattern)
            
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
        
        message_1 = f"""🔍 TOP 5 PATTERNS DETECTED - MULTI-CHART ANALYSIS

⏰ {current_time} | 💰 XAU/USD (1H)
💾 SHARED DATA SOURCE

📊 TOP 5 PATTERNS SUMMARY:
รวมพบ {len(top_5_patterns)} high-quality patterns
🌟 Priority Patterns (Harmonic/Elliott): {priority_count}

💰 CURRENT MARKET DATA:
Open: ${current_data['open']:,.2f} | High: ${current_data['high']:,.2f}
Low: ${current_data['low']:,.2f} | Close: ${current_data['close']:,.2f}
Current Price: ${current_data['close']:,.2f}

"""
        
        # แสดง Harmonic Patterns (ถ้ามี)
        if harmonic_patterns:
            message_1 += f"🦋 HARMONIC PATTERNS ({len(harmonic_patterns)}):\n"
            for i, pattern in enumerate(harmonic_patterns, 1):
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐"
                message_1 += f"{i}. {confidence_emoji} {pattern['pattern_name']}\n"
                message_1 += f"   Confidence: {pattern['confidence']*100:.1f}% | Method: {pattern.get('method', 'HARMONIC')}\n"
            message_1 += "\n"
        
        # แสดง Elliott Wave (ถ้ามี)
        if elliott_patterns:
            message_1 += f"🌊 ELLIOTT WAVE PATTERNS ({len(elliott_patterns)}):\n"
            for i, pattern in enumerate(elliott_patterns, 1):
                confidence_emoji = "🔥" if pattern['confidence'] > 0.8 else "⭐"
                message_1 += f"{i}. {confidence_emoji} {pattern['pattern_name']}\n"
                message_1 += f"   Confidence: {pattern['confidence']*100:.1f}% | Method: {pattern.get('method', 'ELLIOTT')}\n"
            message_1 += "\n"
        
        # แสดง Top 5 Rankings
        message_1 += f"🏆 TOP 5 RANKINGS:\n"
        for i, pattern in enumerate(top_5_patterns, 1):
            signal = "🔴 SELL" if pattern['pattern_name'] in bearish_list else "🟢 BUY"
            priority_badge = "⭐ PRIORITY" if pattern['pattern_name'] in PRIORITY_PATTERNS else ""
            message_1 += f"{i}. {pattern['pattern_name']} {priority_badge}\n"
            message_1 += f"   {signal} | Confidence: {pattern['confidence']*100:.1f}%\n"
        
        message_1 += f"\n📈 5 individual charts will be sent below..."
        
        send_telegram(message_1)
        time.sleep(3)
        
        # ========================================
        # 2) สร้างและส่งกราฟแยกทั้ง 5 patterns
        # ========================================
        
        for idx, pattern in enumerate(top_5_patterns, 1):
            try:
                pattern_name = pattern['pattern_name']
                confidence = pattern['confidence']
                
                # ⭐ สำคัญ: ถ้าเป็น Harmonic/Elliott ต้องมี points/wave_points
                if pattern_name in PRIORITY_PATTERNS:
                    # ตรวจสอบว่ามี points หรือ wave_points
                    if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD']:
                        if 'points' not in pattern or not pattern['points']:
                            print(f"⚠️ {pattern_name} missing points, creating mock data")
                            # สร้าง mock points
                            pattern = create_mock_harmonic_pattern(shared_df, pattern_name)
                    
                    elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
                        if 'wave_points' not in pattern or not pattern['wave_points']:
                            print(f"⚠️ {pattern_name} missing wave_points, creating mock data")
                            # สร้าง mock wave points
                            wave_type = '5' if pattern_name == 'ELLIOTT_WAVE_5' else '3'
                            pattern = create_mock_elliott_wave(shared_df, wave_type)
                
                # สร้าง trading signals สำหรับ pattern นี้
                current_price = float(shared_df['close'].iloc[-1])
                
                # กำหนด action ตาม pattern type
                if pattern_name in bearish_list:
                    action = 'SELL'
                    entry_price = current_price - (current_price * 0.0005)
                    tp1, tp2, tp3 = current_price * 0.997, current_price * 0.994, current_price * 0.991
                    sl = current_price * 1.005
                elif pattern_name in bullish_list:
                    action = 'BUY'
                    entry_price = current_price + (current_price * 0.0005)
                    tp1, tp2, tp3 = current_price * 1.003, current_price * 1.006, current_price * 1.009
                    sl = current_price * 0.995
                else:
                    action = 'WAIT'
                    entry_price = current_price
                    tp1, tp2, tp3, sl = current_price, current_price, current_price, current_price
                
                pattern_signals = {
                    'current_price': current_price,
                    'entry_price': round(entry_price, 2),
                    'tp1': round(tp1, 2),
                    'tp2': round(tp2, 2), 
                    'tp3': round(tp3, 2),
                    'sl': round(sl, 2),
                    'action': action,
                    'confidence': confidence,
                    'rsi': float(shared_df['rsi'].iloc[-1]) if not pd.isna(shared_df['rsi'].iloc[-1]) else 50.0,
                    'ema10': float(shared_df['ema'].iloc[-1]) if not pd.isna(shared_df['ema'].iloc[-1]) else current_price,
                    'ema21': float(shared_df['ema_21'].iloc[-1]) if not pd.isna(shared_df['ema_21'].iloc[-1]) else current_price
                }
                
                # 🎨 สร้างกราฟสำหรับ pattern นี้
                print(f"📊 Creating chart {idx}/5: {pattern_name}")
                chart_buffer = create_candlestick_chart(shared_df, pattern_signals, pattern)
                
                if chart_buffer:
                    # สร้างข้อความสำหรับกราฟนี้
                    priority_badge = "⭐ PRIORITY PATTERN ⭐" if pattern_name in PRIORITY_PATTERNS else ""
                    
                    chart_message = f"""📊 CHART #{idx}/5: {pattern_name}
{priority_badge}

💰 XAU/USD (1H) | ⏰ {current_time}

🔍 Pattern Details:
• Confidence: {confidence*100:.1f}%
• Method: {pattern.get('method', 'PATTERN_ANALYSIS')}
• Signal: {action}

💹 Technical Data:
• Current: ${current_price:,.2f}
• RSI: {pattern_signals['rsi']:.1f}
• EMA10: ${pattern_signals['ema10']:,.2f}
• EMA21: ${pattern_signals['ema21']:,.2f}

"""
                    
                    if action != 'WAIT':
                        chart_message += f"""💼 Trading Setup:
🎯 Entry: ${pattern_signals['entry_price']:,.2f}
🟢 TP1: ${pattern_signals['tp1']:,.2f}
🟢 TP2: ${pattern_signals['tp2']:,.2f}
🟢 TP3: ${pattern_signals['tp3']:,.2f}
🔴 SL: ${pattern_signals['sl']:,.2f}

"""
                    
                    # แสดงข้อมูล points (ถ้ามี)
                    if 'points' in pattern and pattern['points']:
                        chart_message += "🎯 Key Points (XABCD):\n"
                        for point_name, point_data in pattern['points'].items():
                            if point_data:
                                _, price, _ = point_data
                                chart_message += f"• {point_name}: ${price:.2f}\n"
                    
                    elif 'wave_points' in pattern and pattern['wave_points']:
                        chart_message += "🌊 Wave Structure:\n"
                        for wave_name, wave_data in pattern['wave_points'].items():
                            if wave_data:
                                _, price, _ = wave_data
                                chart_message += f"• Wave {wave_name}: ${price:.2f}\n"
                    
                    chart_message += f"\n⚠️ Risk: 1-2% per trade | Use Stop Loss!"
                    
                    # ส่งกราฟ
                    send_status = send_telegram_with_chart(chart_message, chart_buffer)
                    print(f"✅ Chart {idx}/5 sent: {pattern_name} (Status: {send_status})")

                    # 🆕 ส่งรูปทฤษฎีและคำอธิบาย
                    time.sleep(2)
                    pattern_description = get_pattern_description(pattern_name, pattern)
                    theory_status = send_pattern_theory_explanation(pattern_name, pattern_description)
                    print(f"📚 Theory diagram sent for {pattern_name} (Status: {theory_status})")
                    
                    # หน่วงเวลาระหว่างการส่งกราฟ
                    time.sleep(4)
                else:
                    print(f"❌ Chart creation failed for {pattern_name}")
            
            except Exception as e:
                print(f"❌ Error creating chart {idx}/5 for {pattern.get('pattern_name')}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # ========================================
        # 3) ข้อความสรุปท้าย - Priority Alert
        # ========================================
        if priority_count > 0:
            priority_alert = f"""🌟 PRIORITY PATTERN ALERT 🌟

⚠️ {priority_count} Harmonic/Elliott Wave pattern{'s' if priority_count > 1 else ''} detected!

These patterns are:
• Based on Fibonacci ratios (Harmonic)
• Based on wave structure (Elliott)
• High probability reversal/continuation signals
• Detected with complete point marking on charts

🎯 Priority Patterns Detected:
"""
            
            for i, pattern in enumerate([p for p in top_5_patterns if p['pattern_name'] in PRIORITY_PATTERNS], 1):
                priority_alert += f"{i}. {pattern['pattern_name']} - {pattern['confidence']*100:.1f}%\n"
                
                # แสดงข้อมูลเพิ่มเติมสำหรับ Harmonic
                if pattern['pattern_name'] in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
                    priority_alert += f"   📐 Fibonacci structure: XABCD\n"
                    if 'points' in pattern and pattern['points']:
                        d_point = pattern['points'].get('D')
                        if d_point:
                            priority_alert += f"   🎯 Entry zone at D: ${d_point[1]:.2f}\n"
                
                # แสดงข้อมูลเพิ่มเติมสำหรับ Elliott
                elif pattern['pattern_name'] in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
                    wave_type = "5-Wave Impulse" if pattern['pattern_name'] == 'ELLIOTT_WAVE_5' else "3-Wave Corrective"
                    priority_alert += f"   🌊 Wave structure: {wave_type}\n"
            
            priority_alert += f"""
💡 Action Required:
✅ Review all {priority_count} priority pattern charts above
✅ All XABCD points are marked on charts
✅ Wave structures are clearly visualized
✅ Entry/TP/SL levels are provided
✅ Wait for price action confirmation

⚠️ These patterns use advanced mathematical structures (Fibonacci, wave theory) with complete visual marking for accuracy."""
            
            send_telegram(priority_alert)
            time.sleep(2)
        
        # ========================================
        # 4) ข้อความสรุปท้าย
        # ========================================
        
        # คำนวณ bias
        bullish_count = len([p for p in top_5_patterns if p['pattern_name'] in bullish_list])
        bearish_count = len([p for p in top_5_patterns if p['pattern_name'] in bearish_list])
        
        if bullish_count > bearish_count:
            dominant_bias = "🟢 BULLISH BIAS"
            market_sentiment = "ตลาดมีแนวโน้มขาขึ้น"
        elif bearish_count > bullish_count:
            dominant_bias = "🔴 BEARISH BIAS"
            market_sentiment = "ตลาดมีแนวโน้มขาลง"
        else:
            dominant_bias = "🟡 NEUTRAL BIAS"
            market_sentiment = "ตลาดไม่มีทิศทางชัดเจน"
        
        highest_confidence = max([p['confidence'] for p in top_5_patterns])
        
        summary_message = f"""📚 TOP 5 PATTERNS ANALYSIS COMPLETE

✅ Sent 5 individual charts with marked points

📊 MARKET ANALYSIS SUMMARY:
• Total Patterns: {len(top_5_patterns)}
• Harmonic/Elliott: {priority_count} patterns (with XABCD/Wave points marked)
• Bullish Signals: {bullish_count}
• Bearish Signals: {bearish_count}
• Highest Confidence: {highest_confidence*100:.1f}%

🎯 MARKET BIAS: {dominant_bias}
💬 Sentiment: {market_sentiment}

🔝 Top Pattern:
{top_5_patterns[0]['pattern_name']} ({top_5_patterns[0]['confidence']*100:.1f}%)

💡 TRADING RECOMMENDATIONS:
• Priority: Harmonic & Elliott patterns have complete point marking
• Visual Confirmation: Check all marked points on charts
• Multiple Alignment: Look for pattern confluence
• Risk Management: Never risk more than 2% per trade
• Stop Loss: Always use protective stops

⚠️ DISCLAIMER:
All Harmonic and Elliott Wave patterns include marked points on charts
Wait for clear price action confirmation before entry
Market conditions can change rapidly

📊 Next analysis: In 1 hour
🤖 Generated by Advanced Pattern Detection System v3.0"""
        
        send_telegram(summary_message)
        time.sleep(2)
        
        print(f"✅ Top 5 patterns analysis completed: {len(top_5_patterns)} charts sent")
        print(f"   - Priority patterns (with marked points): {priority_count}")
        print(f"   - Regular patterns: {len(top_5_patterns) - priority_count}")
        
        return 200
        
    except Exception as e:
        print(f"❌ Send multiple patterns error: {e}")
        import traceback
        traceback.print_exc()
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
            16: "DOJI",
            17: "HAMMER",
            18: "HANGING_MAN",
            19: "SHOOTING_STAR",
            20: "INVERTED_HAMMER",
            21: "MARUBOZU",
            22: "SPINNING_TOP",
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

    def detect_pattern(self, df):
        """Detect single pattern (basic method)"""
        try:
            if len(df) < 20:
                return {
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'INSUFFICIENT_DATA'
                }
            
            # Check candlestick patterns first
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
                'method': 'NO_PATTERN_FOUND'
            }
            
        except Exception as e:
            print(f"Pattern detection error: {e}")
            return {
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.30,
                'method': 'ERROR'
            }

    def detect_all_patterns(self, df):
        """Detect ALL patterns (candlestick + chart)"""
        try:
            if len(df) < 20:
                return [{
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'INSUFFICIENT_DATA'
                }]
        
            all_patterns = []
        
            # Candlestick patterns
            candlestick_patterns = self.detect_all_candlestick_patterns(df)
            all_patterns.extend(candlestick_patterns)
        
            # Chart patterns
            chart_patterns = self.detect_all_chart_patterns(df)
            all_patterns.extend(chart_patterns)
        
            # Filter out NO_PATTERN
            valid_patterns = [p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN']
        
            if not valid_patterns:
                return [{
                    'pattern_id': 0,
                    'pattern_name': 'NO_PATTERN',
                    'confidence': 0.50,
                    'method': 'NO_PATTERNS_FOUND'
                }]
        
            # Sort by confidence
            valid_patterns.sort(key=lambda x: x['confidence'], reverse=True)
            return valid_patterns
        
        except Exception as e:
            print(f"detect_all_patterns error: {e}")
            return [{
                'pattern_id': 0,
                'pattern_name': 'NO_PATTERN',
                'confidence': 0.30,
                'method': 'ERROR'
            }]

    def detect_candlestick_patterns(self, df):
        """Detect single candlestick pattern"""
        try:
            recent_data = df.tail(5)
            if len(recent_data) < 3:
                return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_CANDLES'}
            
            # Single candlestick
            last_candle = recent_data.iloc[-1]
            single_pattern = self.detect_single_candlestick(last_candle)
            if single_pattern['pattern_name'] != 'NO_PATTERN':
                return single_pattern
            
            # Two candlesticks
            if len(recent_data) >= 2:
                two_candle_pattern = self.detect_two_candlestick(recent_data.tail(2))
                if two_candle_pattern['pattern_name'] != 'NO_PATTERN':
                    return two_candle_pattern
            
            # Three candlesticks
            if len(recent_data) >= 3:
                three_candle_pattern = self.detect_three_candlestick(recent_data.tail(3))
                if three_candle_pattern['pattern_name'] != 'NO_PATTERN':
                    return three_candle_pattern
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_CANDLESTICK_PATTERN'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'ERROR'}

    def detect_all_candlestick_patterns(self, df):
        """Detect ALL candlestick patterns"""
        try:
            patterns_found = []
            recent_data = df.tail(5)
            if len(recent_data) < 3:
                return patterns_found
        
            # Single candlestick
            last_candle = recent_data.iloc[-1]
            single_patterns = self.detect_all_single_candlestick(last_candle)
            patterns_found.extend(single_patterns)
        
            # Hanging Man
            hanging_man = self.check_hanging_man(last_candle)
            patterns_found.extend(hanging_man)
        
            # Two candlesticks
            if len(recent_data) >= 2:
                two_patterns = self.detect_all_two_candlestick(recent_data.tail(2))
                patterns_found.extend(two_patterns)
            
                tweezer_patterns = self.check_tweezer_patterns(recent_data.tail(2))
                patterns_found.extend(tweezer_patterns)
        
            # Three candlesticks
            if len(recent_data) >= 3:
                three_patterns = self.detect_all_three_candlestick(recent_data.tail(3))
                patterns_found.extend(three_patterns)
        
            return patterns_found
        
        except Exception as e:
            print(f"detect_all_candlestick_patterns error: {e}")
            return []

    def detect_chart_patterns(self, df):
        """Detect single chart pattern"""
        try:
            highs = df['high'].values[-30:]
            lows = df['low'].values[-30:]
            closes = df['close'].values[-30:]
            
            # Try different patterns
            patterns_to_check = [
                self.detect_descending_triangle(highs, lows),
                self.detect_symmetrical_triangle(highs, lows),
                self.detect_bear_flag(closes, highs, lows),
                self.detect_wedge_patterns(highs, lows, closes),
                self.detect_cup_and_handle(closes, highs, lows),
                self.detect_rectangle(highs, lows),
                self.detect_existing_patterns(df)
            ]
            
            for pattern in patterns_to_check:
                if pattern['pattern_name'] != 'NO_PATTERN':
                    return pattern
            
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_CHART_PATTERN'}
            
        except Exception as e:
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'ERROR'}

    def detect_all_chart_patterns(self, df):
        """Detect ALL chart patterns"""
        try:
            patterns_found = []
            
            patterns_found.extend(self.check_head_shoulders(df))
            patterns_found.extend(self.check_double_top(df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_double_bottom(df['low'].values[-30:], df['close'].values[-30:]))
            patterns_found.extend(self.check_ascending_triangle(df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_descending_triangle(df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_bull_flag(df['close'].values[-30:], df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_bear_flag(df['close'].values[-30:], df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_symmetrical_triangle(df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_wedge_patterns(df['high'].values[-30:], df['low'].values[-30:], df['close'].values[-30:]))
            patterns_found.extend(self.check_cup_and_handle(df['close'].values[-30:], df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_inverse_head_shoulders(df['low'].values[-30:], df['close'].values[-30:]))
            patterns_found.extend(self.check_rectangle(df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_diamond_pattern(df['high'].values[-30:], df['low'].values[-30:]))
            patterns_found.extend(self.check_pennant_pattern(df['high'].values[-30:], df['low'].values[-30:], df['close'].values[-30:]))
        
            # Remove duplicates
            unique_patterns = []
            seen_patterns = set()
            for pattern in patterns_found:
                if pattern['pattern_name'] not in seen_patterns:
                    unique_patterns.append(pattern)
                    seen_patterns.add(pattern['pattern_name'])
        
            return unique_patterns
        
        except Exception as e:
            print(f"detect_all_chart_patterns error: {e}")
            return []

    def predict_signals(self, df):
        """Predict trading signals"""
        try:
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
            current_ema10 = df['ema'].iloc[-1] if not pd.isna(df['ema'].iloc[-1]) else current_price
            current_ema21 = df['ema_21'].iloc[-1] if not pd.isna(df['ema_21'].iloc[-1]) else current_price
            
            all_patterns = self.detect_all_patterns(df.tail(50))
            main_pattern = all_patterns[0] if all_patterns else {'pattern_name': 'NO_PATTERN', 'confidence': 0.5}
            
            confidence = main_pattern.get('confidence', 0.5)
            
            # Determine action
            bearish_patterns = ['HEAD_SHOULDERS', 'DOUBLE_TOP', 'BEAR_FLAG', 'DESCENDING_TRIANGLE', 'WEDGE_RISING']
            bullish_patterns = ['DOUBLE_BOTTOM', 'BULL_FLAG', 'ASCENDING_TRIANGLE', 'WEDGE_FALLING', 'CUP_AND_HANDLE']
            
            if main_pattern['pattern_name'] in bearish_patterns:
                action = "SELL"
                entry_price = current_price - (current_price * 0.0005)
                tp1, tp2, tp3 = current_price * 0.997, current_price * 0.994, current_price * 0.991
                sl = current_price * 1.005
            elif main_pattern['pattern_name'] in bullish_patterns:
                action = "BUY"
                entry_price = current_price + (current_price * 0.0005)
                tp1, tp2, tp3 = current_price * 1.003, current_price * 1.006, current_price * 1.009
                sl = current_price * 0.995
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
            print(f"predict_signals error: {e}")
            current_price = df['close'].iloc[-1]
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

    def detect_all_patterns_with_priority(self, df):
        """Detect patterns with Harmonic/Elliott priority"""
        try:
            if len(df) < 20:
                return [{'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'INSUFFICIENT_DATA'}]
            
            all_patterns = []
            
            # Priority: Harmonic
            harmonic_detector = HarmonicPatternDetector()
            harmonic_result = harmonic_detector.detect_harmonic_patterns(df)
            if harmonic_result['pattern_name'] != 'NO_PATTERN':
                harmonic_result['priority'] = True
                all_patterns.append(harmonic_result)
            
            # Priority: Elliott
            elliott_detector = ElliottWaveDetector()
            elliott_result = elliott_detector.detect_elliott_waves(df)
            if elliott_result['pattern_name'] != 'NO_PATTERN':
                elliott_result['priority'] = True
                all_patterns.append(elliott_result)
            
            # Regular patterns
            regular_patterns = self.detect_all_patterns(df)
            quality_patterns = [p for p in regular_patterns if p['pattern_name'] != 'NO_PATTERN' and p['confidence'] > 0.60]
            for pattern in quality_patterns:
                pattern['priority'] = False
            all_patterns.extend(quality_patterns)
            
            if not all_patterns:
                return [{'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.50, 'method': 'NO_PATTERNS_FOUND'}]
            
            # Sort: priority first, then confidence
            all_patterns.sort(key=lambda x: (not x.get('priority', False), -x['confidence']))
            return all_patterns[:10]
            
        except Exception as e:
            print(f"detect_all_patterns_with_priority error: {e}")
            return [{'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 'confidence': 0.30, 'method': 'ERROR'}]

       

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
        """Detect two-candlestick patterns - ENHANCED VERSION"""
        try:
            first = candles.iloc[0]
            second = candles.iloc[1]
        
            first_body = abs(first['close'] - first['open'])
            second_body = abs(second['close'] - second['open'])
        
            # 🟢 ENGULFING BULLISH - เงื่อนไขผ่อนปรนกว่าเดิม
            if (first['close'] < first['open'] and  # แท่งแรกดำ
                second['close'] > second['open'] and  # แท่งสองขาว
                second['open'] <= first['close'] and  # เปิดใกล้หรือต่ำกว่าปิดแท่งแรก
                second['close'] >= first['open'] and  # ปิดสูงกว่าหรือเท่ากับเปิดแท่งแรก
                second_body > first_body * 0.8):  # ⬅️ ลดจาก 1.1 เป็น 0.8
                return {'pattern_id': 23, 'pattern_name': 'ENGULFING_BULLISH', 
                       'confidence': 0.85, 'method': 'TWO_CANDLESTICK'}
        
            # 🔴 ENGULFING BEARISH - เงื่อนไขผ่อนปรนกว่าเดิม
            if (first['close'] > first['open'] and  # แท่งแรกขาว
                second['close'] < second['open'] and  # แท่งสองดำ
                second['open'] >= first['close'] and  # เปิดใกล้หรือสูงกว่าปิดแท่งแรก
                second['close'] <= first['open'] and  # ปิดต่ำกว่าหรือเท่ากับเปิดแท่งแรก
                second_body > first_body * 0.8):  # ⬅️ ลดจาก 1.1 เป็น 0.8
                return {'pattern_id': 24, 'pattern_name': 'ENGULFING_BEARISH', 
                       'confidence': 0.85, 'method': 'TWO_CANDLESTICK'}
        
            # 🟢 PIERCING LINE - เงื่อนไขผ่อนปรนกว่าเดิม
            first_midpoint = (first['open'] + first['close']) / 2
            if (first['close'] < first['open'] and  # แท่งแรกดำ
                second['close'] > second['open'] and  # แท่งสองขาว
                second['open'] < first['low'] * 1.002 and  # ⬅️ ผ่อนปรนเปิดต่ำกว่า low เล็กน้อย
                second['close'] > first_midpoint * 0.98):  # ⬅️ ผ่อนปรนปิดใกล้ midpoint
                return {'pattern_id': 25, 'pattern_name': 'PIERCING_LINE', 
                       'confidence': 0.80, 'method': 'TWO_CANDLESTICK'}
        
            # 🔴 DARK CLOUD COVER - เงื่อนไขผ่อนปรนกว่าเดิม
            if (first['close'] > first['open'] and  # แท่งแรกขาว
                second['close'] < second['open'] and  # แท่งสองดำ
                second['open'] > first['high'] * 0.998 and  # ⬅️ ผ่อนปรนเปิดสูงกว่า high เล็กน้อย
                second['close'] < first_midpoint * 1.02):  # ⬅️ ผ่อนปรนปิดใกล้ midpoint
                return {'pattern_id': 26, 'pattern_name': 'DARK_CLOUD_COVER', 
                       'confidence': 0.80, 'method': 'TWO_CANDLESTICK'}
        
            # ส่วนอื่นๆ เหมือนเดิม...
            # HARAMI_BULLISH
            if (first['close'] < first['open'] and second['close'] > second['open'] and
                second['open'] > first['close'] and second['close'] < first['open'] and
                second_body < first_body * 0.6):
                return {'pattern_id': 31, 'pattern_name': 'HARAMI_BULLISH', 
                       'confidence': 0.70, 'method': 'TWO_CANDLESTICK'}
        
            # HARAMI_BEARISH
            if (first['close'] > first['open'] and second['close'] < second['open'] and
                second['open'] < first['close'] and second['close'] > first['open'] and
                second_body < first_body * 0.6):
                return {'pattern_id': 32, 'pattern_name': 'HARAMI_BEARISH', 
                       'confidence': 0.70, 'method': 'TWO_CANDLESTICK'}
        
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 
                   'confidence': 0.50, 'method': 'NO_TWO_PATTERN'}
        
        except Exception as e:
            print(f"Two candlestick error: {e}")
            return {'pattern_id': 0, 'pattern_name': 'NO_PATTERN', 
                   'confidence': 0.30, 'method': 'TWO_ERROR'}

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
    """สร้างข้อความ Telegram แบบละเอียด - Send only for Harmonic & Elliott Patterns"""
    try:
        pattern_name = pattern_info.get('pattern_name', 'NO_PATTERN')
        confidence = pattern_info.get('confidence', 0)
        method = pattern_info.get('method', 'UNKNOWN')
        
        # Harmonic and Elliott Wave patterns to detect
        harmonic_patterns = ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD', 
                            'ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']
        
        # Check if pattern is one of the target patterns
        if pattern_name not in harmonic_patterns:
            # Not a Harmonic/Elliott pattern - return "not found" message
            return f"""📊 {symbol} ({timeframe})
⏰ {datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")}

❌ No Harmonic Patterns or Elliott Wave detected
• GARTLEY: ❌
• BUTTERFLY: ❌
• BAT: ❌
• CRAB: ❌
• AB_CD: ❌
• ELLIOTT WAVE 5: ❌
• ELLIOTT WAVE 3: ❌

💰 Current Price: {current_price:.4f}
🔍 Keep monitoring for patterns...

Next scan: In 1 hour"""
        
        # If we reach here, we have a Harmonic or Elliott pattern
        theory = get_pattern_theory(pattern_name)
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        
        message = f"""
🎯 HARMONIC PATTERN DETECTED! 🎯

📊 Symbol: {symbol} ({timeframe})
⏰ {datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")}

🔍 Pattern: {theory['description']}
{confidence_emoji} Confidence: {confidence:.1%}
⚙️ Method: {method}
💰 Current Price: {current_price:.4f}

{theory['theory']}

📈 CHART ANALYSIS:
"""
        
        # Add pattern-specific information
        if 'points' in pattern_info and pattern_info['points']:
            points = pattern_info['points']
            message += "\n🎯 FIBONACCI POINTS:\n"
            for point_name, point_data in points.items():
                if point_data:
                    _, price, _ = point_data
                    message += f"• Point {point_name}: {price:.4f}\n"
                    
        elif 'wave_points' in pattern_info and pattern_info['wave_points']:
            wave_points = pattern_info['wave_points']
            message += "\n🌊 WAVE POINTS:\n"
            for wave_name, wave_data in wave_points.items():
                if wave_data:
                    _, price, _ = wave_data
                    message += f"• Wave {wave_name}: {price:.4f}\n"
        
        # Add trading strategy
        message += f"\n💡 TRADING STRATEGY:\n"
        message += create_trading_strategy(pattern_name, pattern_info)
        
        message += f"\n\n⚠️ Risk Management:\n"
        message += """• Position Size: 1-2% of capital
• Always use Stop Loss
• Confirm with volume & momentum
• Multiple timeframe analysis

🔗 Harmonic Pattern Detection System"""
        
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

# ============= ส่วนที่ 9: Main Function Integration (ต่อจากไฟล์ที่ 3) =============

def analyze_and_send_telegram(df, symbol='UNKNOWN', timeframe='1H', send_telegram=True):
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

             

# เพิ่มการนับและ Log สำหรับติดตาม
def log_pattern_detection(all_patterns, telegram_sent=False):
    """Log pattern detection results for monitoring"""
    try:
        timestamp = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'total_patterns': len(all_patterns),
            'priority_patterns': sum(1 for p in all_patterns if p.get('priority', False)),
            'charts_sent': min(len(all_patterns), 5) if telegram_sent else 0,
            'patterns': [
                {
                    'name': p['pattern_name'],
                    'confidence': f"{p['confidence']:.1%}",
                    'priority': p.get('priority', False),
                    'method': p.get('method', 'UNKNOWN')
                }
                for p in all_patterns[:5]
            ]
        }
        
        print("\n" + "="*60)
        print(f"📊 PATTERN DETECTION LOG - {timestamp}")
        print("="*60)
        print(f"Total Patterns Found: {log_entry['total_patterns']}")
        print(f"Priority Patterns: {log_entry['priority_patterns']}")
        print(f"Charts Sent: {log_entry['charts_sent']}")
        print("\nTop 5 Patterns:")
        for i, p in enumerate(log_entry['patterns'], 1):
            priority_tag = "⭐" if p['priority'] else "  "
            print(f"  {i}. {priority_tag} {p['name']} ({p['confidence']}) - {p['method']}")
        print("="*60 + "\n")
        
        # Optional: Save to file
        log_file = "pattern_detection_log.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{timestamp}\n")
            f.write(f"Patterns: {log_entry['total_patterns']} (Priority: {log_entry['priority_patterns']})\n")
            for p in log_entry['patterns']:
                f.write(f"  - {p['name']} ({p['confidence']}) {'⭐' if p['priority'] else ''}\n")
        
        return log_entry
        
    except Exception as e:
        print(f"Logging error: {e}")
        return None
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

def detect_pattern(df):
    """Simple pattern detection for AI"""
    try:
        highs = df['high'].values
        closes = df['close'].values
        
        if len(highs) < 20:
            return "NO_PATTERN", 50, False
        
        # Double Top Detection
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 2:
            peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
            if abs(peaks_sorted[0][1] - peaks_sorted[1][1]) / peaks_sorted[0][1] < 0.01:
                return "DOUBLE_TOP", 85, True
        
        # Bull/Bear Flag Detection
        if len(closes) >= 25:
            flagpole_move = closes[-15] - closes[-25]
            if flagpole_move > closes[-25] * 0.03:
                return "BULL_FLAG", 75, False
            elif flagpole_move < -closes[-25] * 0.03:
                return "BEAR_FLAG", 75, False
        
        return "NO_PATTERN", 50, False
        
    except Exception as e:
        print(f"Pattern detection error: {e}")
        return "NO_PATTERN", 50, False

# ====================== Flask Routes ======================
# ====================== Flask Routes (Enhanced) ======================
# ============================================
# Flask Route NO.1: API Status Endpoint (JSON)
# ============================================
# นำโค้ดนี้ไปแทนที่ @app.route('/') แบบเก่าของ NO.1

@app.route('/api/status')
def api_status():
    """
    JSON API endpoint for system status
    Returns comprehensive system information in JSON format
    
    Usage:
        GET /api/status
        
    Returns:
        JSON object with system status, AI configuration, and statistics
    """
    try:
        return jsonify({
            "status": "running",
            "system": "XAU/USD AI Multi-LLM Trading System",
            "version": "4.0",
            "systems": {
                "original_system": "active",
                "pattern_system": "active",
                "harmonic_system": "active",
                "ai_multi_llm": "active"
            },
            "ai_analysts": [
                "OpenAI GPT-4",
                "Google Gemini",
                "DeepSeek AI",
                "Grok (xAI)"
            ],
            "statistics": {
                "total_ai_analyses": len(ai_analysis_history),
                "last_analysis_time": ai_analysis_history[-1]['timestamp'].isoformat() if ai_analysis_history else None
            },
            "ai_configuration": {
                "openai": {
                    "configured": bool(OPENAI_API_KEY and HAS_OPENAI),
                    "status": "✅ Active" if (OPENAI_API_KEY and HAS_OPENAI) else "⚠️ Fallback Mode"
                },
                "gemini": {
                    "configured": bool(GEMINI_API_KEY and HAS_GEMINI),
                    "status": "✅ Active" if (GEMINI_API_KEY and HAS_GEMINI) else "⚠️ Fallback Mode"
                },
                "deepseek": {
                    "configured": bool(DEEPSEEK_API_KEY),
                    "status": "✅ Active" if DEEPSEEK_API_KEY else "⚠️ Fallback Mode"
                },
                "grok": {
                    "configured": bool(GROK_API_KEY),
                    "status": "✅ Active" if GROK_API_KEY else "⚠️ Fallback Mode"
                }
            },
            "endpoints": {
                "home": "/",
                "api_status": "/api/status",
                "ai_analyze": "/analyze",
                "ai_history": "/history",
                "test_ai": "/test-ai",
                "health": "/health",
                "original_ai": "/run-ai",
                "pattern_bot": "/run-pattern-bot",
                "harmonic_bot": "/run-harmonic-bot"
            },
            "timestamp": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get AI analysis history"""
    recent = ai_analysis_history[-10:] if ai_analysis_history else []
    
    formatted = []
    for item in recent:
        formatted.append({
            'timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'action': item['consensus']['action'],
            'confidence': item['consensus']['confidence'],
            'votes': item['consensus']['votes'],
            'pattern': item['pattern']['name'],
            'price': item['price']
        })
    
    return jsonify({
        "total": len(ai_analysis_history),
        "recent": formatted
    })

@app.route('/test-ai', methods=['GET'])
def test_ai():
    """Test AI APIs"""
    results = {
        "openai": "Not configured" if not OPENAI_API_KEY else "Configured",
        "gemini": "Not configured" if not GEMINI_API_KEY else "Configured",
        "deepseek": "Not configured" if not DEEPSEEK_API_KEY else "Configured",
        "grok": "Not configured" if not GROK_API_KEY else "Configured"
    }
    return jsonify(results)

# ====================== Scheduler ======================

def schedule_ai_analysis():
    """Run AI analysis every hour"""
    while True:
        try:
            asyncio.run(run_ai_analysis())
        except Exception as e:
            print(f"❌ Scheduled AI analysis error: {e}")
        
        # Wait 1 hour
        time.sleep(3600)

# ====================== Main Entry Point ======================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    🤖 XAU/USD AI MULTI-LLM TRADING SYSTEM (ENHANCED) 🤖      ║
║                                                               ║
║  DUAL SYSTEM:                                                 ║
║    ✓ Original Pattern Recognition System                     ║
║    ✓ NEW: Multi-LLM AI Analysis System                       ║
║                                                               ║
║  AI Analysts:                                                 ║
║    • OpenAI GPT-4         🇺🇸                                 ║
║    • Google Gemini        🇬🇧                                 ║
║    • DeepSeek AI          🇨🇳                                 ║
║    • Grok (xAI)           🇺🇸                                 ║
║                                                               ║
║  Features:                                                    ║
║    ✓ Real-time market analysis                               ║
║    ✓ Multi-AI consensus voting                               ║
║    ✓ Advanced pattern recognition                            ║
║    ✓ News & sentiment analysis                               ║
║    ✓ Automated Telegram alerts                               ║
║    ✓ Flask API for control                                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    
    # ทดสอบระบบ Pattern Detection (ถ้าต้องการ)
    # print("📚 Testing Pattern Detection System...")
    # test_pattern_detection()
    
    
    print("🔑 Checking API credentials...\n")
    
    # Check original APIs
    if not API_KEY:
        print("❌ TwelveData API key missing!")
    else:
        print("✅ TwelveData API configured")
    
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram credentials missing - notifications disabled")
    else:
        print("✅ Telegram configured")
    
    # Check AI APIs
    print("\n🤖 AI Systems:")
    if OPENAI_API_KEY and HAS_OPENAI:
        print("✅ OpenAI GPT-4 ready")
    else:
        print("⚠️ OpenAI not available - using fallback")
    
    if GEMINI_API_KEY and HAS_GEMINI:
        print("✅ Google Gemini ready")
    else:
        print("⚠️ Gemini not available - using fallback")
    
    if DEEPSEEK_API_KEY:
        print("✅ DeepSeek ready")
    else:
        print("⚠️ DeepSeek not available - using fallback")
    
    if GROK_API_KEY:
        print("✅ Grok ready")
    else:
        print("⚠️ Grok not available - using fallback")
    
    print("\n🚀 Starting dual system...\n")
    
    # Start AI scheduler in background
    ai_scheduler_thread = Thread(target=schedule_ai_analysis, daemon=True, name="AI-Scheduler")
    ai_scheduler_thread.start()
    print("✅ AI scheduler started (runs every 1 hour)")
    
    print("\n🌐 Flask API running on http://0.0.0.0:5000")
    print("   Endpoints:")
    print("   • GET /              - System status")
    print("   • GET /analyze       - Trigger AI analysis manually")
    print("   • GET /history       - View AI analysis history")
    print("   • GET /test-ai       - Check AI API configurations")
    
    print("\n⏰ AI Analysis: Every 1 hour (automatic)")
    print("💡 Tip: Visit http://localhost:5000/analyze to run analysis now\n")
    
    # Run Flask
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

@app.route('/debug-all-patterns')
def debug_all_patterns():
    """Debug: แสดง Patterns ทั้งหมดที่ตรวจพบ - FIXED VERSION"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        current_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M:%S")
        current_price = float(shared_df['close'].iloc[-1])
        
        # ใช้เฉพาะ method ที่มีอยู่จริงใน class
        detector = AdvancedPatternDetector()
        
        # Method นี้มีอยู่จริง - ใช้งานได้
        all_patterns = detector.detect_all_patterns(shared_df.tail(50))
        
        # แยกประเภท pattern จาก method name
        candlestick_patterns = []
        chart_patterns = []
        
        for p in all_patterns:
            if p['pattern_name'] == 'NO_PATTERN':
                continue
            
            method = p.get('method', '')
            
            # จำแนกตาม method
            if 'CANDLESTICK' in method:
                candlestick_patterns.append(p)
            else:
                chart_patterns.append(p)
        
        # ตรวจจับ Harmonic
        try:
            harmonic_detector = HarmonicPatternDetector()
            harmonic_result = harmonic_detector.detect_harmonic_patterns(shared_df)
        except Exception as e:
            harmonic_result = {
                'pattern_name': 'ERROR',
                'confidence': 0,
                'method': f'Error: {str(e)}'
            }
        
        # ตรวจจับ Elliott
        try:
            elliott_detector = ElliottWaveDetector()
            elliott_result = elliott_detector.detect_elliott_waves(shared_df)
        except Exception as e:
            elliott_result = {
                'pattern_name': 'ERROR',
                'confidence': 0,
                'method': f'Error: {str(e)}'
            }
        
        # สร้าง response
        return jsonify({
            "status": "success",
            "timestamp": current_time,
            "data_info": {
                "total_candles": len(shared_df),
                "analyzed_candles": 50,
                "current_price": current_price
            },
            
            "candlestick_patterns": {
                "count": len(candlestick_patterns),
                "patterns": [
                    {
                        "name": p['pattern_name'],
                        "confidence": f"{p['confidence']:.1%}",
                        "method": p['method'],
                        "pattern_id": p.get('pattern_id', 'N/A')
                    }
                    for p in sorted(candlestick_patterns, key=lambda x: x['confidence'], reverse=True)
                ]
            },
            
            "chart_patterns": {
                "count": len(chart_patterns),
                "patterns": [
                    {
                        "name": p['pattern_name'],
                        "confidence": f"{p['confidence']:.1%}",
                        "method": p['method'],
                        "pattern_id": p.get('pattern_id', 'N/A')
                    }
                    for p in sorted(chart_patterns, key=lambda x: x['confidence'], reverse=True)
                ]
            },
            
            "harmonic_pattern": {
                "detected": harmonic_result.get('pattern_name') not in ['NO_PATTERN', 'ERROR'],
                "pattern": harmonic_result.get('pattern_name'),
                "confidence": f"{harmonic_result.get('confidence', 0):.1%}",
                "method": harmonic_result.get('method', 'N/A')
            },
            
            "elliott_wave": {
                "detected": elliott_result.get('pattern_name') not in ['NO_PATTERN', 'ERROR'],
                "pattern": elliott_result.get('pattern_name'),
                "confidence": f"{elliott_result.get('confidence', 0):.1%}",
                "method": elliott_result.get('method', 'N/A')
            },
            
            "summary": {
                "total_patterns_found": len(candlestick_patterns) + len(chart_patterns),
                "candlestick_count": len(candlestick_patterns),
                "chart_count": len(chart_patterns),
                "has_harmonic": harmonic_result.get('pattern_name') not in ['NO_PATTERN', 'ERROR'],
                "has_elliott": elliott_result.get('pattern_name') not in ['NO_PATTERN', 'ERROR'],
                "confidence_range": {
                    "min": f"{min([p['confidence'] for p in all_patterns if p['pattern_name'] != 'NO_PATTERN'], default=0):.1%}",
                    "max": f"{max([p['confidence'] for p in all_patterns if p['pattern_name'] != 'NO_PATTERN'], default=0):.1%}"
                } if len([p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN']) > 0 else None
            },
            
            "all_patterns_list": [
                {
                    "rank": i+1,
                    "name": p['pattern_name'],
                    "confidence": f"{p['confidence']:.1%}",
                    "method": p['method'],
                    "type": "Candlestick" if 'CANDLESTICK' in p.get('method', '') else "Chart"
                }
                for i, p in enumerate(
                    sorted(
                        [p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN'],
                        key=lambda x: x['confidence'],
                        reverse=True
                    )
                )
            ]
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/check-methods')
def check_methods():
    """ตรวจสอบว่า AdvancedPatternDetector มี methods อะไรบ้าง"""
    try:
        detector = AdvancedPatternDetector()
        
        # ดึง methods ทั้งหมด
        all_methods = [method for method in dir(detector) if not method.startswith('_')]
        
        # แยกตาม prefix
        detect_methods = [m for m in all_methods if m.startswith('detect')]
        check_methods = [m for m in all_methods if m.startswith('check')]
        other_methods = [m for m in all_methods if not m.startswith('detect') and not m.startswith('check')]
        
        return jsonify({
            "status": "success",
            "class": "AdvancedPatternDetector",
            "total_methods": len(all_methods),
            "detect_methods": detect_methods,
            "check_methods": check_methods,
            "other_methods": other_methods,
            "note": "Use only methods listed here"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/simple-pattern-check')
def simple_pattern_check():
    """ตรวจสอบ patterns แบบง่ายที่สุด"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 10:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        detector = AdvancedPatternDetector()
        
        # ใช้ method เดียวที่แน่ใจว่ามี
        result = detector.detect_pattern(shared_df.tail(50))
        
        return jsonify({
            "status": "success",
            "current_price": float(shared_df['close'].iloc[-1]),
            "pattern_detected": {
                "name": result.get('pattern_name', 'UNKNOWN'),
                "confidence": f"{result.get('confidence', 0):.1%}",
                "method": result.get('method', 'N/A'),
                "pattern_id": result.get('pattern_id', 0)
            },
            "note": "Using basic detect_pattern() method only"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/test-candlestick-only')
def test_candlestick_only():
    """ทดสอบตรวจจับเฉพาะ Candlestick Patterns"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 5:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        detector = AdvancedPatternDetector()
        
        # ตรวจจับเฉพาะ Candlestick
        patterns = detector.detect_all_candlestick_patterns(shared_df)
        
        # กรองเฉพาะที่พบ
        found_patterns = [p for p in patterns if p['pattern_name'] != 'NO_PATTERN']
        
        if not found_patterns:
            msg = "📊 Candlestick Test\n\n❌ No candlestick patterns detected"
            send_telegram(msg)
            
            return jsonify({
                "status": "info",
                "message": "No candlestick patterns found",
                "last_5_candles": {
                    "open": shared_df['open'].tail(5).tolist(),
                    "high": shared_df['high'].tail(5).tolist(),
                    "low": shared_df['low'].tail(5).tolist(),
                    "close": shared_df['close'].tail(5).tolist()
                }
            })
        
        # สร้างข้อความสรุป
        msg = f"""📊 Candlestick Patterns Detected

Found {len(found_patterns)} pattern(s):

"""
        for i, p in enumerate(found_patterns[:5], 1):
            msg += f"{i}. {p['pattern_name']}\n"
            msg += f"   Confidence: {p['confidence']:.1%}\n"
            msg += f"   Type: {p['method']}\n\n"
        
        send_telegram(msg)
        
        return jsonify({
            "status": "success",
            "patterns_found": len(found_patterns),
            "patterns": [
                {
                    "name": p['pattern_name'],
                    "confidence": f"{p['confidence']:.1%}",
                    "method": p['method']
                }
                for p in found_patterns
            ]
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
                shared_df, 'XAUUSD', '1H', send_telegram=True
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
    """Run pattern AI system - Send Telegram once per hour - Enhanced with Top 5 Charts"""
    global last_pattern_sent_hour, message_sent_this_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        current_time = now_th.strftime("%Y-%m-%d %H:%M")
        
        # ✅ เช็คว่าส่งในชั่วโมงนี้แล้วหรือยัง
        if message_sent_this_hour.get('pattern') == current_hour:
            return jsonify({
                "status": "✅ Pattern AI - Keep Alive",
                "mode": "KEEP_ALIVE", 
                "time": current_time,
                "hour": current_hour,
                "telegram_sent": False,
                "system": "Advanced Pattern Detection",
                "note": f"Signals already sent in hour {current_hour}",
                "next_signal_time": f"{(current_hour + 1) % 24:02d}:00"
            })
        
        # ✅ ยังไม่ได้ส่งในชั่วโมงนี้ -> ทำการส่ง
        message_sent_this_hour['pattern'] = current_hour
        last_pattern_sent_hour = current_hour
        
        def send_pattern_task():
            """Background task for pattern detection and sending"""
            try:
                # 🔧 สร้าง timestamp ใหม่ใน thread
                task_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
                
                print(f"\n{'='*60}")
                print(f"🚀 PATTERN TASK STARTED @ {task_time}")
                print(f"{'='*60}")
                
                # Fetch data
                shared_df = get_shared_xau_data()
                if shared_df is None:
                    error_msg = f"❌ Pattern AI Data Error @ {task_time}\nCannot fetch market data"
                    print(f"❌ Data fetch failed")
                    send_telegram(error_msg)
                    return
                
                print(f"✅ Data fetched: {len(shared_df)} rows")
                
                if len(shared_df) < 20:
                    error_msg = f"❌ Pattern AI Data Error @ {task_time}\nInsufficient data for analysis"
                    print(f"❌ Insufficient data: {len(shared_df)} rows")
                    send_telegram(error_msg)
                    return
                
                print(f"✅ Data validation passed")
                
                # Pattern detection
                detector = AdvancedPatternDetector()
                print(f"🔍 Starting pattern detection...")
                
                all_patterns = detector.detect_all_patterns_with_priority(shared_df.tail(50))
                print(f"✅ Pattern detection completed: {len(all_patterns)} patterns found")
                
                # กรอง NO_PATTERN
                all_patterns = [p for p in all_patterns if p['pattern_name'] != 'NO_PATTERN']
                print(f"✅ After filtering: {len(all_patterns)} valid patterns")
                
                # 📊 Log pattern detection
                log_pattern_detection(all_patterns, telegram_sent=False)
                
                # Handle no patterns case
                if not all_patterns:
                    current_price = shared_df['close'].iloc[-1]
                    no_pattern_msg = f"""📊 Pattern AI System
⏰ {task_time}

❌ No patterns detected this hour
Current Price: ${current_price:,.2f}

🔍 Monitoring:
• Harmonic Patterns (GARTLEY, BUTTERFLY, BAT, CRAB, AB=CD)
• Elliott Wave (5-Wave, 3-Wave)
• Classic Chart Patterns

Waiting for clear pattern formation..."""
                    
                    print(f"📤 Sending 'no pattern' message...")
                    telegram_status = send_telegram(no_pattern_msg)
                    print(f"✅ Telegram status: {telegram_status}")
                    return
                
                # นับ priority patterns
                priority_count = sum(1 for p in all_patterns if p.get('priority', False))
                
                print(f"📊 [{task_time}] Patterns found: {len(all_patterns)} total, {priority_count} priority")
                print(f"📊 Patterns: {[p['pattern_name'] for p in all_patterns[:5]]}") 
                
                # ส่งแบบ multiple patterns (สร้าง top 5 charts)
                print(f"📤 Sending multiple patterns message...")
                send_status = send_multiple_patterns_message(all_patterns, shared_df)
                print(f"✅ Send status: {send_status}")
                
                # 📊 Log after sending
                log_pattern_detection(all_patterns, telegram_sent=(send_status == 200))
                
                print(f"✅ [{task_time}] Pattern analysis completed: {min(len(all_patterns), 5)} charts sent")
                print(f"{'='*60}\n")
                
            except Exception as e:
                task_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
                print(f"\n{'='*60}")
                print(f"❌ [{task_time}] Pattern AI send error: {e}")
                print(f"{'='*60}")
                import traceback
                traceback.print_exc()
                error_msg = f"❌ Pattern AI Error @ {task_time}\nError: {str(e)[:100]}"
                send_telegram(error_msg)
                print(f"{'='*60}\n")
        
        # ✅ เริ่ม thread
        Thread(target=send_pattern_task, daemon=True).start()
        
        return jsonify({
            "status": "✅ Pattern AI - Top 5 Charts Sending", 
            "mode": "TELEGRAM_SENT",
            "time": current_time,
            "hour": current_hour,
            "telegram_sent": True,
            "system": "Advanced Pattern Detection + Harmonic + Elliott",
            "note": f"🚀 Top 5 patterns with individual charts sent at {current_time}",
            "charts_count": "Up to 5 charts",
            "next_signal_time": f"{(current_hour + 1) % 24:02d}:00"
        })
        
    except Exception as e:
        print(f"❌ Pattern AI Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": str(e)
        }), 500

@app.route('/test-pattern-bot-direct')
def test_pattern_bot_direct():
    """Test Pattern Bot โดยตรงโดยไม่สนใจ hourly limit - FIXED"""
    try:
        current_time = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")
        
        shared_df = get_shared_xau_data()
        if shared_df is None:
            return jsonify({
                "status": "error",
                "message": "Cannot fetch data"
            }), 500
        
        detector = AdvancedPatternDetector()
        
        # ใช้ method ที่มีอยู่แล้ว - ไม่กรอง confidence สูง
        all_patterns = detector.detect_all_patterns(shared_df.tail(50))
        
        # Harmonic
        harmonic_detector = HarmonicPatternDetector()
        harmonic_result = harmonic_detector.detect_harmonic_patterns(shared_df)
        
        # Elliott
        elliott_detector = ElliottWaveDetector()
        elliott_result = elliott_detector.detect_elliott_waves(shared_df)
        
        # รวม patterns และกรองเฉพาะที่มีคุณภาพ (ลด threshold เหลือ 0.50)
        combined_patterns = []
        
        # Priority patterns (ไม่สนใจ confidence)
        if harmonic_result['pattern_name'] != 'NO_PATTERN':
            harmonic_result['priority'] = True
            combined_patterns.append(harmonic_result)
            
        if elliott_result['pattern_name'] != 'NO_PATTERN':
            elliott_result['priority'] = True
            combined_patterns.append(elliott_result)
        
        # Regular patterns (confidence >= 0.50)
        for p in all_patterns:
            if p['pattern_name'] != 'NO_PATTERN' and p['confidence'] >= 0.50:
                p['priority'] = False
                combined_patterns.append(p)
        
        # เรียงตาม priority แล้วตาม confidence
        combined_patterns.sort(
            key=lambda x: (not x.get('priority', False), -x['confidence'])
        )
        
        if not combined_patterns:
            no_pattern_msg = f"""📊 Test @ {current_time}

❌ No patterns detected (threshold: 50%+)

Checked:
• Harmonic Patterns
• Elliott Wave
• Chart Patterns (50+ candles)
• Candlestick Patterns

Waiting for clearer patterns..."""
            
            telegram_status = send_telegram(no_pattern_msg)
            
            return jsonify({
                "status": "success",
                "message": "No patterns found",
                "telegram_status": telegram_status,
                "threshold_used": "50%",
                "patterns_checked": len(all_patterns)
            })
        
        # ส่งแบบ multiple patterns (Top 5)
        send_status = send_multiple_patterns_message(combined_patterns, shared_df)
        
        # จำแนกประเภท
        candlestick_count = sum(
            1 for p in combined_patterns 
            if p.get('method') in ['SINGLE_CANDLESTICK', 'TWO_CANDLESTICK', 'THREE_CANDLESTICK']
        )
        
        chart_count = sum(
            1 for p in combined_patterns 
            if p.get('method') in ['CHART_PATTERN', 'RULE_BASED', 'TREND_ANALYSIS']
        )
        
        priority_count = sum(1 for p in combined_patterns if p.get('priority', False))
        
        return jsonify({
            "status": "success",
            "message": f"Sent {min(len(combined_patterns), 5)} charts",
            "patterns_found": len(combined_patterns),
            "telegram_status": send_status,
            "patterns": [
                {
                    "rank": i+1,
                    "name": p['pattern_name'],
                    "confidence": f"{p['confidence']:.1%}",
                    "priority": p.get('priority', False),
                    "method": p.get('method')
                }
                for i, p in enumerate(combined_patterns[:5])
            ],
            "breakdown": {
                "priority": priority_count,
                "candlestick": candlestick_count,
                "chart": chart_count,
                "total": len(combined_patterns)
            },
            "threshold_used": "50%"
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/show-recent-candles')
def show_recent_candles():
    """แสดงแท่งเทียน 5 แท่งล่าสุด"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 5:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        recent = shared_df.tail(5)
        
        candles_info = []
        for i, (idx, row) in enumerate(recent.iterrows(), 1):
            body_size = abs(row['close'] - row['open'])
            upper_shadow = row['high'] - max(row['open'], row['close'])
            lower_shadow = min(row['open'], row['close']) - row['low']
            candle_range = row['high'] - row['low']
            
            candle_type = "BULLISH" if row['close'] >= row['open'] else "BEARISH"
            
            body_ratio = body_size / candle_range if candle_range > 0 else 0
            
            # ตรวจสอบว่าเป็น pattern ประเภทไหน
            pattern_hints = []
            
            if body_ratio < 0.1:
                pattern_hints.append("DOJI-like")
            if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
                pattern_hints.append("HAMMER-like")
            if upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
                pattern_hints.append("SHOOTING_STAR-like")
            if body_ratio > 0.9:
                pattern_hints.append("MARUBOZU-like")
            
            candles_info.append({
                "candle": i,
                "time": idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx),
                "type": candle_type,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "body_size": float(body_size),
                "upper_shadow": float(upper_shadow),
                "lower_shadow": float(lower_shadow),
                "body_ratio": f"{body_ratio:.2%}",
                "pattern_hints": pattern_hints if pattern_hints else ["No clear pattern"]
            })
        
        return jsonify({
            "status": "success",
            "current_price": float(recent['close'].iloc[-1]),
            "recent_candles": candles_info,
            "note": "Pattern hints are approximations, not official detections"
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

# ============================================
# Flask Route NO.2: Home Page (HTML Interface)
# ============================================
# นำโค้ดนี้ไปแทนที่ @app.route('/') แบบเก่าของ NO.2

@app.route('/')
def home():
    """
    Home page with comprehensive HTML interface
    Displays system documentation, API endpoints, and AI status
    
    Usage:
        GET /
        
    Returns:
        HTML page with full system documentation
    """
    try:
        # Get AI configuration status
        ai_status = {
            'openai': '✅ Active' if (OPENAI_API_KEY and HAS_OPENAI) else '⚠️ Fallback',
            'gemini': '✅ Active' if (GEMINI_API_KEY and HAS_GEMINI) else '⚠️ Fallback',
            'deepseek': '✅ Active' if DEEPSEEK_API_KEY else '⚠️ Fallback',
            'grok': '✅ Active' if GROK_API_KEY else '⚠️ Fallback'
        }
        
        total_analyses = len(ai_analysis_history)
        last_analysis = ai_analysis_history[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if ai_analysis_history else 'No analyses yet'
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAU AI Trading Bot v4.0 - Multi-LLM Edition</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #ffffff;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #00ff88 0%, #00ddff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        h1 {{
            font-size: 3em;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
            margin-bottom: 10px;
        }}
        
        .status-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .card {{
            background: rgba(26, 26, 46, 0.8);
            border-radius: 12px;
            padding: 25px;
            border-left: 4px solid #00ff88;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
        }}
        
        .card.ai-card {{
            border-left-color: #00ddff;
        }}
        
        .card.harmonic-card {{
            border-left-color: #ff00ff;
        }}
        
        .card-title {{
            font-size: 1.3em;
            color: #00ff88;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .ai-card .card-title {{
            color: #00ddff;
        }}
        
        .harmonic-card .card-title {{
            color: #ff00ff;
        }}
        
        .endpoint {{
            background: rgba(10, 10, 10, 0.6);
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 3px solid #00ff88;
            transition: all 0.3s ease;
        }}
        
        .endpoint:hover {{
            background: rgba(0, 255, 136, 0.1);
            border-left-width: 5px;
        }}
        
        .endpoint.ai-endpoint {{
            border-left-color: #00ddff;
        }}
        
        .endpoint.harmonic-endpoint {{
            border-left-color: #ff00ff;
        }}
        
        .method {{
            display: inline-block;
            background: #007acc;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .badge {{
            display: inline-block;
            background: #ff00ff;
            color: white;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.7em;
            font-weight: bold;
            margin-left: 8px;
        }}
        
        .ai-status-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        
        .ai-item {{
            background: rgba(0, 0, 0, 0.3);
            padding: 12px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .pattern-list {{
            background: rgba(255, 0, 255, 0.1);
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #ff00ff;
            margin: 20px 0;
        }}
        
        .pattern-list ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .pattern-list li {{
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 0, 255, 0.2);
        }}
        
        .pattern-list li:last-child {{
            border-bottom: none;
        }}
        
        .stats-box {{
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 221, 255, 0.1));
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            border: 2px solid #00ddff;
        }}
        
        .stats-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .stats-item:last-child {{
            border-bottom: none;
        }}
        
        .warning-box {{
            background: rgba(255, 170, 0, 0.1);
            border: 2px solid #ffaa00;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 50px;
        }}
        
        a {{
            color: #00ddff;
            text-decoration: none;
            transition: color 0.3s ease;
        }}
        
        a:hover {{
            color: #00ff88;
        }}
        
        code {{
            background: rgba(0, 0, 0, 0.5);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            color: #00ff88;
        }}
        
        @media (max-width: 768px) {{
            h1 {{
                font-size: 2em;
            }}
            
            .grid {{
                grid-template-columns: 1fr;
            }}
            
            .ai-status-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 XAU AI Trading Bot v4.0</h1>
            <div class="status-badge">✅ System Online - Multi-LLM Edition</div>
        </header>
        
        <!-- AI Multi-Agent Status -->
        <div class="card ai-card">
            <div class="card-title">
                🧠 AI Multi-Agent System
            </div>
            <div class="stats-box">
                <div class="stats-item">
                    <span><strong>Total AI Analyses:</strong></span>
                    <span style="color: #00ff88; font-weight: bold;">{total_analyses}</span>
                </div>
                <div class="stats-item">
                    <span><strong>Last Analysis:</strong></span>
                    <span style="color: #00ddff;">{last_analysis}</span>
                </div>
            </div>
            
            <h4 style="color: #00ddff; margin: 20px 0 10px 0;">Active AI Agents:</h4>
            <div class="ai-status-grid">
                <div class="ai-item">
                    <span>🇺🇸 OpenAI GPT-4</span>
                    <span>{ai_status['openai']}</span>
                </div>
                <div class="ai-item">
                    <span>🇬🇧 Google Gemini</span>
                    <span>{ai_status['gemini']}</span>
                </div>
                <div class="ai-item">
                    <span>🇨🇳 DeepSeek AI</span>
                    <span>{ai_status['deepseek']}</span>
                </div>
                <div class="ai-item">
                    <span>🇺🇸 Grok (xAI)</span>
                    <span>{ai_status['grok']}</span>
                </div>
            </div>
        </div>
        
        <!-- Trading Systems -->
        <h2 style="color: #ffaa00; margin: 40px 0 20px 0; text-align: center;">🎯 Trading Systems</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">📊 System 1</div>
                <p><strong>RSI + EMA + Price Change Analysis</strong></p>
                <p style="color: #aaa; font-size: 0.9em; margin-top: 10px;">
                    Machine learning-based technical analysis using RSI indicators and moving averages.
                </p>
            </div>
            
            <div class="card">
                <div class="card-title">📈 System 2</div>
                <p><strong>Classic Chart Pattern Detection</strong></p>
                <p style="color: #aaa; font-size: 0.9em; margin-top: 10px;">
                    Detects traditional patterns: Double Top/Bottom, Head & Shoulders, Triangles, Flags, etc.
                </p>
            </div>
            
            <div class="card harmonic-card">
                <div class="card-title">🌟 System 3</div>
                <p><strong>Harmonic + Elliott Wave</strong></p>
                <p style="color: #aaa; font-size: 0.9em; margin-top: 10px;">
                    Advanced Fibonacci-based patterns: Gartley, Butterfly, Bat, Crab, AB=CD, Elliott Wave.
                </p>
            </div>
            
            <div class="card ai-card">
                <div class="card-title">🤖 System 4 <span class="badge">NEW!</span></div>
                <p><strong>Multi-LLM AI Consensus</strong></p>
                <p style="color: #aaa; font-size: 0.9em; margin-top: 10px;">
                    4 AI agents analyze market data and vote on trading decisions with confidence scores.
                </p>
            </div>
        </div>
        
        <!-- Harmonic Patterns -->
        <div class="pattern-list">
            <h3 style="color: #ff00ff; margin-top: 0;">🌟 Harmonic Patterns Detected</h3>
            <ul>
                <li>🦋 <strong>GARTLEY</strong> - 61.8% XA retracement (High accuracy)</li>
                <li>🦋 <strong>BUTTERFLY</strong> - 127-161.8% XA extension</li>
                <li>🦇 <strong>BAT</strong> - 88.6% XA retracement</li>
                <li>🦀 <strong>CRAB</strong> - 161.8% XA extension (Extreme reversal)</li>
                <li>📐 <strong>AB=CD</strong> - Equal leg structure pattern</li>
                <li>🌊 <strong>ELLIOTT WAVE 5</strong> - Impulse wave pattern</li>
                <li>🌊 <strong>ELLIOTT WAVE 3</strong> - Corrective wave (ABC)</li>
            </ul>
        </div>
        
        <!-- API Endpoints -->
        <h2 style="color: #ffaa00; margin: 40px 0 20px 0; text-align: center;">📡 API Endpoints</h2>
        
        <h3 style="color: #00ddff; margin: 30px 0 15px 0;">🤖 AI Multi-LLM Endpoints</h3>
        
        <div class="endpoint ai-endpoint">
            <span class="method">GET</span> <strong>/analyze</strong> <span class="badge">AI</span>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>Trigger AI Multi-LLM Analysis:</strong> Queries all 4 AI agents and returns consensus decision with trading signals.
            </p>
        </div>
        
        <div class="endpoint ai-endpoint">
            <span class="method">GET</span> <strong>/history</strong> <span class="badge">AI</span>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>View AI Analysis History:</strong> Returns the last 10 AI analyses with timestamps and decisions.
            </p>
        </div>
        
        <div class="endpoint ai-endpoint">
            <span class="method">GET</span> <strong>/test-ai</strong> <span class="badge">AI</span>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>Test AI Configurations:</strong> Check which AI APIs are properly configured.
            </p>
        </div>
        
        <div class="endpoint ai-endpoint">
            <span class="method">GET</span> <strong>/api/status</strong>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>System Status (JSON):</strong> Comprehensive system information in JSON format.
            </p>
        </div>
        
        <h3 style="color: #00ff88; margin: 30px 0 15px 0;">📊 Original Systems</h3>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/health</strong>
            <p style="margin-top: 10px; color: #ddd;">Health check endpoint for monitoring</p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/run-ai</strong>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>Original AI System:</strong> RSI + EMA analysis
                <br><em>Frequency:</em> Every 3 minutes | <em>Output:</em> Once per hour
            </p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/run-pattern-bot</strong>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>Classic Pattern AI:</strong> Traditional chart patterns
                <br><em>Frequency:</em> Every 3 minutes | <em>Output:</em> Once per hour
            </p>
        </div>
        
        <div class="endpoint harmonic-endpoint">
            <span class="method">GET</span> <strong>/run-harmonic-bot</strong>
            <p style="margin-top: 10px; color: #ddd;">
                <strong>Harmonic + Elliott Wave AI:</strong> Advanced Fibonacci patterns
                <br><em>Frequency:</em> Every 3 minutes | <em>Output:</em> Once per hour
            </p>
        </div>
        
        <h3 style="color: #ff00ff; margin: 30px 0 15px 0;">🧪 Test Endpoints</h3>
        
        <div class="endpoint harmonic-endpoint">
            <span class="method">GET</span> <strong>/test-harmonic</strong>
            <p style="margin-top: 10px; color: #ddd;">Test harmonic pattern detection (JSON response)</p>
        </div>
        
        <div class="endpoint harmonic-endpoint">
            <span class="method">GET</span> <strong>/test-specific-harmonic?pattern=GARTLEY</strong>
            <p style="margin-top: 10px; color: #ddd;">
                Test specific harmonic pattern
                <br><em>Parameters:</em> GARTLEY, BUTTERFLY, BAT, CRAB, AB_CD
            </p>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/test-telegram</strong>
            <p style="margin-top: 10px; color: #ddd;">Test Telegram connection</p>
        </div>
        
        <!-- Setup Guide -->
        <div class="card ai-card" style="margin-top: 40px;">
            <div class="card-title">⚡ UptimeRobot Setup Guide</div>
            <p style="margin-bottom: 20px;">Configure 4 monitors for complete system coverage:</p>
            
            <div style="margin: 15px 0;">
                <strong style="color: #00ff88;">Monitor 1:</strong> <code>/run-ai</code> → Every 3 minutes
                <br><span style="color: #aaa; margin-left: 20px;">→ Original AI signals (RSI+EMA)</span>
            </div>
            
            <div style="margin: 15px 0;">
                <strong style="color: #00ff88;">Monitor 2:</strong> <code>/run-pattern-bot</code> → Every 3 minutes
                <br><span style="color: #aaa; margin-left: 20px;">→ Classic pattern signals</span>
            </div>
            
            <div style="margin: 15px 0;">
                <strong style="color: #ff00ff;">Monitor 3:</strong> <code>/run-harmonic-bot</code> → Every 3 minutes
                <br><span style="color: #aaa; margin-left: 20px;">→ Harmonic + Elliott Wave signals</span>
            </div>
            
            <div style="margin: 15px 0;">
                <strong style="color: #00ddff;">Monitor 4:</strong> <code>/analyze</code> → Every 1 hour <span class="badge">AI</span>
                <br><span style="color: #aaa; margin-left: 20px;">→ Multi-LLM consensus signals</span>
            </div>
            
            <div class="stats-box" style="margin-top: 25px;">
                <h4 style="color: #00ff88; margin-bottom: 15px;">📊 Expected Results:</h4>
                <ul style="list-style: none; padding: 0;">
                    <li style="padding: 8px 0;">🤖 <strong>1 signal/hour:</strong> Original AI (RSI+EMA)</li>
                    <li style="padding: 8px 0;">📈 <strong>1 signal/hour:</strong> Classic Patterns</li>
                    <li style="padding: 8px 0; color: #ff00ff;">🎯 <strong>1 signal/hour:</strong> Harmonic + Elliott Wave</li>
                    <li style="padding: 8px 0; color: #00ddff;">🧠 <strong>1 signal/hour:</strong> Multi-LLM Consensus</li>
                </ul>
                <p style="color: #00ff88; font-weight: bold; margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
                    ✅ Total: 4 independent signals per hour!
                </p>
            </div>
        </div>
        
        <!-- Risk Disclaimer -->
        <div class="warning-box">
            <h3 style="color: #ffaa00; margin-top: 0;">⚠️ Risk Disclaimer</h3>
            <p>
                This is an automated AI trading bot for <strong>educational and research purposes</strong>. 
                AI-generated signals are not financial advice. Harmonic patterns and Elliott Wave analysis 
                require experience and interpretation. Always use proper risk management (1-2% per trade). 
                Past performance does not guarantee future results. Trade at your own risk.
            </p>
        </div>
        
        <footer>
            <p>🚀 <strong>XAU AI Trading Bot v4.0</strong></p>
            <p style="margin-top: 10px;">Powered by Multi-LLM Technology | Advanced Pattern Detection</p>
            <p style="margin-top: 20px; font-size: 0.9em;">
                <a href="/api/status">JSON API</a> | 
                <a href="/analyze">Trigger Analysis</a> | 
                <a href="/history">View History</a>
            </p>
        </footer>
    </div>
</body>
</html>
"""
        return html_content
        
    except Exception as e:
        return f"""
        <html>
        <body style="background: #0a0a0a; color: #ff4444; font-family: monospace; padding: 40px;">
            <h1>⚠️ Error Loading Home Page</h1>
            <p>{str(e)}</p>
            <a href="/api/status" style="color: #00ddff;">View JSON Status</a>
        </body>
        </html>
        """, 500


@app.route('/run-harmonic-bot')
def run_harmonic_bot():
    """Run Harmonic + Elliott Wave patterns - Send once per hour - Fixed"""
    global last_harmonic_sent_hour, message_sent_this_hour
    
    try:
        now_th = datetime.now(ZoneInfo("Asia/Bangkok"))
        current_hour = now_th.hour
        current_time = now_th.strftime("%Y-%m-%d %H:%M")
        
        # Reset เมื่อเปลี่ยนชั่วโมง
        if current_hour != last_harmonic_sent_hour:
            last_harmonic_sent_hour = current_hour
            message_sent_this_hour['harmonic'] = None
        
        # ตรวจสอบว่าส่งในชั่วโมงนี้แล้วหรือยัง
        if message_sent_this_hour['harmonic'] == current_hour:
            # ส่งแล้ว - ไม่ส่งอีก
            return jsonify({
                "status": "✅ ระบบพร้อม",
                "message": f"สัญญาณ Harmonic ส่งแล้วในชั่วโมงที่ {current_hour}",
                "next_send": f"{current_hour + 1}:00"
            })
        
        # ส่งเครื่องหมายว่าส่งแล้ว
        message_sent_this_hour['harmonic'] = current_hour
        
        def send_harmonic_task():
            """ฟังก์ชันส่งสัญญาณ Harmonic ไปยัง Telegram"""
            try:
                # ดึงข้อมูล XAU
                shared_df = get_shared_xau_data()
                if shared_df is None:
                    error_msg = f"❌ ข้อผิดพลาด: ไม่สามารถดึงข้อมูลได้ @ {current_time}"
                    send_telegram(error_msg)
                    print(error_msg)
                    return
        
                # ตรวจจับแพทเทิร์น
                pattern_result, _ = detect_all_patterns_enhanced(
                    shared_df, 'XAUUSD', '1H'
                )
        
                # สร้างข้อความ
                telegram_msg = create_enhanced_telegram_message(
                    pattern_result, 'XAUUSD', '1H', shared_df['close'].iloc[-1]
                )
        
                # ส่งไปยัง Telegram
                status = send_telegram(telegram_msg)
                
                if status == 200:
                    print(f"✅ [{current_time}] ส่งสัญญาณ Harmonic สำเร็จ")
                else:
                    print(f"⚠️ [{current_time}] ส่งสัญญาณ Harmonic แต่ status: {status}")
                
                return True
        
            except Exception as e:
                error_msg = f"❌ ข้อผิดพลาด Harmonic Bot: {str(e)[:100]}"
                print(error_msg)
                send_telegram(error_msg)
                return False
        
        # เรียกฟังก์ชันส่งข้อความแบบ background
        Thread(target=send_harmonic_task, daemon=True).start()
        
        return jsonify({
            "status": "✅ สัญญาณถูกส่ง",
            "time": current_time,
            "message": "สัญญาณ Harmonic + Elliott Wave ถูกส่งไป Telegram"
        })
            
    except Exception as e:
        print(f"❌ ข้อผิดพลาด: {e}")
        return jsonify({
            "status": "❌ ข้อผิดพลาด", 
            "message": str(e)
        }), 500

@app.route('/test-harmonic-chart')
def test_harmonic_chart():
    """ทดสอบวาดกราฟ Harmonic Pattern"""
    try:
        pattern_type = request.args.get('pattern', 'GARTLEY')
        valid_patterns = ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD']
        
        if pattern_type not in valid_patterns:
            return jsonify({
                "status": "error",
                "message": f"Invalid pattern. Choose from: {valid_patterns}"
            })
        
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        # ตรวจจับ pattern จริง
        harmonic_detector = HarmonicPatternDetector()
        result = harmonic_detector.detect_harmonic_patterns(shared_df)
        
        # ถ้าไม่ตรงกับที่ต้องการ ให้สร้าง mock data
        if result.get('pattern_name') != pattern_type:
            # สร้าง mock points สำหรับทดสอบ
            result = create_mock_harmonic_pattern(shared_df, pattern_type)
        
        # สร้าง trading signals
        current_price = float(shared_df['close'].iloc[-1])
        trading_signals = {
            'current_price': current_price,
            'entry_price': current_price,
            'tp1': current_price * 1.003,
            'tp2': current_price * 1.006,
            'tp3': current_price * 1.009,
            'sl': current_price * 0.995,
            'action': 'BUY' if pattern_type in ['GARTLEY', 'BAT'] else 'SELL',
            'confidence': result.get('confidence', 0.75),
            'rsi': 50.0,
            'ema10': current_price,
            'ema21': current_price
        }
        
        # สร้างกราฟ
        chart_buffer = create_candlestick_chart(shared_df, trading_signals, result)
        
        if chart_buffer:
            msg = f"""🧪 Testing {pattern_type} Harmonic Pattern

Pattern: {pattern_type}
Confidence: {result.get('confidence', 0.75):.1%}
Method: {result.get('method', 'HARMONIC')}

Check the chart for XABCD points and Fibonacci ratios!"""
            
            status = send_telegram_with_chart(msg, chart_buffer)
            
            return jsonify({
                "status": "success",
                "message": f"{pattern_type} chart sent to Telegram",
                "telegram_status": status,
                "pattern_detected": result.get('pattern_name'),
                "confidence": result.get('confidence', 0)
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Chart creation failed"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/test-elliott-chart')
def test_elliott_chart():
    """ทดสอบวาดกราฟ Elliott Wave"""
    try:
        wave_type = request.args.get('type', '5')  # 5 or 3
        
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        # ตรวจจับ Elliott Wave
        elliott_detector = ElliottWaveDetector()
        result = elliott_detector.detect_elliott_waves(shared_df)
        
        pattern_name = f'ELLIOTT_WAVE_{wave_type}'
        
        # ถ้าไม่พบ pattern ให้สร้าง mock
        if result.get('pattern_name') == 'NO_PATTERN':
            result = create_mock_elliott_wave(shared_df, wave_type)
            result['pattern_name'] = pattern_name
        
        current_price = float(shared_df['close'].iloc[-1])
        trading_signals = {
            'current_price': current_price,
            'entry_price': current_price,
            'tp1': current_price * 1.005,
            'tp2': current_price * 1.010,
            'tp3': current_price * 1.015,
            'sl': current_price * 0.990,
            'action': 'BUY',
            'confidence': result.get('confidence', 0.70),
            'rsi': 50.0,
            'ema10': current_price,
            'ema21': current_price
        }
        
        chart_buffer = create_candlestick_chart(shared_df, trading_signals, result)
        
        if chart_buffer:
            msg = f"""🧪 Testing Elliott Wave {wave_type}

Pattern: {pattern_name}
Confidence: {result.get('confidence', 0.70):.1%}
Method: {result.get('method', 'ELLIOTT_WAVE')}

🌊 Check the chart for wave structure!"""
            
            status = send_telegram_with_chart(msg, chart_buffer)
            
            return jsonify({
                "status": "success",
                "message": f"Elliott Wave {wave_type} chart sent",
                "telegram_status": status
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Chart creation failed"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/test-all-patterns')
def test_all_patterns():
    """ทดสอบวาดกราฟทุก Pattern แบบ"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        current_price = float(shared_df['close'].iloc[-1])
        
        # รายการ patterns ที่จะทดสอบ
        test_patterns = [
            'DOUBLE_TOP',
            'DOUBLE_BOTTOM', 
            'HEAD_SHOULDERS',
            'GARTLEY',
            'BUTTERFLY',
            'BAT',
            'CRAB',
            'AB_CD',
            'ELLIOTT_WAVE_5',
            'ELLIOTT_WAVE_3',
            'ASCENDING_TRIANGLE',
            'DESCENDING_TRIANGLE'
        ]
        
        results = []
        
        for pattern_name in test_patterns:
            try:
                # สร้าง mock pattern info
                if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
                    pattern_info = create_mock_harmonic_pattern(shared_df, pattern_name)
                elif pattern_name == 'AB_CD':
                    pattern_info = create_mock_abcd_pattern(shared_df)
                elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
                    wave_type = '5' if pattern_name == 'ELLIOTT_WAVE_5' else '3'
                    pattern_info = create_mock_elliott_wave(shared_df, wave_type)
                else:
                    pattern_info = {
                        'pattern_name': pattern_name,
                        'confidence': 0.75,
                        'method': 'TEST'
                    }
                
                trading_signals = {
                    'current_price': current_price,
                    'entry_price': current_price,
                    'tp1': current_price * 1.003,
                    'tp2': current_price * 1.006,
                    'tp3': current_price * 1.009,
                    'sl': current_price * 0.995,
                    'action': 'BUY',
                    'confidence': 0.75,
                    'rsi': 50.0,
                    'ema10': current_price,
                    'ema21': current_price
                }
                
                chart_buffer = create_candlestick_chart(shared_df, trading_signals, pattern_info)
                
                if chart_buffer:
                    msg = f"""📊 Pattern Test: {pattern_name}

Testing pattern visualization and marking system.
Check if all points and lines are correctly drawn!"""
                    
                    status = send_telegram_with_chart(msg, chart_buffer)
                    results.append({
                        'pattern': pattern_name,
                        'status': 'sent' if status == 200 else 'failed',
                        'telegram_status': status
                    })
                    
                    # หน่วงเวลาระหว่างการส่ง
                    time.sleep(3)
                else:
                    results.append({
                        'pattern': pattern_name,
                        'status': 'chart_creation_failed'
                    })
                    
            except Exception as e:
                results.append({
                    'pattern': pattern_name,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            "status": "success",
            "message": f"Tested {len(test_patterns)} patterns",
            "results": results,
            "total_sent": sum(1 for r in results if r.get('status') == 'sent')
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-pattern-drawing')
def test_pattern_drawing():
    """ทดสอบวาด Pattern ที่ระบุ"""
    try:
        pattern_name = request.args.get('pattern', 'BULL_FLAG')
        
        all_patterns = [
            'DOUBLE_TOP', 'DOUBLE_BOTTOM', 'HEAD_SHOULDERS', 'INVERSE_HEAD_SHOULDERS',
            'ASCENDING_TRIANGLE', 'DESCENDING_TRIANGLE', 'SYMMETRICAL_TRIANGLE',
            'BULL_FLAG', 'BEAR_FLAG', 'PENNANT',
            'WEDGE_RISING', 'WEDGE_FALLING',
            'CUP_AND_HANDLE', 'RECTANGLE', 'DIAMOND',
            'GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD',
            'ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3'
        ]
        
        if pattern_name not in all_patterns:
            return jsonify({
                "status": "error",
                "message": f"Invalid pattern. Choose from: {all_patterns}"
            })
        
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        # สร้าง pattern info
        if pattern_name in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB']:
            pattern_info = create_mock_harmonic_pattern(shared_df, pattern_name)
        elif pattern_name == 'AB_CD':
            pattern_info = create_mock_abcd_pattern(shared_df)
        elif pattern_name in ['ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']:
            wave_type = '5' if pattern_name == 'ELLIOTT_WAVE_5' else '3'
            pattern_info = create_mock_elliott_wave(shared_df, wave_type)
        else:
            pattern_info = {
                'pattern_name': pattern_name,
                'confidence': 0.75,
                'method': 'TEST'
            }
        
        current_price = float(shared_df['close'].iloc[-1])
        
        # กำหนด action ตาม pattern
        bullish_patterns = ['DOUBLE_BOTTOM', 'INVERSE_HEAD_SHOULDERS', 'ASCENDING_TRIANGLE',
                           'BULL_FLAG', 'WEDGE_FALLING', 'CUP_AND_HANDLE']
        action = 'BUY' if pattern_name in bullish_patterns else 'SELL'
        
        trading_signals = {
            'current_price': current_price,
            'entry_price': current_price,
            'tp1': current_price * (1.003 if action == 'BUY' else 0.997),
            'tp2': current_price * (1.006 if action == 'BUY' else 0.994),
            'tp3': current_price * (1.009 if action == 'BUY' else 0.991),
            'sl': current_price * (0.995 if action == 'BUY' else 1.005),
            'action': action,
            'confidence': 0.75,
            'rsi': 50.0,
            'ema10': current_price,
            'ema21': current_price
        }
        
        # สร้างกราฟ
        chart_buffer = create_candlestick_chart(shared_df, trading_signals, pattern_info)
        
        if chart_buffer:
            msg = f"""🧪 Testing Pattern Drawing: {pattern_name}

Pattern: {pattern_name}
Action: {action}
Confidence: 75%

✅ Check if pattern lines and points are correctly drawn!"""
            
            status = send_telegram_with_chart(msg, chart_buffer)
            
            return jsonify({
                "status": "success",
                "message": f"{pattern_name} chart sent to Telegram",
                "telegram_status": status,
                "pattern": pattern_name,
                "action": action
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Chart creation failed"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/test-all-pattern-drawings')
def test_all_pattern_drawings():
    """ทดสอบวาดทุก Pattern ทีละอัน"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        all_patterns = [
            'BULL_FLAG', 'BEAR_FLAG', 'PENNANT',
            'ASCENDING_TRIANGLE', 'DESCENDING_TRIANGLE', 'SYMMETRICAL_TRIANGLE',
            'WEDGE_RISING', 'WEDGE_FALLING',
            'CUP_AND_HANDLE', 'RECTANGLE', 'DIAMOND',
            'INVERSE_HEAD_SHOULDERS'
        ]
        
        results = []
        current_price = float(shared_df['close'].iloc[-1])
        
        for pattern_name in all_patterns:
            try:
                pattern_info = {
                    'pattern_name': pattern_name,
                    'confidence': 0.75,
                    'method': 'TEST'
                }
                
                bullish = pattern_name in ['BULL_FLAG', 'ASCENDING_TRIANGLE', 
                                          'WEDGE_FALLING', 'CUP_AND_HANDLE', 
                                          'INVERSE_HEAD_SHOULDERS']
                action = 'BUY' if bullish else 'SELL'
                
                trading_signals = {
                    'current_price': current_price,
                    'entry_price': current_price,
                    'tp1': current_price * (1.003 if bullish else 0.997),
                    'tp2': current_price * (1.006 if bullish else 0.994),
                    'tp3': current_price * (1.009 if bullish else 0.991),
                    'sl': current_price * (0.995 if bullish else 1.005),
                    'action': action,
                    'confidence': 0.75,
                    'rsi': 50.0,
                    'ema10': current_price,
                    'ema21': current_price
                }
                
                chart_buffer = create_candlestick_chart(shared_df, trading_signals, pattern_info)
                
                if chart_buffer:
                    emoji_map = {
                        'BULL_FLAG': '🚩', 'BEAR_FLAG': '🏴',
                        'PENNANT': '🎏', 'ASCENDING_TRIANGLE': '🔺',
                        'DESCENDING_TRIANGLE': '🔻', 'SYMMETRICAL_TRIANGLE': '⚖️',
                        'WEDGE_RISING': '📐', 'WEDGE_FALLING': '📐',
                        'CUP_AND_HANDLE': '☕', 'RECTANGLE': '🔲',
                        'DIAMOND': '💎', 'INVERSE_HEAD_SHOULDERS': '🔄'
                    }
                    
                    msg = f"""{emoji_map.get(pattern_name, '📊')} {pattern_name}

Action: {action}
Test #{all_patterns.index(pattern_name) + 1}/{len(all_patterns)}

Check pattern visualization!"""
                    
                    status = send_telegram_with_chart(msg, chart_buffer)
                    results.append({
                        'pattern': pattern_name,
                        'status': 'sent' if status == 200 else 'failed',
                        'telegram_status': status
                    })
                    
                    print(f"✅ {pattern_name} sent")
                    time.sleep(3)
                else:
                    results.append({
                        'pattern': pattern_name,
                        'status': 'chart_failed'
                    })
                    print(f"❌ {pattern_name} chart creation failed")
                    
            except Exception as e:
                results.append({
                    'pattern': pattern_name,
                    'status': 'error',
                    'error': str(e)
                })
                print(f"❌ {pattern_name} error: {e}")
        
        return jsonify({
            "status": "success",
            "message": f"Tested {len(all_patterns)} patterns",
            "results": results,
            "total_sent": sum(1 for r in results if r.get('status') == 'sent'),
            "total_failed": sum(1 for r in results if r.get('status') != 'sent')
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/test-top5-charts')
def test_top5_charts():
    """ทดสอบการสร้าง Top 5 charts พร้อม priority patterns"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        detector = AdvancedPatternDetector()
        
        # ตรวจหา Harmonic
        harmonic_detector = HarmonicPatternDetector()
        harmonic_result = harmonic_detector.detect_harmonic_patterns(shared_df)
        
        # ตรวจหา Elliott
        elliott_detector = ElliottWaveDetector()
        elliott_result = elliott_detector.detect_elliott_waves(shared_df)
        
        # ตรวจหา Classic
        classic_patterns = detector.detect_all_patterns(shared_df.tail(50))
        
        # รวมทั้งหมด
        all_patterns = []
        if harmonic_result['pattern_name'] != 'NO_PATTERN':
            all_patterns.append(harmonic_result)
        if elliott_result['pattern_name'] != 'NO_PATTERN':
            all_patterns.append(elliott_result)
        all_patterns.extend([p for p in classic_patterns if p['pattern_name'] != 'NO_PATTERN'])
        
        if not all_patterns:
            return jsonify({
                "status": "info",
                "message": "No patterns detected for testing"
            })
        
        # ส่งไปยัง Telegram
        send_status = send_multiple_patterns_message(all_patterns, shared_df)
        
        return jsonify({
            "status": "success",
            "message": f"Top 5 charts test completed",
            "patterns_found": len(all_patterns),
            "charts_created": min(len(all_patterns), 5),
            "telegram_status": send_status,
            "patterns": [
                {
                    "name": p['pattern_name'],
                    "confidence": p['confidence'],
                    "priority": p['pattern_name'] in ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'AB_CD', 'ELLIOTT_WAVE_5', 'ELLIOTT_WAVE_3']
                } 
                for p in all_patterns[:5]
            ]
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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
    
@app.route('/debug-pattern-priority')
def debug_pattern_priority():
    """Debug endpoint to check priority pattern detection"""
    try:
        shared_df = get_shared_xau_data()
        if shared_df is None or len(shared_df) < 50:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            })
        
        # Test all detection methods
        detector = AdvancedPatternDetector()
        
        # 1. Harmonic
        harmonic_detector = HarmonicPatternDetector()
        harmonic_result = harmonic_detector.detect_harmonic_patterns(shared_df)
        
        # 2. Elliott
        elliott_detector = ElliottWaveDetector()
        elliott_result = elliott_detector.detect_elliott_waves(shared_df)
        
        # 3. Classic
        classic_patterns = detector.detect_all_patterns(shared_df.tail(50))
        
        # 4. All with priority
        all_with_priority = detector.detect_all_patterns_with_priority(shared_df.tail(50))
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "harmonic_detection": {
                "pattern": harmonic_result.get('pattern_name'),
                "confidence": harmonic_result.get('confidence'),
                "priority": True,
                "has_points": 'points' in harmonic_result
            },
            "elliott_detection": {
                "pattern": elliott_result.get('pattern_name'),
                "confidence": elliott_result.get('confidence'),
                "priority": True,
                "has_waves": 'wave_points' in elliott_result
            },
            "classic_patterns_count": len([p for p in classic_patterns if p['pattern_name'] != 'NO_PATTERN']),
            "all_patterns_with_priority": {
                "total": len(all_with_priority),
                "priority_count": sum(1 for p in all_with_priority if p.get('priority', False)),
                "patterns": [
                    {
                        "rank": i+1,
                        "name": p['pattern_name'],
                        "confidence": f"{p['confidence']:.1%}",
                        "priority": p.get('priority', False),
                        "method": p.get('method')
                    }
                    for i, p in enumerate(all_with_priority[:10])
                ]
            },
            "priority_patterns": [
                p['pattern_name'] for p in all_with_priority if p.get('priority', False)
            ],
            "regular_patterns": [
                p['pattern_name'] for p in all_with_priority if not p.get('priority', False)
            ][:5]
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
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
