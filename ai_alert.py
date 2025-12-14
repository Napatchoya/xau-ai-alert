# === ai_alert.py (part 1/4) ===
# Header, imports, config, utilities, and MultiAnalystSystem (LLM orchestration)
from __future__ import annotations

import os
import sys
import time
import json
import logging
import traceback
import requests
import asyncio
import concurrent.futures
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Flask is used (per your choice)
from flask import Flask, request, jsonify

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai_alert")

# -----------------------
# Config & Globals
# -----------------------
GLOBAL_LOG_BUFFER: List[str] = []
_DEFAULT_TIMEOUT = 30

# Environment variables for Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# LLM API keys (may or may not be used depending on mode)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Control: by default we use simulated LLMs. Set USE_REAL_LLM=1 to attempt real calls (requires libs/keys)
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "0") in ("1", "true", "True")

# Gating: avoid sending the same message multiple times in same UTC hour
_last_sent_hour: Optional[str] = None
_message_sent_this_hour: Dict[str, bool] = {}

# History & debug
ai_analysis_history: List[Dict[str, Any]] = []
MAX_HISTORY = 500

# Executor for blocking HTTP calls (Telegram etc.)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

# Flask app
app = Flask(__name__)

# -----------------------
# Utilities
# -----------------------
def _now_hour_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")

def _append_log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    GLOBAL_LOG_BUFFER.append(f"{ts} {msg}")
    if len(GLOBAL_LOG_BUFFER) > 2000:
        GLOBAL_LOG_BUFFER.pop(0)
    logger.debug(msg)

# -----------------------
# MultiAnalystSystem: orchestrate multiple LLMs (OpenAI, Gemini, Grok, DeepSeek)
# - By default uses deterministic simulated responses (safe to run without API keys)
# - If USE_REAL_LLM=1 and keys present, will attempt to call provider SDK/HTTP
# -----------------------
class MultiAnalystSystem:
    def __init__(self, api_keys: Dict[str, Optional[str]], executor_workers: int = 6, timeout: int = _DEFAULT_TIMEOUT):
        self.api_keys = api_keys
        self.timeout = int(timeout)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=executor_workers)

    async def _run_blocking(self, fn, *args, timeout: int = None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    # ---------- OpenAI ----------
    async def ask_openai(self, prompt: str, timeout: int = None) -> Dict[str, Any]:
        model_name = "openai"
        timeout = self.timeout if timeout is None else timeout
        api_key = self.api_keys.get("openai")
        # If not using real LLM, simulate
        if not USE_REAL_LLM or not api_key:
            return await self._run_blocking(lambda: {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None})
        # Attempt real call (try python openai SDK, fallback to HTTP)
        def _call_openai():
            try:
                import openai  # type: ignore
                openai.api_key = api_key
                resp = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    timeout=timeout,
                    max_tokens=512,
                )
                # parse choices
                text = ""
                try:
                    choices = resp.choices
                    if choices and len(choices) > 0:
                        msg = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
                        if isinstance(msg, dict):
                            text = msg.get("content", "")
                        else:
                            # some SDK returns objects with .content
                            text = getattr(choices[0], "message", {}).get("content", "") if isinstance(choices[0], dict) else ""
                except Exception:
                    # fallback: try resp['choices']
                    try:
                        text = resp["choices"][0]["message"]["content"]
                    except Exception:
                        text = str(resp)
                return {"model": model_name, "text": (text or "").strip(), "error": None, "raw": resp}
            except Exception as e:
                logger.exception("OpenAI request failed")
                return {"model": model_name, "text": "", "error": str(e), "raw": None}
        return await self._run_blocking(_call_openai)

    # ---------- Gemini ----------
    async def ask_gemini(self, prompt: str, timeout: int = None) -> Dict[str, Any]:
        model_name = "gemini"
        timeout = self.timeout if timeout is None else timeout
        api_key = self.api_keys.get("gemini")
        if not USE_REAL_LLM or not api_key:
            return await self._run_blocking(lambda: {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None})
        def _call_gemini():
            try:
                # Placeholder: implement real Gemini HTTP call if needed
                # For now fallback to simulated response to ensure safe behavior
                return {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None}
            except Exception as e:
                logger.exception("Gemini request failed")
                return {"model": model_name, "text": "", "error": str(e), "raw": None}
        return await self._run_blocking(_call_gemini)

    # ---------- DeepSeek ----------
    async def ask_deepseek(self, prompt: str, timeout: int = None) -> Dict[str, Any]:
        model_name = "deepseek"
        timeout = self.timeout if timeout is None else timeout
        api_key = self.api_keys.get("deepseek")
        if not USE_REAL_LLM or not api_key:
            return await self._run_blocking(lambda: {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None})
        def _call_deepseek():
            try:
                # Placeholder: implement real DeepSeek client
                return {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None}
            except Exception as e:
                logger.exception("DeepSeek request failed")
                return {"model": model_name, "text": "", "error": str(e), "raw": None}
        return await self._run_blocking(_call_deepseek)

# end of part 1/4
# === ai_alert.py (part 2/4) ===
# Continue MultiAnalystSystem (Grok), simulation, and Telegram helpers + data fetchers + pattern detector stubs

class MultiAnalystSystem(MultiAnalystSystem):  # extend previous definition to add missing methods
    async def ask_grok(self, prompt: str, timeout: int = None) -> Dict[str, Any]:
        model_name = "grok"
        timeout = self.timeout if timeout is None else timeout
        api_key = self.api_keys.get("grok")
        if not USE_REAL_LLM or not api_key:
            return await self._run_blocking(lambda: {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None})
        def _call_grok():
            try:
                # Placeholder: implement real Grok client if available
                return {"model": model_name, "text": self._simulate_response(prompt, model_name), "error": None, "raw": None}
            except Exception as e:
                logger.exception("Grok request failed")
                return {"model": model_name, "text": "", "error": str(e), "raw": None}
        return await self._run_blocking(_call_grok)

    def _simulate_response(self, prompt: str, model_name: str) -> str:
        """
        Deterministic simulated response generator.
        Parses RSI and Price from prompt (if present) and returns a compact signal text.
        """
        # simple parse
        rsi = None
        price = None
        for ln in prompt.splitlines():
            ln_s = ln.strip()
            if ln_s.lower().startswith("rsi"):
                # extract number
                for token in ln_s.replace(":", " ").split():
                    try:
                        v = float(token)
                        if 0 <= v <= 100:
                            rsi = v
                            break
                    except Exception:
                        continue
            if ln_s.lower().startswith("price"):
                for token in ln_s.replace(":", " ").split():
                    try:
                        p = float(token)
                        price = p
                        break
                    except Exception:
                        continue

        # rule-based signal
        signal = "NEUTRAL"
        confidence = 50
        if rsi is not None:
            if rsi < 35:
                signal = "BUY"
                confidence = min(95, int((40 - rsi) * 1.3 + 60))
            elif rsi > 65:
                signal = "SELL"
                confidence = min(95, int((rsi - 60) * 1.3 + 60))
            else:
                signal = "NEUTRAL"
                confidence = max(30, int(100 - abs(50 - rsi) * 1.0))
        # targets
        targets = "N/A"
        if price is not None:
            if signal == "BUY":
                targets = f"{round(price*1.01,2)},{round(price*1.02,2)}"
            elif signal == "SELL":
                targets = f"{round(price*0.99,2)},{round(price*0.98,2)}"
            else:
                targets = f"{round(price*1.005,2)},{round(price*0.995,2)}"
        text = f"Signal: {signal}\nConfidence: {confidence}%\nTargets: {targets}\nNote: simulated by {model_name}"
        return text

    async def analyze_all(self, market_data: Dict[str, Any], news: List[str], economic: Dict[str, Any], per_model_timeout: int = None) -> List[Dict[str, Any]]:
        """
        Call all 4 models in parallel (ask_openai, ask_gemini, ask_deepseek, ask_grok).
        Returns list of dicts with keys model,text,error,raw
        """
        per_model_timeout = self.timeout if per_model_timeout is None else per_model_timeout
        prompt = self._build_prompt(market_data, news, economic)
        logger.debug("analyze_all prompt preview: %s", prompt[:500])
        tasks = [
            asyncio.create_task(self.ask_openai(prompt, timeout=per_model_timeout)),
            asyncio.create_task(self.ask_gemini(prompt, timeout=per_model_timeout)),
            asyncio.create_task(self.ask_deepseek(prompt, timeout=per_model_timeout)),
            asyncio.create_task(self.ask_grok(prompt, timeout=per_model_timeout)),
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        results: List[Dict[str, Any]] = []
        for g in gathered:
            if isinstance(g, Exception):
                logger.exception("LLM task exception")
                results.append({"model": "unknown", "text": "", "error": str(g), "raw": None})
            else:
                results.append({
                    "model": g.get("model", "unknown"),
                    "text": (g.get("text") or "").strip(),
                    "error": g.get("error"),
                    "raw": g.get("raw"),
                })
        # store trimmed history
        ai_analysis_history.append({"ts": time.time(), "prompt": prompt[:2000], "results": results})
        if len(ai_analysis_history) > MAX_HISTORY:
            ai_analysis_history.pop(0)
        return results

    # format for telegram
    # (re-declare to ensure method exists if child overrides earlier)
    def format_results_for_telegram(self, results: List[Dict[str, Any]], market_data: Dict[str, Any], pattern_info: Optional[Dict[str, Any]] = None) -> str:
        symbol = market_data.get("symbol", "UNKNOWN")
        price = market_data.get("price", "N/A")
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        parts = [f"<b>AI Analysis</b> — {symbol}  Price: {price}\nTime: {ts}\n"]
        for r in results:
            model = (r.get("model") or "UNKNOWN").upper()
            error = r.get("error")
            text = r.get("text") or ""
            parts.append(f"<b>{model}:</b>")
            if error:
                parts.append(f"<i>Error:</i> {error}")
            else:
                snippet = text if len(text) <= 1200 else text[:1200] + "..."
                snippet = snippet.replace("<", "").replace(">", "")
                parts.append(snippet)
        if pattern_info:
            parts.append("<b>Detected pattern:</b>")
            parts.append(json.dumps(pattern_info, ensure_ascii=False, indent=2)[:1500])
        parts.append("\n-- End --")
        return "\n".join(parts)

# -----------------------
# Telegram helpers
# -----------------------
def send_telegram_message(bot_token: str, chat_id: str, text: str, parse_mode: str = "HTML", retries: int = 3) -> Dict[str, Any]:
    if not bot_token or not chat_id:
        logger.error("Missing BOT_TOKEN or CHAT_ID for Telegram send")
        return {"ok": False, "error": "missing_bot_or_chat"}
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    for attempt in range(1, retries + 1):
        try:
            logger.info("Telegram send attempt %d", attempt)
            r = requests.post(url, data=payload, timeout=15)
            try:
                data = r.json()
            except Exception:
                logger.warning("Telegram non-json response: %s", r.text)
                data = {"ok": False, "http_status": r.status_code, "raw": r.text}
            if r.status_code == 200 and data.get("ok"):
                logger.info("Telegram send ok")
                return data
            else:
                logger.warning("Telegram returned not-ok: %s", data)
                if r.status_code == 429:
                    wait = int(data.get("parameters", {}).get("retry_after", 5)) + 1 if isinstance(data, dict) else 5
                    logger.warning("Rate limited, sleeping %s", wait)
                    time.sleep(wait)
                else:
                    time.sleep(1)
        except Exception:
            logger.exception("Exception while sending Telegram message")
            time.sleep(1)
    return {"ok": False, "error": "failed_after_retries"}

def send_telegram_photo(bot_token: str, chat_id: str, caption: str, photo_bytes: bytes, filename: str = "chart.png", retries: int = 3) -> Dict[str, Any]:
    if not bot_token or not chat_id:
        logger.error("Missing BOT_TOKEN or CHAT_ID for Telegram photo")
        return {"ok": False, "error": "missing_bot_or_chat"}
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {"photo": (filename, photo_bytes, "image/png")}
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    for attempt in range(1, retries + 1):
        try:
            logger.info("Telegram photo attempt %d", attempt)
            r = requests.post(url, files=files, data=data, timeout=30)
            try:
                resp = r.json()
            except Exception:
                resp = {"ok": False, "http_status": r.status_code, "raw": r.text}
            if r.status_code == 200 and resp.get("ok"):
                logger.info("Telegram photo send ok")
                return resp
            else:
                logger.warning("Telegram photo returned not-ok: %s", resp)
                time.sleep(1)
        except Exception:
            logger.exception("Exception while sending Telegram photo")
            time.sleep(1)
    return {"ok": False, "error": "failed_after_retries"}

# -----------------------
# Minimal data fetchers (stubs)
# -----------------------
def get_shared_xau_data() -> Dict[str, Any]:
    """
    Replace with real fetcher. Return minimal dict with symbol, price, rsi, ema if possible.
    """
    return {"symbol": "XAUUSD", "price": 2050.25, "rsi": 48.3, "ema": 2047.1}

def get_market_news() -> List[str]:
    return ["Gold steadies ahead of US CPI data.", "Central bank buying supports demand."]

def get_economic_indicators() -> Dict[str, Any]:
    return {"US_CPI_mom": "0.2%", "US_unemployment": "3.7%"}

# -----------------------
# Pattern detection stubs
# -----------------------
class SimplePatternDetector:
    def __init__(self):
        pass
    def detect_pattern(self, df=None):
        return {"pattern": None, "confidence": 0}
    def predict_signals(self, df=None):
        return {"signal": "NEUTRAL", "reason": "stub"}

class AdvancedPatternDetector:
    def __init__(self):
        pass
    def detect_all_patterns(self, df=None):
        # return list of mock patterns
        return []
    def detect_chart_patterns(self, df=None):
        return []
    def predict_signals(self, df=None):
        return {"signal": "NEUTRAL", "confidence": 40}
# end of part 2/4

# === ai_alert.py (part 3/4) ===
# Plotting / chart generation stubs, explain_prediction, calc_targets, and additional helpers

# -----------------------
# Chart / plotting stubs
# -----------------------
def create_candlestick_chart(df=None, trading_signals=None, pattern_info=None) -> Optional[bytes]:
    """
    Placeholder: return PNG bytes of chart or None.
    To implement real charts, use matplotlib + mplfinance to draw candlesticks and save to BytesIO.
    """
    return None  # no chart generated in minimal implementation

def create_ai_enhanced_chart(df=None, consensus=None, pattern_info=None) -> Optional[bytes]:
    return create_candlestick_chart(df=df, trading_signals=consensus, pattern_info=pattern_info)

# -----------------------
# Explain & targets helpers (simple)
# -----------------------
def explain_prediction(model: str, x_vec, price: float, ema_val: float, rsi_val: float, pred_label: int) -> str:
    # Minimal explanation generator
    return f"Model {model}: predicted_label={pred_label}; price={price}, ema={ema_val}, rsi={rsi_val}"

def calc_targets(pred_label: int, price: float) -> Dict[str, float]:
    try:
        price = float(price)
    except Exception:
        return {}
    if pred_label == 1:  # BUY
        return {"target1": round(price * 1.01, 4), "target2": round(price * 1.02, 4)}
    elif pred_label == -1:  # SELL
        return {"target1": round(price * 0.99, 4), "target2": round(price * 0.98, 4)}
    else:
        return {"target1": round(price * 1.005, 4), "target2": round(price * 0.995, 4)}

# -----------------------
# Save report / batch utils (lightweight)
# -----------------------
def save_analysis_report(analysis_result: Dict[str, Any], filename: Optional[str] = None) -> str:
    if not filename:
        filename = f"analysis_{int(time.time())}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        return filename
    except Exception:
        logger.exception("Failed to save analysis report")
        return ""

def create_pattern_summary_table(results_dict: Dict[str, Any]) -> str:
    # Return simple text summary
    lines = []
    for k, v in results_dict.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

# -----------------------
# Validation / safety helpers
# -----------------------
class PatternDetectionError(Exception):
    pass

class DataValidationError(Exception):
    pass

def validate_input_data(df):
    # Minimal validation stub
    if df is None:
        raise DataValidationError("Dataframe is None")
    return True

# -----------------------
# Logging / persistence helpers
# -----------------------
def log_pattern_detection(all_patterns, telegram_sent=False):
    logger.info("Logging pattern detection: patterns=%s telegram_sent=%s", all_patterns, telegram_sent)
    # Could append to file/db; for now append to GLOBAL_LOG_BUFFER
    _append_log(f"Patterns: {all_patterns} telegram_sent={telegram_sent}")

# -----------------------
# Legacy send_telegram wrapper (used in some code paths)
# -----------------------
def send_telegram(bot_token: str, chat_id: str, message: str) -> int:
    # Use send_telegram_message and return message_id or -1
    resp = send_telegram_message(bot_token, chat_id, message)
    if resp.get("ok"):
        try:
            return resp.get("result", {}).get("message_id", 0)
        except Exception:
            return 0
    return -1

# -----------------------
# Functions that may be used by other modules but minimal here
# -----------------------
def detect_all_patterns_enhanced(df, symbol='UNKNOWN', timeframe='1H'):
    return {"patterns": [], "symbol": symbol, "timeframe": timeframe}

def create_pattern_points_info(pattern_info):
    return {"points": []}

def create_trading_strategy(pattern_name, pattern_info):
    # stub strategy
    return {"entry": "N/A", "stop": "N/A", "targets": []}

# end of part 3/4

# === ai_alert.py (part 4/4) ===
# High-level flows, Flask endpoints, CLI run, and instructions for assembling+creating PR

# Instantiate MultiAnalystSystem (by default simulated responses)
api_keys = {
    "openai": OPENAI_API_KEY,
    "gemini": GEMINI_API_KEY,
    "grok": GROK_API_KEY,
    "deepseek": DEEPSEEK_API_KEY,
}
multi_system = MultiAnalystSystem(api_keys=api_keys, executor_workers=6, timeout=_DEFAULT_TIMEOUT)

# Core flow to run one analysis and send to Telegram
async def run_ai_once_and_send(multi: MultiAnalystSystem, market_data: Optional[Dict[str, Any]] = None, news: Optional[List[str]] = None, economic: Optional[Dict[str, Any]] = None, send_image: bool = False) -> Dict[str, Any]:
    global _last_sent_hour, _message_sent_this_hour
    try:
        if market_data is None:
            market_data = get_shared_xau_data()
        if news is None:
            news = get_market_news()
        if economic is None:
            economic = get_economic_indicators()

        logger.info("Running AI analysis for %s", market_data.get("symbol", "UNKNOWN"))
        results = await multi.analyze_all(market_data, news, economic)
        logger.info("Analysis complete: models=%s", [r.get("model") for r in results])

        message_text = multi.format_results_for_telegram(results, market_data, pattern_info=None)

        # gating: avoid duplicates inside same UTC hour
        cur_hour = _now_hour_utc()
        msg_hash = str(hash(message_text))
        if _last_sent_hour != cur_hour:
            _message_sent_this_hour = {}
            _last_sent_hour = cur_hour

        if _message_sent_this_hour.get(msg_hash):
            logger.info("Message already sent this hour. Skipping.")
            return {"ok": False, "reason": "duplicate_this_hour"}

        # optionally generate chart (stub)
        image_bytes = None
        if send_image:
            image_bytes = create_ai_enhanced_chart(None, results, None)

        if image_bytes:
            resp = send_telegram_photo(BOT_TOKEN, CHAT_ID, caption=message_text, photo_bytes=image_bytes)
        else:
            resp = send_telegram_message(BOT_TOKEN, CHAT_ID, message_text)

        if resp.get("ok"):
            _message_sent_this_hour[msg_hash] = True
        else:
            logger.warning("Telegram responded not-ok: %s", resp)
        return resp
    except Exception:
        logger.exception("run_ai_once_and_send failed")
        return {"ok": False, "error": "exception", "trace": traceback.format_exc()}


# Flask endpoints
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/status", methods=["GET"])
def status():
    recent = GLOBAL_LOG_BUFFER[-40:]
    return jsonify({
        "bot_token_set": bool(BOT_TOKEN),
        "chat_id_set": bool(CHAT_ID),
        "last_sent_hour": _last_sent_hour,
        "message_sent_count_this_hour": len(_message_sent_this_hour),
        "recent_logs": recent,
        "history_len": len(ai_analysis_history),
    })

@app.route("/history", methods=["GET"])
def history():
    limit = int(request.args.get("limit", 50))
    data = ai_analysis_history[-limit:]
    sanitized = []
    for e in data:
        sanitized.append({
            "ts": e.get("ts"),
            "prompt_preview": (e.get("prompt") or "")[:1000],
            "results": e.get("results"),
        })
    return jsonify({"history": sanitized})

@app.route("/analyze", methods=["POST", "GET"])
def analyze_endpoint():
    payload = {}
    if request.method == "POST" and request.is_json:
        try:
            payload = request.get_json()
        except Exception:
            payload = {}
    market_data = payload.get("market_data")
    news = payload.get("news")
    economic = payload.get("economic")
    send_image = payload.get("send_image", False)

    def _bg_job():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run_ai_once_and_send(multi_system, market_data=market_data, news=news, economic=economic, send_image=send_image))
            logger.info("Background analyze job finished: %s", res)
        except Exception:
            logger.exception("Background job failed")

    t = threading.Thread(target=_bg_job, daemon=True)
    t.start()
    return jsonify({"status": "started", "send_image": send_image})


# CLI helper
def run_once_cli():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    resp = loop.run_until_complete(run_ai_once_and_send(multi_system))
    print("Result:", resp)

# Entrypoint
if __name__ == "__main__":
    run_mode = os.getenv("RUN_MODE", "flask").lower()
    if run_mode == "cli":
        logger.info("Running CLI one-shot")
        run_once_cli()
    else:
        if not BOT_TOKEN or not CHAT_ID:
            logger.warning("BOT_TOKEN or CHAT_ID not set. Telegram sending will fail or be skipped.")
        logger.info("Starting Flask on 0.0.0.0:5000")
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
# end of part 4/4
