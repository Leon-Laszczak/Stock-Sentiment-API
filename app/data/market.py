import yfinance as yf
from yfinance.exceptions import YFRateLimitError

import pandas as pd
import pandas_ta as ta
from math import tanh

import time

from app.core.helpers import _normalize
cache = {}

last_fetch = {}

CACHE_TTL = 30
MAX_RETRIES = 3


def fetch_stock_data(ticker, period="max", interval="1d"):
    now = time.time()

    if ticker in cache and ticker in last_fetch:
        if now - last_fetch[ticker] < CACHE_TTL:
            return cache[ticker]

    for attempt in range(MAX_RETRIES):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval)

            if data is None or data.empty:
                raise ValueError("Empty data")

            cache[ticker] = data
            last_fetch[ticker] = now

            return data

        except YFRateLimitError:
            print(f"[WARN] Rate limited (attempt {attempt+1})")
            time.sleep(2 * (attempt + 1))  # backoff

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    if ticker in cache:
        print("[INFO] Using cached fallback")
        return cache[ticker]

    return {"error": "Data unavailable (rate limited or failed)"}

def compute_technical(df):
    """Compute RSI, EMA, and MACD columns in the given price dataframe."""
    df['RSI'] = ta.rsi(close = df['Close'],length = 14)
    
    df['EMA_20'] = ta.ema(close = df['Close'],length = 20)
    df['EMA_50'] = ta.ema(close = df['Close'],length = 50)
    df['EMA_100'] = ta.ema(close = df['Close'],length = 100)
    df['EMA_200'] = ta.ema(close = df['Close'],length = 200)

    macd = ta.macd(close = df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9'], macd['MACDh_12_26_9']

    return df

def analyze_macd(macd,hist,signal):
    """Score MACD signals into a normalized sentiment contribution."""
    score = 0

    curr_macd = macd.iloc[-1]
    score += tanh(curr_macd/macd.std())

    delta_macd = curr_macd - macd.iloc[-30]
    score += tanh(delta_macd)

    z = (curr_macd - macd.iloc[-100:].mean())/macd.iloc[-100:].std() if macd.iloc[-100:].std() != 0 else 0
    score += -0.2 if z > 2 or z < -2 else 0
    
    t = -1
    hist_slope = hist.iloc[t] - hist.iloc[t-1]
    signal_slope = signal.iloc[t] - signal.iloc[t-1]

    if hist.iloc[t] > 0 and hist_slope > 0 and signal_slope > 0:
        score += 0.35
    elif hist.iloc[t] > 0 and hist_slope < 0 and signal_slope > 0:
        score += 0.10
    elif hist.iloc[t] < 0 and hist_slope < 0 and signal_slope < 0:
        score -= 0.35
    elif hist.iloc[t] < 0 and hist_slope < 0 and signal_slope > 0:
        score -= 0.10
    

    return tanh(score)

def analyze_ema(price, ema_20, ema_50, ema_100, ema_200):
    """Score EMA structure, momentum, and crosses into a single signal."""
    score = 0
    if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1] > ema_200.iloc[-1]:
        score += 0.3
        trend = 'bullish'

    elif ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1]:
        score += 0.05
        trend = 'bullish'

    elif ema_20.iloc[-1] < ema_50.iloc[-1] < ema_100.iloc[-1] < ema_200.iloc[-1]:
        score -= 0.3
        trend = 'bearish'

    else:
        trend = 'neutral'

    if price.iloc[-1] > ema_50.iloc[-1] and price.iloc[-1] > ema_100.iloc[-1]:
        score += 0.15
    elif price.iloc[-1] < ema_50.iloc[-1] and price.iloc[-1] < ema_100.iloc[-1]:
        score -= 0.15

    slope_1w = ema_50.iloc[-1] - ema_50.iloc[-5]

    if slope_1w > 0:
        score += 0.1
        momentum = "bullish"
    else:
        score -= 0.1
        momentum = "bearish"
    distance = (price.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]

    if distance > 0.05:
        score -= 0.05
    elif distance < -0.05:
        score += 0.05

    def bullish_cross(fast, slow):
        if len(fast) < 2 or len(slow) < 2:
            return False
        prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
        curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
        if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
            return False
        return prev_fast < prev_slow and curr_fast > curr_slow

    def bearish_cross(fast, slow):
        if len(fast) < 2 or len(slow) < 2:
            return False
        prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
        curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
        if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
            return False
        return prev_fast > prev_slow and curr_fast < curr_slow

    if bullish_cross(ema_20, ema_50) and trend == "bullish" and momentum == "bullish":
        score += 0.1
    elif bearish_cross(ema_20, ema_50) and trend == "bearish" and momentum == "bearish":
        score -= 0.1

    if bullish_cross(ema_50, ema_100) and trend == "bullish" and momentum == "bullish":
        score += 0.15
    elif bearish_cross(ema_50, ema_100) and trend == "bearish" and momentum == "bearish":
        score -= 0.15

    if bullish_cross(ema_100, ema_200) and price.iloc[-1] > ema_200.iloc[-1]:
        score += 0.2
    elif bearish_cross(ema_100, ema_200) and price.iloc[-1] < ema_200.iloc[-1]:
        score -= 0.2

    return tanh(score)

def technical_score(df):
    """Compute a technical-only sentiment score from indicator columns."""
    rsi = -_normalize(df['RSI'],30,70,-1,1).iloc[-1]   

    macd = analyze_macd(df['MACD'], df['MACD_Hist'], df['MACD_Signal'])

    ema = analyze_ema(df['Close'],df['EMA_20'],df['EMA_50'],df['EMA_100'],df['EMA_200'])

    return {
        'Score' : (macd+rsi+ema)/3,
        'RSI' : rsi,
        'MACD' : macd,
        'EMA' : ema
                } 
