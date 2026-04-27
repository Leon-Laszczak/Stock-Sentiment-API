import yfinance as yf
from yfinance.exceptions import YFRateLimitError

import requests
import dotenv
import os

import pandas as pd
import pandas_ta as ta
from math import tanh
from functools import lru_cache

import time

from app.core.helpers import _normalize
market_cache = {}
market_last_fetch = {}
yf_cache = {}
yf_last_fetch = {}

MIN_ROWS_FOR_LONG_EMA = 220
TECHNICAL_HISTORY_ROWS = 260
PATTERN_LOOKBACK_ROWS = 64
TIME_INTERVALS = ['1min','5min','15min','60min']
CACHE_TTL = 30
MAX_RETRIES = 3
FALLBACK_PERIOD_BY_INTERVAL = {
    '1m': '7d',
    '5m': '30d',
    '15m': '60d',
    '1h': '2y',
    '1d': '2y',
    '1wk': '5y',
}

def _alpha_vantage_to_yfinance_interval(interval: str) -> str:
    mapping = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '60min': '1h',
        '1d': '1d',
        '1wk': '1wk',
    }
    return mapping.get(interval, interval)

def _cache_key(ticker: str, period: str, interval: str) -> tuple[str, str, str]:
    return (ticker.upper(), period, interval)

def _trim_history(df: pd.DataFrame, keep_rows: int = TECHNICAL_HISTORY_ROWS) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.sort_index().tail(keep_rows).copy()

@lru_cache(maxsize=None)
def _require_env(key : str):
    dotenv.load_dotenv('.env')
    try:
        value = os.environ[key]
        return value
    except KeyError:
        return None
    
def _backfill_history_with_yfinance(ticker: str, df: pd.DataFrame, min_rows: int = TECHNICAL_HISTORY_ROWS,period = 'max', interval = '1d') -> pd.DataFrame:
    """Backfill premium series with free Yahoo history when Alpha Vantage is too short."""
    if len(df) >= min_rows:
        return _trim_history(df, keep_rows=max(min_rows, TECHNICAL_HISTORY_ROWS))

    try:
        yf_df = _fetch_stock_data(
            ticker=ticker,
            period=period,
            interval=interval,
            keep_rows=max(min_rows, TECHNICAL_HISTORY_ROWS),
        )
        if yf_df is None or yf_df.empty:
            return _trim_history(df, keep_rows=max(min_rows, TECHNICAL_HISTORY_ROWS))

        yf_df = yf_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        yf_df.index = pd.to_datetime(yf_df.index, errors='coerce')
        if getattr(yf_df.index, 'tz', None) is not None:
            yf_df.index = yf_df.index.tz_localize(None)

        yf_df = yf_df.apply(pd.to_numeric, errors='coerce')
        yf_df = yf_df.sort_index().dropna(subset=['Close'])

        merged = pd.concat([yf_df, df]).sort_index()
        merged = merged[~merged.index.duplicated(keep='last')]
        merged = merged.dropna(subset=['Close'])
        return _trim_history(merged, keep_rows=max(min_rows, TECHNICAL_HISTORY_ROWS))
    except Exception:
        return _trim_history(df, keep_rows=max(min_rows, TECHNICAL_HISTORY_ROWS))
    
def _fetch_stock_data(ticker, period="max", interval="1d", keep_rows: int = TECHNICAL_HISTORY_ROWS):
    now = time.time()
    key = _cache_key(ticker, period, interval)

    if key in yf_cache and key in yf_last_fetch:
        if now - yf_last_fetch[key] < CACHE_TTL:
            return yf_cache[key]

    for attempt in range(MAX_RETRIES):
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval)

            if data is None or data.empty:
                raise ValueError("Empty data")

            data = _trim_history(data, keep_rows=keep_rows)

            yf_cache[key] = data
            yf_last_fetch[key] = now

            return data

        except YFRateLimitError:
            print(f"[WARN] Rate limited (attempt {attempt+1})")
            time.sleep(2 * (attempt + 1))  # backoff

        except Exception as e:
            print(f"[ERROR] {e}")
            break

    if key in yf_cache:
        print("[INFO] Using cached fallback")
        return yf_cache[key]

    return {"error": "Data unavailable (rate limited or failed)"}
  
def fetch_stock_data(ticker):
    now = time.time()
    ticker_key = ticker.upper()

    if ticker_key in market_cache and ticker_key in market_last_fetch:
        if now - market_last_fetch[ticker_key] < CACHE_TTL:
            return market_cache[ticker_key]

    dfs = []
    alpha_vantage_api_key = _require_env('ALPHA_VANTAGE_API_KEY')
    try:
        response = requests.get(
            'https://www.alphavantage.co/query',
            params={
                'apikey': alpha_vantage_api_key,
                'symbol': ticker,
                'function': 'TIME_SERIES_DAILY',
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        response = payload.get('Time Series (Daily)')

        if not response:
            details = payload.get('Note') or payload.get('Error Message') or payload.get('Information')
            if details:
                raise RuntimeError(f'Alpha Vantage error: {details}')
            raise RuntimeError('Alpha Vantage did not return daily time series data.')

        df = pd.DataFrame.from_dict(response, orient='index')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume',
        })
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise RuntimeError(f'Missing expected Alpha Vantage columns: {missing_cols}')

        df = df[expected_cols].apply(pd.to_numeric, errors='coerce')
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index().dropna(subset=['Close'])
        df = _backfill_history_with_yfinance(
            ticker=ticker,
            df=df,
            min_rows=TECHNICAL_HISTORY_ROWS,
            period=FALLBACK_PERIOD_BY_INTERVAL['1d'],
            interval='1d',
        )
        dfs.append(df)
    except:
        dfs.append(
            _fetch_stock_data(
                ticker=ticker,
                period=FALLBACK_PERIOD_BY_INTERVAL['1d'],
                interval='1d',
            )
        )
    
    for interval in TIME_INTERVALS:
        yf_interval = _alpha_vantage_to_yfinance_interval(interval)
        fallback_period = FALLBACK_PERIOD_BY_INTERVAL.get(yf_interval, '60d')
        try:
            response = requests.get(
                'https://www.alphavantage.co/query',
                params={
                    'apikey': alpha_vantage_api_key,
                    'symbol': ticker,
                    'function': 'TIME_SERIES_INTRADAY',
                    'interval' : interval,
                    'outputsize' : 'full'
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            response = payload.get(f'Time Series ({interval})')

            if not response:
                details = payload.get('Note') or payload.get('Error Message') or payload.get('Information')
                if details:
                    raise RuntimeError(f'Alpha Vantage error for {interval}: {details}')
                raise RuntimeError(f'Alpha Vantage did not return intraday time series data for {interval}.')

            df = pd.DataFrame.from_dict(response, orient='index')
            df = df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
            })
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                raise RuntimeError(f'Missing expected Alpha Vantage columns: {missing_cols}')

            df = df[expected_cols].apply(pd.to_numeric, errors='coerce')
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index().dropna(subset=['Close'])

            df = _backfill_history_with_yfinance(
                ticker=ticker,
                df=df,
                min_rows=TECHNICAL_HISTORY_ROWS,
                period=fallback_period,
                interval=yf_interval,
            )
            dfs.append(df)
        except:
            dfs.append(
                _fetch_stock_data(
                    ticker=ticker,
                    period=fallback_period,
                    interval=yf_interval,
                )
            )
        
    try:
        response = requests.get(
            'https://www.alphavantage.co/query',
            params={
                'apikey': alpha_vantage_api_key,
                'symbol': ticker,
                'function': 'TIME_SERIES_WEEKLY',
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        response = payload.get('Weekly Time Series')

        if not response:
            details = payload.get('Note') or payload.get('Error Message') or payload.get('Information')
            if details:
                raise RuntimeError(f'Alpha Vantage error: {details}')
            raise RuntimeError('Alpha Vantage did not return daily time series data.')

        df = pd.DataFrame.from_dict(response, orient='index')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume',
        })
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise RuntimeError(f'Missing expected Alpha Vantage columns: {missing_cols}')

        df = df[expected_cols].apply(pd.to_numeric, errors='coerce')
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.sort_index().dropna(subset=['Close'])
        df = _backfill_history_with_yfinance(
            ticker=ticker,
            df=df,
            min_rows=TECHNICAL_HISTORY_ROWS,
            period=FALLBACK_PERIOD_BY_INTERVAL['1wk'],
            interval='1wk',
        )
        dfs.append(df)
    except:
        dfs.append(
            _fetch_stock_data(
                ticker=ticker,
                period=FALLBACK_PERIOD_BY_INTERVAL['1wk'],
                interval='1wk',
            )
        )
            
    market_cache[ticker_key] = dfs
    market_last_fetch[ticker_key] = now
    return dfs

def compute_technicals(df):
    df = _trim_history(df.copy())
    df['RSI'] = ta.rsi(close = df['Close'],length = 14)
    
    df['EMA_20'] = ta.ema(close = df['Close'],length = 20)
    df['EMA_50'] = ta.ema(close = df['Close'],length = 50)
    df['EMA_100'] = ta.ema(close = df['Close'],length = 100)
    df['EMA_200'] = ta.ema(close = df['Close'],length = 200)

    macd = ta.macd(close = df['Close'])
    if macd is None or macd.empty:
        raise RuntimeError('Unable to compute MACD for premium data.')
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd['MACD_12_26_9'], macd['MACDs_12_26_9'], macd['MACDh_12_26_9']

    bbands = ta.bbands(close=df['Close'], length=20)
    if bbands is None or bbands.empty:
        raise RuntimeError('Unable to compute Bollinger Bands for premium data.')

    bbl_col = next((c for c in bbands.columns if c.startswith('BBL_')), None)
    bbm_col = next((c for c in bbands.columns if c.startswith('BBM_')), None)
    bbu_col = next((c for c in bbands.columns if c.startswith('BBU_')), None)
    if not all([bbl_col, bbm_col, bbu_col]):
        raise RuntimeError('Unable to map Bollinger Bands columns from pandas_ta output.')

    df['BBL'] = bbands[bbl_col]
    df['BBM'] = bbands[bbm_col]
    df['BBU'] = bbands[bbu_col]

    df[['DL','DU']]=ta.donchian(df['High'],df['Low'])[['DCL_20_20','DCU_20_20']]

    df['ATR'] = ta.atr(df['High'],df['Low'],df['Close'])
    df['ADX'] = ta.adx(df['High'],df['Low'],df['Close'])['ADX_14']

    df['Stoch'] = ta.stoch(df['High'],df['Low'],df['Close'])['STOCHk_14_3_3']
    return df

def compute_technical(data):
    if isinstance(data, list):
        return [compute_technicals(df) if not isinstance(df, dict) else df for df in data]
    if isinstance(data, dict):
        return data
    return compute_technicals(data)

def analyze_patterns(df):
    try:
        pattern_frame = df[['Open', 'High', 'Low', 'Close']].tail(PATTERN_LOOKBACK_ROWS).copy()
        patterns = ta.cdl_pattern(
            pattern_frame['Open'],
            pattern_frame['High'],
            pattern_frame['Low'],
            pattern_frame['Close'],
        )
        if patterns is None or patterns.empty:
            return {
                'Score' : 0.0,
                'Bullish Patterns' : [],
                'Bearish Patterns' : []
            }

        latest_patterns = patterns.iloc[-1]
        bull = [name for name, value in latest_patterns.items() if value > 0]
        bear = [name for name, value in latest_patterns.items() if value < 0]

        curr_patterns = [float(value) for value in latest_patterns.values.tolist() if value != 0]
        score = sum(curr_patterns)/len(curr_patterns) if curr_patterns else 0.0

        return {
            'Score' : _normalize(score,-100,100,-1,1),
            'Bullish Patterns' : bull,
            'Bearish Patterns' : bear
        }
    except Exception as e:
        return {
            'Score' : 0.0,
            'Bullish Patterns' : [],
            'Bearish Patterns' : [],
            'error' : str(e)
        }


def analyze_resistance_and_support(df):
    import numpy as np
    from scipy.signal import argrelextrema
    from sklearn.cluster import KMeans

    n = 5 
    df['min'] = df.iloc[argrelextrema(df.Close.values, np.less_equal, order=n)[0]]['Close']
    df['max'] = df.iloc[argrelextrema(df.Close.values, np.greater_equal, order=n)[0]]['Close']

    supports = df.dropna(subset=['min'])['min'].values.reshape(-1,1)
    resistances = df.dropna(subset=['max'])['max'].values.reshape(-1,1)

    k = max(2, min(10, int(np.sqrt(len(supports)))))
    s_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    k = max(2, min(10, int(np.sqrt(len(resistances)))))
    r_kmeans = KMeans(k,random_state=42,n_init=10)

    s_kmeans.fit(supports)
    r_kmeans.fit(resistances)

    s_clusters = s_kmeans.cluster_centers_.flatten()
    s_clusters.sort()

    r_clusters = r_kmeans.cluster_centers_.flatten()
    r_clusters.sort()

    

def analyze_ema(price, ema_20, ema_50, ema_100, ema_200):
    """Score EMA structure, momentum, and crosses into a single signal."""

    score = 0
    try:
        if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1] > ema_200.iloc[-1]:
            score += 0.3
            trend = 'bullish'
            stack = 'bullish'

        elif ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1]:
            score += 0.05
            trend = 'bullish'
            stack = 'partially bullish'

        elif ema_20.iloc[-1] < ema_50.iloc[-1] < ema_100.iloc[-1] < ema_200.iloc[-1]:
            score -= 0.3
            trend = 'bearish'
            stack ='bearish'

        else:
            trend = 'neutral'
            stack = None

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

        if bullish_cross(ema_20, ema_50):
            cross_20_50 = 'bullish'
            score += 0.05
        elif bearish_cross(ema_20, ema_50):
            cross_20_50 = 'bearish'
            score -= 0.05
        else:
            cross_20_50 = None
        
        if bullish_cross(ema_50, ema_100):
            cross_50_100 = 'bullish'
            score += 0.1
        elif bearish_cross(ema_50, ema_100):
            cross_50_100 = 'bearish'
            score -= 0.1
        else: 
            cross_50_100 = None

        if bullish_cross(ema_100, ema_200):
            cross_100_200 = 'bullish'
            score += 0.15
        elif bearish_cross(ema_100, ema_200):
            cross_100_200 = 'bearish'
            score -= 0.15
        else: 
            cross_100_200 = None

        return {
        'Score' : tanh(score),
        'Stack' : stack,
        'Trend' : trend,
        'Momentum' : momentum,
        'EMA Crosses' : {
                'Cross 20 & 50' : cross_20_50 or None,
                'Cross 50 & 100' : cross_50_100 or None,
                'Cross 100 & 200' : cross_100_200 or None
                },
        'Details' : {
            'Current Price' : price.iloc[-1],
            'Current EMA20' : ema_20.iloc[-1],
            'Current EMA50' : ema_50.iloc[-1],
            'Current EMA100' : ema_100.iloc[-1],
            'Current EMA200' : ema_200.iloc[-1]
        }
            }
    except Exception as e:
        return {
            'Score' : tanh(score),
            'error' : str(e)
            }
    
def analyze_macd(macd,hist,signal):
    """Score MACD signals into a normalized sentiment contribution."""
    score = 0

    try:
        curr_macd = macd.iloc[-1]
        score += tanh(curr_macd/macd.std())

        delta_macd = curr_macd - macd.iloc[-30]
        score += tanh(delta_macd)

        z = (curr_macd - macd.iloc[-100:].mean())/macd.iloc[-100:].std() if macd.iloc[-100:].std() != 0 else 0
        score += -0.2 if z > 2 or z < -2 else 0
        
        t = -1
        macd_slope = macd.iloc[t] - macd.iloc[t-1]
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
        

        return {
            'Score' : tanh(score),
            'MACD Slope' : 'bullish' if macd_slope > 0 else 'bearish',
            'Signal Slope' : 'bullish' if signal_slope > 0 else 'bearish',
            'Histogram Slope' : 'bullish' if hist_slope > 0 else 'bearish',
            'MACD Z Score' : z,
            'Details' : {
                'Current MACD' : curr_macd,
                'Current Signal' : signal.iloc[t],
                'Current Histogram' : hist.iloc[t]
                }
            }
    except Exception as e:
        return {
            'Score' : tanh(score),
            'error' : str(e)
        }

def analyze_bbands(BBL,BBU,BBM,price):
    score = 0
    try:
        breakout_down = price.iloc[-1] < BBL.iloc[-1]
        breakout_up = price.iloc[-1] > BBU.iloc[-1]

        if breakout_down : breakout = 'down'
        elif breakout_up: breakout = 'up'
        else: breakout = None

        bandwith = (BBU-BBL)/BBM
        squezze = bandwith.iloc[-1] < bandwith[-100:].mean()*0.8

        if bandwith.iloc[-1] < 0.20:
            score += 0.15

        if squezze and breakout_down:
            score -= 0.25

        elif squezze and breakout_up:
            score += 0.25

        elif squezze:
            score += 0.05 #Potencial of going up

        elif not squezze and breakout_down:
            score += 0.1

        elif not squezze and breakout_up:
            score -= 0.1

        pct_bb = (price-BBL)/(BBU-BBL)

        if BBM.iloc[-5:].pct_change().mean() > 0 and pct_bb.iloc[-5:].mean() > 0.8:
            score += 0.2
        
        if BBM.iloc[-5:].pct_change().mean() < 0.007 and BBM.iloc[-5:].pct_change().mean() > -0.007:
            if pct_bb.iloc[-1] > 0.95:
                score -= 0.2
            elif pct_bb.iloc[-1] < 0.05:
                score += 0.2
        
        if pct_bb.iloc[-10:].pct_change().mean() > 0.02:
            score += 0.1
        if pct_bb.iloc[-10:].pct_change().mean() < -0.02:
            score -= 0.1
        
        return {
            'Score' : tanh(score),
            'Bandwith' : bandwith.iloc[-1],
            '%B' : pct_bb.iloc[-1],
            'Squezze' : bool(squezze),
            'Breakout' : breakout,
            'Details' : {
                'Current Price' : price.iloc[-1],
                'Current Upper Band' : BBU.iloc[-1],
                'Current Medium Band' : BBM.iloc[-1],
                'Current Lower Band' : BBL.iloc[-1]
            }    
                }
    except Exception as e:
        return {
            'Score' : tanh(score),
            'error' : str(e)
            }
    
def detect_overbought_peaks(rsi):
        if rsi.iloc[-1] < 70:
            return {
                'Score' : 0,
                'Days' : 0,
                'State' : None
                }
        
        slope = rsi.iloc[-1] - rsi.iloc[-2]
        state_of_peak = 'opening' if slope > 0 else 'closing'
        days_of_peak = 0

        for idx in range(1,len(rsi)):
            if rsi.iloc[-idx]>70:
                days_of_peak += 1
            else:
                break
        
        if days_of_peak<=3 and slope < 0:
            return {
                'Score' : -0.1,
                'Days' : days_of_peak,
                'State' : state_of_peak
                }
        elif days_of_peak>=5 and slope < 0:
            return {
                'Score' : -0.25,
                'Days' : days_of_peak,
                'State' : state_of_peak
                }
        else:
            return {
                'Score' : tanh(slope/10),
                'Days' : days_of_peak,
                'State' : state_of_peak
                }

def detect_oversold_troughs(rsi):
    if rsi.iloc[-1] > 30:
            return {
                'Score' : 0,
                'Days' : 0,
                'State' : None
                }
    slope = rsi.iloc[-1]- rsi.iloc[-2]

    state_of_trough = 'opening' if slope < 0 else 'closing'
    days_of_trough = 0

    for idx in range(1,len(rsi)):
        if rsi.iloc[-idx]<30:
            days_of_trough += 1
        else:
            break
    
    if days_of_trough<=3 and slope > 0:
        return {
            'Score' : 0.1,
            'Days' : days_of_trough,
            'State' : state_of_trough
            }
    elif days_of_trough>=5 and slope > 0:
        return {
            'Score' : 0.25,
            'Days' : days_of_trough,
            'State' : state_of_trough
            }
    else:
        return {
            'Score' : tanh(slope/10),
            'Days' : days_of_trough,
            'State' : state_of_trough
            }
    
def analyze_rsi(rsi_14):
    score = 0
         
    try:
        score -= _normalize(rsi_14.iloc[-1],30,70,-0.5,0.5)
        slope = rsi_14.iloc[-1] - rsi_14.iloc[-2]

        if rsi_14.iloc[-2] < rsi_14.iloc[-1]:
            if rsi_14.iloc[-1] < 45:
                score += 0.1
            elif rsi_14.iloc[-1]>70:
                score -= 0.05

        elif rsi_14.iloc[-2] > rsi_14.iloc[-1]:
            if rsi_14.iloc[-1] > 55:
                score -= 0.1
            elif rsi_14.iloc[-1] < 30:
                score += 0.05
        
        peaks = detect_overbought_peaks(rsi_14)
        troughs = detect_oversold_troughs(rsi_14)

        score += peaks['Score']
        score += troughs['Score']

        return {
            'Score' : tanh(score),
            'Slope' : 'bullish' if slope > 0 else 'bearish',
            'Overbought Peaks' : peaks if peaks['Score'] else None,
            'Oversold troughs' : troughs if troughs['Score'] else None,
            'Current RSI' : rsi_14.iloc[-1]
            }
    except Exception as e:
        return {
            'Score' : tanh(score),
            'error' : str(e)
            }

def analyze_donchian(lower, upper, price):
    score = 0
    try:
        if len(price) < 20 or len(lower) < 20 or len(upper) < 20:
            return score

        channel_range = upper - lower
        valid_range = channel_range.replace(0, pd.NA)
        midpoint = (upper + lower) / 2
        pct_in_channel = (price - lower) / valid_range

        breakout_up = price.iloc[-1] > upper.iloc[-2]
        breakout_down = price.iloc[-1] < lower.iloc[-2]

        if breakout_down : breakout = 'down'
        elif breakout_up: breakout = 'up'
        else: breakout = None

        if breakout_up:
            score += 0.35
        elif breakout_down:
            score -= 0.35

        if price.iloc[-1] > midpoint.iloc[-1]:
            score += 0.1
        elif price.iloc[-1] < midpoint.iloc[-1]:
            score -= 0.1

        if pct_in_channel.iloc[-5:].mean() > 0.75:
            score += 0.15
        elif pct_in_channel.iloc[-5:].mean() < 0.25:
            score -= 0.15

        range_strength = (channel_range.iloc[-1] / channel_range.iloc[-10:].mean()) if channel_range.iloc[-10:].mean() != 0 else 1
        if range_strength > 1.15:
            if price.iloc[-1] > midpoint.iloc[-1]:
                score += 0.1
            else:
                score -= 0.1

        flat_channel = abs(channel_range.iloc[-10:].pct_change().mean()) < 0.01
        if flat_channel:
            if pct_in_channel.iloc[-1] > 0.95:
                score -= 0.1
            elif pct_in_channel.iloc[-1] < 0.05:
                score += 0.1

        return {
            'Score' : tanh(score),
            'Breakout' : breakout,
            'Channel Range' : channel_range.iloc[-1],
            '% Position In Range' : pct_in_channel.iloc[-1],
            'Range Strength' : range_strength
            }
    except Exception as e:
        return {
            'Score' : tanh(score),
            'error' : str(e)
            }

def calculate_trend_strength(adx,volume):
    score = 0
    try:
        if len(volume) < 6 or len(adx) < 1:
            return score

        volume_mean = volume.iloc[-20:].mean()
        if pd.isna(volume_mean) or volume_mean <= 0:
            volume_mean = volume.mean()

        if not pd.isna(volume_mean) and volume_mean > 0:
            score += _normalize(volume.iloc[-1],volume_mean,1.5*volume_mean,0,1)
            score += tanh((volume.iloc[-1] - volume.iloc[-5]) / volume_mean)

        score += _normalize(adx.iloc[-1],20,40,0.1,2)
        return tanh(score)
    except:
        return tanh(score)
    
def analyze_trend(rsi,stochastic,adx,atr,volume,price,ema_100):
    score = 0
    rsi_up = False
    rsi_down = False

    try:
        if len(rsi) < 6:
            return {
                'Score' : tanh(score),
                'Trend Strength' : 0.0,
                'Direction' : 'neutral',
                'Flags': {
                    'overbought': False,
                    'oversold': False,
                    'high_volatility': False
                },
                'Details' : {
                    'Current RSI' : rsi.iloc[-1] if len(rsi) else None,
                    'Current Stochastic' : None,
                    'Current ADX' : adx.iloc[-1] if len(adx) else None,
                    'Current ATR' : atr.iloc[-1] if len(atr) else None,
                    'Current Volume' : volume.iloc[-1] if len(volume) else None
                }
            }

        trend_strength = calculate_trend_strength(adx,volume)
        k_now = None
        atr_now = None
        atr_ratio = None

        if trend_strength <= 0:
            trend_strength = 0.1

        if rsi.iloc[-5] < rsi.iloc[-1] and 45 < rsi.iloc[-1] < 70:
            rsi_up = True
            score += trend_strength * _normalize(rsi.iloc[-1],45,65,0,1)

        elif rsi.iloc[-5] > rsi.iloc[-1] and 55 > rsi.iloc[-1] > 30:
            rsi_down = True
            score -= trend_strength * _normalize(rsi.iloc[-1],35,55,0,1)

        stoch_k = stochastic
        stoch_d = None

        if isinstance(stoch_k,pd.Series) and len(stoch_k) >= 6:
            k_now = stoch_k.iloc[-1]
            k_prev = stoch_k.iloc[-5]

            if not pd.isna(k_now) and not pd.isna(k_prev):
                if k_now < 20 and k_now > k_prev:
                    score += 0.2 * trend_strength
                elif k_now > 80 and k_now < k_prev:
                    score -= 0.2 * trend_strength

                score += 0.1 * tanh((k_now - 50) / 15)

        if isinstance(stoch_k,pd.Series) and isinstance(stoch_d,pd.Series) and len(stoch_k) >= 2 and len(stoch_d) >= 2:
            prev_k, prev_d = stoch_k.iloc[-2], stoch_d.iloc[-2]
            curr_k, curr_d = stoch_k.iloc[-1], stoch_d.iloc[-1]

            if not pd.isna(prev_k) and not pd.isna(prev_d) and not pd.isna(curr_k) and not pd.isna(curr_d):
                if prev_k < prev_d and curr_k > curr_d and curr_k < 50:
                    score += 0.15 * trend_strength
                elif prev_k > prev_d and curr_k < curr_d and curr_k > 50:
                    score -= 0.15 * trend_strength

        if isinstance(atr,pd.Series) and len(atr) >= 20:
            atr_now = atr.iloc[-1]
            atr_mean = atr.iloc[-20:].mean()

            if not pd.isna(atr_now) and not pd.isna(atr_mean) and atr_mean > 0:
                atr_ratio = atr_now / atr_mean

                if atr_ratio > 1.15:
                    if rsi_up:
                        score += 0.12 * trend_strength
                    elif rsi_down:
                        score -= 0.12 * trend_strength
                elif atr_ratio < 0.85:
                    if rsi_up:
                        score -= 0.08 * trend_strength
                    elif rsi_down:
                        score += 0.08 * trend_strength

        if adx.iloc[-1] > 25:
            if price.iloc[-1] < ema_100.iloc[-1]:
                trend_direction = "bearish"
            else:
                trend_direction = "bullish"
        else:
            trend_direction = "neutral"
        return {
            'Score' : tanh(score),
            'Trend Strength' : trend_strength,
            'Direction' : trend_direction,
            'Flags': {
                'overbought': bool(rsi.iloc[-1] > 70),
                'oversold': bool(rsi.iloc[-1] < 30),
                'high_volatility': bool(atr_ratio is not None and atr_ratio > 1.2)
                },
            'Details' : {
                'Current RSI' : rsi.iloc[-1],
                'Current Stochastic' : k_now,
                'Current ADX' : adx.iloc[-1],
                'Current ATR' : atr_now,
                'Current Volume' : volume.iloc[-1]
                }
            }
    except Exception as e:
        return {
            'Score' : tanh(score),
            'error' : str(e)
            }
