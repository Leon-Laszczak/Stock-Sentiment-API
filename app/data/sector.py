import yfinance as yf
import pandas_ta as ta
import pandas as pd
import requests
import time
from math import tanh

from app.data.news import score_news_for_ticker
from app.core.helpers import _require_env

SECTOR_ETF = {
    "technology": "XLK",
    "energy": "XLE",
    "financial": "XLF",
    "healthcare": "XLV",
    "industrial": "XLI",
    "consumer_cyclical": "XLY",
    "consumer_defensive": "XLP",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "materials": "XLB",
    "communication": "XLC"
}

NORMALIZE = {
    "Technology": "technology",
    "Financial Services": "financial",
    "Healthcare": "healthcare",
    "Energy": "energy",
    "Industrials": "industrial",
    "Consumer Cyclical": "consumer_cyclical",
    "Consumer Defensive": "consumer_defensive",
    "Utilities": "utilities",
    "Real Estate": "real_estate",
    "Materials": "materials",
    "Communication Services": "communication"
}

ALPHA_VANTAGE_SECTOR_MAP = {
    "technology": "Information Technology",
    "financial": "Financials",
    "healthcare": "Health Care",
    "energy": "Energy",
    "industrial": "Industrials",
    "consumer_cyclical": "Consumer Discretionary",
    "consumer_defensive": "Consumer Staples",
    "utilities": "Utilities",
    "real_estate": "Real Estate",
    "materials": "Materials",
    "communication": "Communication Services"
}

def get_ticker_sector(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return NORMALIZE.get(info.get('sector'), None)
    except Exception:
        return None


def _get_close_series(data):
    """Return a 1-D close-price series from yfinance output."""
    if data is None or data.empty:
        return None

    close = data.get('Close')
    if close is None:
        return None

    if isinstance(close, pd.DataFrame):
        if close.empty:
            return None
        close = close.iloc[:, 0]

    close = close.dropna()
    if close.empty:
        return None

    return close


def sector_tech(sector, period='1y'):
    if not sector:
        return 0.0

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return 0.0

    data = yf.download(etf, period=period, progress=False)
    close = _get_close_series(data)

    if close is None or len(close) < 200:
        return 0.0
    
    ema_50 = ta.ema(close, 50)
    ema_200 = ta.ema(close, 200)
    if ema_50 is None or ema_200 is None:
        return 0.0

    last_50 = ema_50.iloc[-1]
    last_200 = ema_200.iloc[-1]
    if pd.isna(last_50) or pd.isna(last_200):
        return 0.0

    if last_50 > last_200:
        return 0.4
    else:
        return -0.4


def relative_strength(
    sector,
    benchmark='^GSPC',
    period='3mo',
):
    if not sector:
        return 0.0

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return 0.0

    s = yf.download(etf, period=period, progress=False)
    b = yf.download(benchmark, period=period, progress=False)
    s_close = _get_close_series(s)
    b_close = _get_close_series(b)

    if s_close is None or b_close is None or len(s_close) < 2 or len(b_close) < 2:
        return 0.0

    s_start, s_end = s_close.iloc[0], s_close.iloc[-1]
    b_start, b_end = b_close.iloc[0], b_close.iloc[-1]
    if b_start == 0 or s_start == 0:
        return 0.0

    rs = (s_end / s_start) / (b_end / b_start)
    if pd.isna(rs):
        return 0.0

    if rs > 1.05:
        return 0.3
    elif rs < 0.95:
        return -0.3
    else:
        return 0.0

def sector_news(sector):
    if not sector:
        return {'score': 0.0, 'has_data': False, 'article_count': 0}

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return {'score': 0.0, 'has_data': False, 'article_count': 0}

    return score_news_for_ticker(etf, data_mode='fast')

def score_sector(ticker, sector=None):
    try:
        sector = sector if sector is not None else get_ticker_sector(ticker)

        tech = sector_tech(sector)
        rs = relative_strength(sector)
        news = sector_news(sector)

        score = tech + rs + news['score']

        return {
            'Score' : tanh(score),
            'Trend' : tech,
            'Relative Strength' : rs,
            'News' : news['score']
                }
    except Exception as e:
        return {
            'Score' : 0,
            'Trend' : 0,
            'Relative Strength' : 0,
            'News' : 0,
            'error' : str(e)
        }


#======================================
#========Premium Features==============
#======================================

_SECTOR_PERF_CACHE = {'timestamp': 0.0, 'payload': None}


def _fetch_premium_history(ticker: str, period: str = '2y') -> pd.DataFrame:
    try:
        from app.data.market import fetch_stock_data_premium
        df = fetch_stock_data_premium(ticker)
        if df is None or df.empty:
            raise RuntimeError('Missing premium history.')
        return df
    except Exception:
        return yf.download(ticker, period=period, progress=False)


def _get_close_premium(ticker: str, period: str = '2y'):
    data = _fetch_premium_history(ticker, period=period)
    return _get_close_series(data)


def _align_series(left: pd.Series, right: pd.Series):
    if left is None or right is None:
        return None, None
    df = pd.concat([left, right], axis=1, join='inner').dropna()
    if df.empty:
        return None, None
    return df.iloc[:, 0], df.iloc[:, 1]


def _window_return(series: pd.Series, window: int) -> float | None:
    if series is None or len(series) < window + 1:
        return None
    start = series.iloc[-window-1]
    end = series.iloc[-1]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return None
    return (end / start) - 1


def sector_trend_premium(sector, period='2y'):
    if not sector:
        return 0.0

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return 0.0

    close = _get_close_premium(etf, period=period)
    if close is None or len(close) < 220:
        return 0.0

    ema_50 = ta.ema(close, 50)
    ema_200 = ta.ema(close, 200)
    if ema_50 is None or ema_200 is None:
        return 0.0

    ema_50 = ema_50.dropna()
    ema_200 = ema_200.dropna()
    if ema_50.empty or ema_200.empty or len(ema_50) < 20:
        return 0.0

    last_50 = ema_50.iloc[-1]
    last_200 = ema_200.iloc[-1]
    price = close.iloc[-1]
    if pd.isna(last_50) or pd.isna(last_200) or pd.isna(price):
        return 0.0

    score = 0
    if last_50 > last_200:
        score += 0.3
    else:
        score -= 0.3

    if price > last_50:
        score += 0.15
    else:
        score -= 0.15

    slope_1m = last_50 - ema_50.iloc[-20]
    if slope_1m > 0:
        score += 0.1
    else:
        score -= 0.1

    if last_50 != 0:
        distance = (price - last_50) / last_50
        if distance > 0.08:
            score -= 0.1
        elif distance < -0.08:
            score += 0.1

    return tanh(score)


def sector_momentum_premium(sector):
    if not sector:
        return 0.0

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return 0.0

    close = _get_close_premium(etf)
    if close is None or len(close) < 60:
        return 0.0

    windows = {21: 0.35, 63: 0.3, 126: 0.2, 252: 0.15}
    weighted = 0
    total = 0

    for window, weight in windows.items():
        ret = _window_return(close, window)
        if ret is None:
            continue
        weighted += weight * ret
        total += weight

    if total == 0:
        return 0.0

    momentum = weighted / total
    return tanh(momentum * 3)


def sector_relative_strength_premium(
    sector,
    benchmark='SPY',
):
    if not sector:
        return 0.0

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return 0.0

    s_close = _get_close_premium(etf)
    b_close = _get_close_premium(benchmark)
    s_close, b_close = _align_series(s_close, b_close)
    if s_close is None or b_close is None or len(s_close) < 60:
        return 0.0

    windows = {21: 0.35, 63: 0.3, 126: 0.2, 252: 0.15}
    weighted = 0
    total = 0

    for window, weight in windows.items():
        s_ret = _window_return(s_close, window)
        b_ret = _window_return(b_close, window)
        if s_ret is None or b_ret is None:
            continue
        weighted += weight * (s_ret - b_ret)
        total += weight

    if total == 0:
        return 0.0

    excess = weighted / total
    return tanh(excess * 3)


def sector_volatility_premium(
    sector,
    benchmark='SPY',
    window=60,
):
    if not sector:
        return 0.0

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return 0.0

    s_close = _get_close_premium(etf)
    b_close = _get_close_premium(benchmark)
    s_close, b_close = _align_series(s_close, b_close)
    if s_close is None or b_close is None or len(s_close) < window + 2:
        return 0.0

    s_ret = s_close.pct_change().dropna()
    b_ret = b_close.pct_change().dropna()
    if len(s_ret) < window or len(b_ret) < window:
        return 0.0

    s_vol = s_ret.iloc[-window:].std()
    b_vol = b_ret.iloc[-window:].std()
    if pd.isna(s_vol) or pd.isna(b_vol) or b_vol == 0:
        return 0.0

    vol_ratio = s_vol / b_vol
    score = 0

    if vol_ratio < 0.85:
        score += 0.2
    elif vol_ratio > 1.25:
        score -= 0.2

    short_vol = s_ret.iloc[-20:].std() if len(s_ret) >= 20 else s_vol
    long_vol = s_ret.iloc[-60:].std() if len(s_ret) >= 60 else s_vol
    if not pd.isna(short_vol) and not pd.isna(long_vol) and long_vol != 0:
        if short_vol > long_vol * 1.2:
            score -= 0.1
        elif short_vol < long_vol * 0.8:
            score += 0.1

    s_window_ret = _window_return(s_close, min(window, 63))
    b_window_ret = _window_return(b_close, min(window, 63))
    if s_window_ret is not None and b_window_ret is not None:
        if s_window_ret < b_window_ret and vol_ratio > 1:
            score -= 0.1
        elif s_window_ret > b_window_ret and vol_ratio < 1:
            score += 0.1

    return tanh(score)


def _parse_percent(value):
    if value is None:
        return None
    try:
        if isinstance(value, str):
            cleaned = value.strip()
            has_pct = '%' in cleaned
            cleaned = cleaned.replace('%', '')
            num = float(cleaned)
            return num / 100.0 if has_pct else num
        num = float(value)
        if abs(num) > 1:
            return num / 100.0
        return num
    except Exception:
        return None


def _fetch_sector_performance_payload(ttl_seconds: int = 900):
    now = time.time()
    cached = _SECTOR_PERF_CACHE.get('payload')
    cached_ts = _SECTOR_PERF_CACHE.get('timestamp', 0)
    if cached and now - cached_ts < ttl_seconds:
        return cached

    response = requests.get(
        'https://www.alphavantage.co/query',
        params={
            'function': 'SECTOR',
            'apikey': _require_env('ALPHA_VANTAGE_API_KEY'),
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise RuntimeError('Alpha Vantage did not return sector data.')

    err = payload.get('Error Message') or payload.get('Note') or payload.get('Information')
    if err:
        raise RuntimeError(f'Alpha Vantage error: {err}')

    _SECTOR_PERF_CACHE['payload'] = payload
    _SECTOR_PERF_CACHE['timestamp'] = now
    return payload


def _parse_sector_table(payload: dict, key: str) -> dict:
    table = payload.get(key)
    if not isinstance(table, dict):
        return {}
    out = {}
    for sector_name, value in table.items():
        parsed = _parse_percent(value)
        if parsed is None:
            continue
        out[str(sector_name).strip()] = parsed
    return out


def _sector_percentile(table: dict, sector_name: str) -> float | None:
    if not table or sector_name not in table:
        return None
    items = [(k, v) for k, v in table.items() if isinstance(v, (int, float))]
    if not items:
        return None
    items.sort(key=lambda row: row[1], reverse=True)
    names = [name for name, _ in items]
    if sector_name not in names:
        return None
    n = len(names)
    if n == 1:
        return 0.5
    rank = names.index(sector_name)
    return (n - 1 - rank) / (n - 1)


def sector_rotation_premium(sector):
    if not sector:
        return 0.0

    sector_name = ALPHA_VANTAGE_SECTOR_MAP.get(sector.lower())
    if not sector_name:
        return 0.0

    try:
        payload = _fetch_sector_performance_payload()
    except Exception:
        return 0.0

    weights = {
        'Rank B: 1 Day Performance': 0.1,
        'Rank C: 5 Day Performance': 0.15,
        'Rank D: 1 Month Performance': 0.2,
        'Rank E: 3 Month Performance': 0.2,
        'Rank F: Year-to-Date (YTD) Performance': 0.15,
        'Rank G: 1 Year Performance': 0.2,
    }

    total = 0
    weighted = 0

    for key, weight in weights.items():
        table = _parse_sector_table(payload, key)
        percentile = _sector_percentile(table, sector_name)
        if percentile is None:
            continue
        score = (percentile * 2) - 1
        weighted += score * weight
        total += weight

    if total == 0:
        return 0.0

    return tanh((weighted / total) * 1.2)


def sector_news_premium(sector):
    if not sector:
        return {'score': 0.0, 'has_data': False, 'article_count': 0}

    etf = SECTOR_ETF.get(sector.lower())
    if not etf:
        return {'score': 0.0, 'has_data': False, 'article_count': 0}

    return score_news_for_ticker(etf, data_mode='fast')


def score_sector_premium(ticker, sector=None):
    try:
        sector = sector if sector is not None else get_ticker_sector(ticker)

        trend = sector_trend_premium(sector)
        momentum = sector_momentum_premium(sector)
        relative_strength = sector_relative_strength_premium(sector)
        volatility = sector_volatility_premium(sector)
        rotation = sector_rotation_premium(sector)
        news = sector_news_premium(sector)

        score = trend + momentum + relative_strength + volatility + rotation + news['score']

        return {
            'Score' : score/6,
            'Trend' : trend,
            'Momentum' : momentum,
            'Relative Strength' : relative_strength,
            'Volatility' : volatility,
            'Rotation' : rotation,
            'News' : news['score']
                }
    except Exception as e:
        return {
            'Score' : 0,
            'Trend' : 0,
            'Momentum' : 0,
            'Relative Strength' : 0,
            'Volatility' : 0,
            'Rotation' : 0,
            'News' : 0,
            'error' : str(e)
        }
