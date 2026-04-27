import yfinance as yf

from app.data.fundamentals import *
from app.data.market import *

def _build_pro_interval(interval_name, df, include_patterns: bool = True):
    if df is None or df.empty:
        return None, f'{interval_name}: missing market data'

    components = {
        'RSI': analyze_rsi(df['RSI']),
        'MACD': analyze_macd(df['MACD'], df['MACD_Hist'], df['MACD_Signal']),
        'EMA': analyze_ema(df['Close'], df['EMA_20'], df['EMA_50'], df['EMA_100'], df['EMA_200']),
        'Bollinger Bands': analyze_bbands(df['BBL'], df['BBU'], df['BBM'], df['Close']),
        'Donchian Channels': analyze_donchian(df['DL'], df['DU'], df['Close']),
        'Trend': analyze_trend(df['RSI'], df['Stoch'], df['ADX'], df['ATR'], df['Volume'],df['Close'],df['EMA_100']),
    }

    weights = {
        'EMA': 0.25,
        'MACD': 0.2,
        'RSI': 0.15,
        'Trend': 0.15,
        'Bollinger Bands': 0.1,
        'Donchian Channels': 0.05,
    }

    if include_patterns:
        components['Patterns'] = analyze_patterns(df)
        weights['Patterns'] = 0.1

    for component_name, component_payload in components.items():
        if isinstance(component_payload, dict):
            error = component_payload.get('error')
            if error:
                return None, f'{interval_name} {component_name}: {error}'
        else:
            return None, f'{interval_name} {component_name}: invalid payload type {type(component_payload).__name__}'

    interval_score = sum(
        components[name]['Score'] * weights[name]
        for name in components
    )/sum(weights[name] for name in components)

    return {
        'Score': interval_score,
        **components,
    }, None

def technical_score(dfs):
    if len(dfs) != 6:
        return {
            'Score': 0.0,
            'error': f'Expected 6 dataframes for scoring, got {len(dfs)}'
        }

    interval_specs = [
        ('1 day', dfs[0], 8, True),
        ('1 min', dfs[1], 1, False),
        ('5 min', dfs[2], 2, False),
        ('15 min', dfs[3], 3, False),
        ('1 hour', dfs[4], 5, False),
        ('1 week', dfs[5], 13, True),
    ]

    intervals = {}
    weighted_score = 0.0
    total_weight = 0.0

    try:
        for interval_name, df, weight, include_patterns in interval_specs:
            if isinstance(df, dict) and df.get('error'):
                return {
                    'Score': 0.0,
                    'error': f"{interval_name}: {df['error']}",
                }

            interval_payload, error = _build_pro_interval(interval_name, df, include_patterns=include_patterns)
            if error:
                return {
                    'Score': 0.0,
                    'error': error,
                }

            intervals[interval_name] = interval_payload
            weighted_score += weight * interval_payload['Score']
            total_weight += weight

        return {
            'Score': weighted_score / total_weight if total_weight else 0.0,
            'Intervals': intervals,
        }
    except Exception as exc:
        return {
            'Score': 0.0,
            'error': str(exc),
        }

def fundamental_score(ticker: str, ticker_obj: yf.Ticker | None = None):
    try:
        t = ticker_obj or yf.Ticker(ticker)

        revenue = analyze_revenue(t)
        income = analyze_income_and_margins(t)
        eps = analyze_eps(t)
        predictions = analyze_predictions(t, ticker)

        score = (revenue + income + eps + predictions) / 4

        return {
            'Score': score,
            'Score Breakdown': {
                'Revenue Growth': revenue,
                'Income and Margins': income,
                'EPS Growth': eps,
                'Analyst Predictions': predictions
            }
        }

    except Exception as e:
        print(f"[FUND ERROR] {e}")
        return {
            'Score': 0,
            'Score Breakdown': {},
            'status': 'unavailable'
        }

def get_final_score(tech,news,fund):
    """Combine component scores using configurable weights."""
    return (
        0.3 * tech +
        0.3 * news +
        0.4 * fund
    )
