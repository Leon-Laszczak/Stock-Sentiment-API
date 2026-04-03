import yfinance as yf

from app.data.fundamentals import *
from app.data.market import *
from app.core.helpers import _normalize

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

def fundamental_score(ticker: str):
    try:
        t = yf.Ticker(ticker)

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