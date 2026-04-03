import yfinance as yf
import pandas as pd
import time
from math import tanh
from yfinance.exceptions import YFRateLimitError

# ===== CACHE =====
fund_cache = {}
fund_last_fetch = {}
FUND_TTL = 60 * 60 * 24 # 1 day


# ===== HELPERS =====
def _get_row(df: pd.DataFrame, label: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if label not in df.index:
        return None
    series = df.loc[label]
    if isinstance(series, pd.Series):
        return series.sort_index()
    return None


def safe_recommendations(ticker_obj, ticker):
    now = time.time()

    if ticker in fund_cache and ticker in fund_last_fetch:
        if now - fund_last_fetch[ticker] < FUND_TTL:
            return fund_cache[ticker]

    try:
        data = ticker_obj.recommendations
        fund_cache[ticker] = data
        fund_last_fetch[ticker] = now
        return data

    except YFRateLimitError:
        print("[WARN] recommendations rate limited")

    except Exception as e:
        print(f"[ERROR] recommendations: {e}")

    return fund_cache.get(ticker, None)


def safe_price_targets(ticker_obj):
    try:
        return ticker_obj.analyst_price_targets
    except Exception as e:
        print(f"[ERROR] price targets: {e}")
        return None


# ===== ANALYSIS =====
def analyze_revenue(t: yf.Ticker):
    try:
        score = 0
        revenue = _get_row(t.financials, 'Total Revenue')

        if revenue is None or revenue.size < 2:
            return 0

        growth = revenue.pct_change().iloc[-1]

        if growth > 0.15:
            score += 0.25
        elif growth > 0.05:
            score += 0.1
        elif growth < 0:
            score -= 0.25

        time.sleep(0.05)
        return tanh(score)

    except Exception:
        return 0


def analyze_income_and_margins(t: yf.Ticker):
    try:
        score = 0
        df = t.financials

        net_income = _get_row(df, 'Net Income')
        gross_profit = _get_row(df, 'Gross Profit')
        operating_income = _get_row(df, 'Operating Income')
        revenue = _get_row(df, 'Total Revenue')

        if any(x is None for x in [net_income, gross_profit, operating_income, revenue]):
            return 0

        if revenue.size < 1 or revenue.iloc[-1] == 0:
            return 0

        net_growth = net_income.pct_change().iloc[-1]

        latest_rev = revenue.iloc[-1]
        net_margin = net_income.iloc[-1] / latest_rev
        gross_margin = gross_profit.iloc[-1] / latest_rev
        op_margin = operating_income.iloc[-1] / latest_rev

        if net_growth > 0.15:
            score += 0.25
        elif net_growth < 0:
            score -= 0.25

        if gross_margin > 0.5:
            score += 0.3
        elif gross_margin > 0.3:
            score += 0.15
        elif gross_margin < 0.2:
            score -= 0.2

        if op_margin > 0.25:
            score += 0.3
        elif op_margin > 0.15:
            score += 0.2
        elif op_margin < 0.05:
            score -= 0.3

        if net_margin > 0.2:
            score += 0.3
        elif net_margin > 0.1:
            score += 0.2
        elif net_margin < 0:
            score -= 0.3

        time.sleep(0.05)
        return tanh(score)

    except Exception:
        return 0


def analyze_eps(t: yf.Ticker):
    try:
        score = 0
        eps = _get_row(t.financials, 'Basic EPS')

        if eps is None or eps.size < 2:
            return 0

        growth = eps.pct_change().iloc[-1]

        if growth > 0.1:
            score += 0.15
        elif growth < 0:
            score -= 0.2

        time.sleep(0.05)
        return tanh(score)

    except Exception:
        return 0


def analyze_predictions(t: yf.Ticker, ticker: str):
    score = 0

    signal = safe_recommendations(t, ticker)
    pred = safe_price_targets(t)

    # price targets
    if pred:
        try:
            median = pred.get('median')
            mean = pred.get('mean')
            current = pred.get('current')

            if median and mean and current:
                avg = (median + mean) / 2
                if avg > current:
                    score += 0.4
                elif avg < current:
                    score -= 0.4
        except Exception:
            pass

    # recommendations
    if signal is not None and not signal.empty:
        try:
            strong_buy = signal['strongBuy'].mean()
            buy = signal['buy'].mean()
            sell = signal['sell'].mean()
            strong_sell = signal['strongSell'].mean()

            if strong_buy + buy > strong_sell + sell:
                score += 0.3
            else:
                score -= 0.3
        except Exception:
            pass
            
    time.sleep(0.05)
    return tanh(score)
