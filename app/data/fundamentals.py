import requests

import yfinance as yf
import pandas as pd
from math import tanh

from typing import Callable

from app.core.helpers import _require_env

MASSIVE_BASE_URL = 'https://api.massive.com'

def _get_row(df: pd.DataFrame, label: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if label not in df.index:
        return None
    series = df.loc[label]
    if isinstance(series, pd.Series):
        return series.sort_index()
    return None

def analyze_revenue(t : yf.Ticker):
    score = 0

    income_statement = t.financials
    revenue = _get_row(income_statement, 'Total Revenue')
    if revenue is None or revenue.size < 2:
        return 0
    revenue_growth = revenue.pct_change().iloc[-1]

    if revenue_growth > 0.15:
        score += 0.25
    elif revenue_growth > 0.05:
        score += 0.1
    elif revenue_growth < 0:
        score -= 0.25
    
    return tanh(score)

def analyze_income_and_margins(t : yf.Ticker):
    score = 0

    income_statement = t.financials

    net_income = _get_row(income_statement, 'Net Income')
    gross_profit = _get_row(income_statement, 'Gross Profit')
    operating_income = _get_row(income_statement, 'Operating Income')
    revenue = _get_row(income_statement, 'Total Revenue')

    if (
        net_income is None
        or gross_profit is None
        or operating_income is None
        or revenue is None
        or revenue.size < 1
    ):
        return 0

    net_income_growth = net_income.pct_change().iloc[-1]
    latest_revenue = revenue.iloc[-1]
    if latest_revenue == 0:
        return 0

    net_margin = (net_income.iloc[-1] / latest_revenue)
    gross_margin = (gross_profit.iloc[-1] / latest_revenue)
    operating_margin = (operating_income.iloc[-1] / latest_revenue)

    if net_income_growth > 0.15:
        score += 0.25
    elif net_income_growth < 0:
        score -= 0.25
    
    if gross_margin > 0.5:
        score += 0.3
    elif gross_margin > 0.3:
        score += 0.15
    elif gross_margin < 0.2:
        score -= 0.2

    if operating_margin > 0.25:
        score += 0.3
    elif operating_margin > 0.15:
        score += 0.2
    elif operating_margin < 0.5:
        score -= 0.3

    if net_margin > 0.2:
        score += 0.3
    elif net_margin > 0.1:
        score += 0.2
    elif net_margin < 0:
        score -= 0.3
    
    return tanh(score)

def analyze_eps(t : yf.Ticker):
    score = 0
    
    income_statement = t.financials

    eps = _get_row(income_statement, 'Basic EPS')
    if eps is None or eps.size < 2:
        return 0

    eps_growth = eps.pct_change().iloc[-1]
    if eps_growth > 0.1:
        score += 0.15
    elif eps_growth < 0:
        score -= 0.2
    
    return tanh(score)

def analyze_predictions(t : yf.Ticker):
    score = 0

    signal = t.recommendations
    pred = t.analyst_price_targets

    if pred is not None:
        if (pred.get('median') and pred.get('mean')) > pred.get('current'):
            score += 0.4
        elif (pred.get('median') and pred.get('mean')) < pred.get('current'):
            score -= 0.4
    
    if not signal.empty:
        strong_buy = signal['strongBuy'].mean()
        buy = signal['buy'].mean()
        sell = signal['sell'].mean()
        strong_sell = signal['strongSell'].mean()

        if strong_buy + buy > strong_sell + sell:
            if strong_buy > buy:
                score += 0.4
            else: 
                score += 0.2
        elif strong_sell > sell:
            score -= 0.4
        else:
            score -= 0.2

    return tanh(score)

def _resolve_massive_api_key() -> str:
    for env_var in ['MASSIVE_API_KEY', 'POLYGON_API_KEY']:
        value = _require_env(env_var)
        return value

    raise RuntimeError(
        "Missing Massive API key. Set MASSIVE_API_KEY (preferred) "
        "or POLYGON_API_KEY in your environment."
    )


def _massive_get(path: str, params: dict | None = None) -> dict:
    api_key = _resolve_massive_api_key()
    request_params = dict(params or {})
    request_params.setdefault('apiKey', api_key)

    response = requests.get(
        f'{MASSIVE_BASE_URL}{path}',
        headers={
            'Authorization': f'Bearer {api_key}',
        },
        params=request_params,
        timeout=15,
    )
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, dict) and payload.get('error'):
        raise RuntimeError(f'Massive error: {payload.get("error")}')
    if isinstance(payload, dict) and payload.get('status') == 'ERROR':
        msg = payload.get('message') or payload.get('error') or 'Unknown Massive API error'
        raise RuntimeError(f'Massive error: {msg}')

    if isinstance(payload, dict):
        return payload
    return {}


def _massive_get_results(path: str, params: dict | None = None) -> list[dict]:
    payload = _massive_get(path=path, params=params)
    results = payload.get('results', [])

    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    if isinstance(results, dict):
        return [results]
    return []


def _prepare_time_sorted_frame(rows: list[dict], date_col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(by=date_col, ascending=True)
    else:
        df = df.reset_index(drop=True)
    return df


def _get_numeric_column(df: pd.DataFrame, label: str) -> pd.Series | None:
    if df is None or df.empty or label not in df.columns:
        return None
    series = pd.to_numeric(df[label], errors='coerce').dropna()
    if series.empty:
        return None
    return series


def _get_latest_numeric(df: pd.DataFrame, label: str) -> float | None:
    series = _get_numeric_column(df, label)
    if series is None:
        return None
    return float(series.iloc[-1])


def _resolve_income_column(df: pd.DataFrame, *labels: str) -> str | None:
    for label in labels:
        series = _get_numeric_column(df, label)
        if series is not None and not series.empty:
            return label
    return None


def _download_massive_income_statement(ticker: str) -> pd.DataFrame:
    try:
        rows = _massive_get_results(
            '/stocks/financials/v1/income-statements',
            params={
                'tickers': ticker,
                'timeframe': 'annual',
                'limit': 8,
                'sort': 'asc',
            },
        )
        return _prepare_time_sorted_frame(rows, date_col='period_end')
    except Exception:
        return pd.DataFrame()


def _download_massive_ratios(ticker: str) -> pd.DataFrame:
    try:
        rows = _massive_get_results(
            '/stocks/financials/v1/ratios',
            params={
                'ticker': ticker,
                'limit': 8,
                'sort': 'asc',
            },
        )
        return _prepare_time_sorted_frame(rows, date_col='date')
    except Exception:
        return pd.DataFrame()


def _download_massive_consensus(ticker: str) -> pd.DataFrame:
    try:
        rows = _massive_get_results(
            f'/benzinga/v1/consensus-ratings/{ticker}',
            params={
                'limit': 5,
            },
        )
        return _prepare_time_sorted_frame(rows, date_col='date')
    except Exception:
        return pd.DataFrame()


class MassiveTicker:
    def __init__(self, ticker: str):
        self.ticker = ticker.strip().upper()
        self.income_statement = _download_massive_income_statement(self.ticker)
        self.ratios = _download_massive_ratios(self.ticker)
        self.consensus = _download_massive_consensus(self.ticker)
        self._yf_ticker = None

    def get_yf_ticker(self) -> yf.Ticker:
        if self._yf_ticker is None:
            self._yf_ticker = yf.Ticker(self.ticker)
        return self._yf_ticker


def _fallback_component_score(t: MassiveTicker, analyzer: Callable[[yf.Ticker], float]) -> float:
    try:
        return analyzer(t.get_yf_ticker())
    except Exception:
        return 0


def analyze_massive_revenue(t: MassiveTicker):
    score = 0

    income_statement = t.income_statement
    revenue_col = _resolve_income_column(income_statement, 'revenue')
    if revenue_col is None:
        return _fallback_component_score(t, analyze_revenue)

    revenue = _get_numeric_column(income_statement, revenue_col)
    if revenue is None or revenue.size < 2:
        return _fallback_component_score(t, analyze_revenue)

    revenue_growth = revenue.pct_change().iloc[-1]
    if pd.isna(revenue_growth):
        return _fallback_component_score(t, analyze_revenue)

    if revenue_growth > 0.15:
        score += 0.25
    elif revenue_growth > 0.05:
        score += 0.1
    elif revenue_growth < 0:
        score -= 0.25

    return tanh(score)


def analyze_massive_income_and_margins(t: MassiveTicker):
    score = 0

    income_statement = t.income_statement

    net_income_col = _resolve_income_column(
        income_statement,
        'net_income_loss_attributable_common_shareholders',
        'consolidated_net_income_loss',
    )
    gross_profit_col = _resolve_income_column(income_statement, 'gross_profit')
    operating_income_col = _resolve_income_column(income_statement, 'operating_income')
    revenue_col = _resolve_income_column(income_statement, 'revenue')

    if (
        net_income_col is None
        or gross_profit_col is None
        or operating_income_col is None
        or revenue_col is None
    ):
        return _fallback_component_score(t, analyze_income_and_margins)

    net_income = _get_numeric_column(income_statement, net_income_col)
    gross_profit = _get_numeric_column(income_statement, gross_profit_col)
    operating_income = _get_numeric_column(income_statement, operating_income_col)
    revenue = _get_numeric_column(income_statement, revenue_col)

    if (
        net_income is None
        or gross_profit is None
        or operating_income is None
        or revenue is None
        or revenue.size < 1
    ):
        return _fallback_component_score(t, analyze_income_and_margins)

    net_income_growth = net_income.pct_change().iloc[-1] if net_income.size >= 2 else 0
    latest_revenue = revenue.iloc[-1]
    if latest_revenue == 0:
        return _fallback_component_score(t, analyze_income_and_margins)

    net_margin = (net_income.iloc[-1] / latest_revenue)
    gross_margin = (gross_profit.iloc[-1] / latest_revenue)
    operating_margin = (operating_income.iloc[-1] / latest_revenue)

    if net_income_growth > 0.15:
        score += 0.25
    elif net_income_growth < 0:
        score -= 0.25

    if gross_margin > 0.5:
        score += 0.3
    elif gross_margin > 0.3:
        score += 0.15
    elif gross_margin < 0.2:
        score -= 0.2

    if operating_margin > 0.25:
        score += 0.3
    elif operating_margin > 0.15:
        score += 0.2
    elif operating_margin < 0.5:
        score -= 0.3

    if net_margin > 0.2:
        score += 0.3
    elif net_margin > 0.1:
        score += 0.2
    elif net_margin < 0:
        score -= 0.3

    return tanh(score)


def analyze_massive_eps(t: MassiveTicker):
    score = 0

    income_statement = t.income_statement

    eps_col = _resolve_income_column(
        income_statement,
        'basic_earnings_per_share',
        'diluted_earnings_per_share',
    )
    if eps_col is None:
        return _fallback_component_score(t, analyze_eps)

    eps = _get_numeric_column(income_statement, eps_col)
    if eps is None or eps.size < 2:
        return _fallback_component_score(t, analyze_eps)

    eps_growth = eps.pct_change().iloc[-1]
    if pd.isna(eps_growth):
        return _fallback_component_score(t, analyze_eps)

    if eps_growth > 0.1:
        score += 0.15
    elif eps_growth < 0:
        score -= 0.2

    return tanh(score)


def _get_consensus_count(df: pd.DataFrame, label: str) -> float:
    value = _get_latest_numeric(df, label)
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def analyze_massive_predictions(t: MassiveTicker):
    score = 0
    has_massive_signal = False

    signal = t.consensus
    pred = t.ratios

    consensus_target = _get_latest_numeric(signal, 'consensus_price_target')
    current_price = _get_latest_numeric(pred, 'price')

    if consensus_target is not None and current_price is not None:
        has_massive_signal = True
        if consensus_target > current_price:
            score += 0.4
        elif consensus_target < current_price:
            score -= 0.4

    strong_buy = _get_consensus_count(signal, 'strong_buy_ratings')
    buy = _get_consensus_count(signal, 'buy_ratings')
    sell = _get_consensus_count(signal, 'sell_ratings')
    strong_sell = _get_consensus_count(signal, 'strong_sell_ratings')

    if strong_buy + buy + sell + strong_sell > 0:
        has_massive_signal = True
        if strong_buy + buy > strong_sell + sell:
            if strong_buy > buy:
                score += 0.4
            else:
                score += 0.2
        elif strong_sell > sell:
            score -= 0.4
        else:
            score -= 0.2
    elif signal is not None and not signal.empty and 'consensus_rating' in signal.columns:
        consensus_rating = signal['consensus_rating'].dropna()
        if not consensus_rating.empty:
            has_massive_signal = True
            label = str(consensus_rating.iloc[-1]).strip().lower()
            if 'strong buy' in label:
                score += 0.4
            elif label == 'buy':
                score += 0.2
            elif 'strong sell' in label:
                score -= 0.4
            elif label == 'sell':
                score -= 0.2

    if not has_massive_signal:
        return _fallback_component_score(t, analyze_predictions)

    return tanh(score)


def analyze_valuation(t: MassiveTicker):
    score = 0

    ratios = t.ratios

    pe_ratio = _get_latest_numeric(ratios, 'pe_ratio')
    pb_ratio = _get_latest_numeric(ratios, 'pb_ratio')
    ps_ratio = _get_latest_numeric(ratios, 'ps_ratio')

    if pe_ratio is not None:
        if pe_ratio < 15:
            score += 0.3
        elif pe_ratio < 25:
            score += 0.1
        elif pe_ratio > 50:
            score -= 0.3

    if pb_ratio is not None:
        if pb_ratio < 1.5:
            score += 0.3
        elif pb_ratio < 3:
            score += 0.1
        elif pb_ratio > 5:
            score -= 0.3

    if ps_ratio is not None:
        if ps_ratio < 1:
            score += 0.3
        elif ps_ratio < 3:
            score += 0.1
        elif ps_ratio > 5:
            score -= 0.3

    ev_ebitda = _get_latest_numeric(ratios, 'ev_to_ebitda')
    if ev_ebitda is not None:
        if ev_ebitda < 10:
            score += 0.3
        elif ev_ebitda < 20:
            score += 0.1
        elif ev_ebitda > 30:
            score -= 0.3

    return tanh(score)


def analyze_debt(t: MassiveTicker):
    score = 0

    ratios = t.ratios

    debt_to_equity = _get_latest_numeric(ratios, 'debt_to_equity')

    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 0.3
        elif debt_to_equity < 1:
            score += 0.1
        elif debt_to_equity > 2:
            score -= tanh(debt_to_equity - 2) * 0.3
    
    return score
        