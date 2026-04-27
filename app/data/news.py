from typing import Any

import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import pandas as pd

import time

from app.models.sentiment_model import predict_sentiment_scores

cache = {}
last_fetch = {}
score_cache = {}
score_last_fetch = {}
CACHE_TTL = 60 * 60 * 4 # 4 hours

def _empty_news_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['title', 'summary', 'pub_date', 'provider'])

def _extract_provider_name(provider: Any) -> str:
    if isinstance(provider, dict):
        val = provider.get('displayName') or provider.get('name') or provider.get('url')
        return str(val).strip() if val else ''
    if provider is None:
        return ''
    return str(provider).strip()

def fetch_news(ticker):
    now = time.time()

    if ticker in cache and ticker in last_fetch:
        if now - last_fetch[ticker] < CACHE_TTL:
            return cache[ticker]

    try:
        raw_news = yf.Ticker(ticker).news
        
        if not raw_news:
            return _empty_news_df()
        df = pd.DataFrame(raw_news)
        if df.empty or 'content' not in df.columns:
            return _empty_news_df()

        content = pd.json_normalize(df['content'])
        if content.empty:
            return _empty_news_df()
        
        news = pd.DataFrame(index=content.index)
        news['title'] = content['title'] if 'title' in content.columns else ''
        news['summary'] = content['summary'] if 'summary' in content.columns else ''
        news['pub_date'] = content['pubDate'] if 'pubDate' in content.columns else pd.NaT
        if 'provider' in content.columns:
            news['provider'] = content['provider'].map(_extract_provider_name)
        elif 'provider.displayName' in content.columns:
            news['provider'] = content['provider.displayName'].fillna('').astype(str).str.strip()
        elif 'provider.name' in content.columns:
            news['provider'] = content['provider.name'].fillna('').astype(str).str.strip()
        elif 'provider.url' in content.columns:
            news['provider'] = content['provider.url'].fillna('').astype(str).str.strip()
        else:
            news['provider'] = ''

        cache[ticker] = news
        last_fetch[ticker] = now
        return news
    
    except YFRateLimitError:
        print("[WARN] news rate limited")
        return cache.get(ticker, _empty_news_df())

    except Exception as e:
        print(f"[ERROR] fetch_news: {e}")
        return cache.get(ticker, _empty_news_df())

def score_news_dataframe(
    news: pd.DataFrame,
) -> dict[str, Any]:
    """Compute a continuous sentiment score from downloaded news rows."""
    if news is None or news.empty:
        return {
            'score': 0.0,
            'has_data': False,
            'article_count': 0,
            'distinct_sources': 0,
            'latest_pub_date': None,
        }

    title = news['title'] if 'title' in news.columns else pd.Series('', index=news.index)
    summary = news['summary'] if 'summary' in news.columns else pd.Series('', index=news.index)
    texts = (title.fillna('').astype(str).str.strip() + ' ' + summary.fillna('').astype(str).str.strip()).str.strip()
    text_list = [t for t in texts.tolist() if t]
    if not text_list:
        return {
            'score': 0.0,
            'has_data': False,
            'article_count': 0,
            'distinct_sources': 0,
            'latest_pub_date': None,
        }

    providers = news['provider'] if 'provider' in news.columns else pd.Series('', index=news.index)
    providers = providers.fillna('').astype(str).str.strip()
    distinct_sources = int((providers[providers != '']).nunique())

    latest_pub_date = None
    if 'pub_date' in news.columns:
        pub_dates = pd.to_datetime(news['pub_date'], errors='coerce', utc=True).dropna()
        if not pub_dates.empty:
            latest_pub_date = pub_dates.max().isoformat()

    tier = 'free'
    score,results = predict_sentiment_scores(text_list, tier=tier)
    for r in results:
        if 'error' in r:
            return {
                'score': 0.0,
                'has_data': False,
                'article_count': len(text_list),
                'distinct_sources': distinct_sources,
                'latest_pub_date': latest_pub_date,
                'error': r['error'],
            }

    components = {str(idx) : {'score' : r['score'],'pub_date' : str(pd.to_datetime(news['pub_date'], errors='coerce', utc=True)[idx]),'source' : providers[idx]} for (idx,r) in enumerate(results)} 
    return {
        'score': score,
        'has_data': True,
        'article_count': len(text_list),
        'distinct_sources': distinct_sources,
        'latest_pub_date': latest_pub_date,
        'components' : components
    }

def score_news_for_ticker(
    ticker: str,
) -> dict[str, Any]:
    """Download news and return aggregate sentiment score for one ticker."""
    try:
        now = time.time()
        ticker_key = ticker.upper()
        if ticker_key in score_cache and ticker_key in score_last_fetch:
            if now - score_last_fetch[ticker_key] < CACHE_TTL:
                return score_cache[ticker_key]

        news = fetch_news(ticker)
        result = score_news_dataframe(news)
        if 'error' in result:
            return {
                'ticker': ticker,
                'score': 0.0,
                'has_data': False,
                'article_count': 0,
                'distinct_sources': 0,
                'latest_pub_date': None,
                'error': result['error'],
            }
        result['ticker'] = ticker
        score_cache[ticker_key] = result
        score_last_fetch[ticker_key] = now
        return result
    except Exception as exc:
        return {
            'ticker': ticker,
            'score': 0.0,
            'has_data': False,
            'article_count': 0,
            'distinct_sources': 0,
            'latest_pub_date': None,
            'error': str(exc),
        }
