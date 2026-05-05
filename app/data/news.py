from typing import Any, Optional

import yfinance as yf
from yfinance.exceptions import YFRateLimitError
import pandas as pd

import requests
import finnhub
import feedparser
from bs4 import BeautifulSoup
import re

from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
import time

from app.models.sentiment_model import predict_sentiment_scores
from app.core.helpers import _require_env

cache = {}
last_fetch = {}
score_cache = {}
score_last_fetch = {}
CACHE_TTL = 60 * 60 * 4 # 4 hours

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
DAYS_BACK = 30

def _empty_news_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['title', 'summary', 'pub_date', 'provider'])


def _empty_parser_news_df() -> pd.DataFrame:
    return pd.DataFrame(columns=['title', 'summary', 'published', 'url', 'ticker', 'source'])


def _extract_provider_name(provider: Any) -> str:
    if isinstance(provider, dict):
        val = provider.get('displayName') or provider.get('name') or provider.get('url')
        return str(val).strip() if val else ''
    if provider is None:
        return ''
    return str(provider).strip()

DAYS_BACK = 30

def _cutoff() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

def _parse_date(raw) -> Optional[datetime]:
    """Robustly parse any date string/struct into a timezone-aware datetime."""
    if raw is None:
        return None
    try:
        if hasattr(raw, "tm_year"):          # time.struct_time from feedparser
            ts = time.mktime(raw)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        return dateparser.parse(str(raw), fuzzy=True).replace(tzinfo=timezone.utc)
    except Exception:
        return None
 
 
def _is_recent(dt: Optional[datetime]) -> bool:
    if dt is None:
        return True   # Keep if we can't parse the date
    return dt >= _cutoff()


def _prepare_scored_news(
    news: pd.DataFrame,
) -> pd.DataFrame:
    if news is None or news.empty:
        return pd.DataFrame(columns=['title', 'summary', 'pub_date', 'provider', 'text'])

    prepared = pd.DataFrame(index=news.index)
    prepared['title'] = news['title'] if 'title' in news.columns else ''
    prepared['summary'] = news['summary'] if 'summary' in news.columns else ''
    prepared['provider'] = news['provider'] if 'provider' in news.columns else ''
    prepared['pub_date'] = (
        pd.to_datetime(news['pub_date'], errors='coerce', utc=True)
        if 'pub_date' in news.columns
        else pd.NaT
    )

    prepared['title'] = prepared['title'].fillna('').astype(str).str.strip()
    prepared['summary'] = prepared['summary'].fillna('').astype(str).str.strip()
    prepared['provider'] = prepared['provider'].fillna('').astype(str).str.strip()
    prepared['text'] = (prepared['title'] + ' ' + prepared['summary']).str.strip()

    prepared = prepared[prepared['text'] != '']
    if prepared.empty:
        return prepared

    prepared = prepared.sort_values('pub_date', ascending=False, na_position='last')
    prepared = prepared.drop_duplicates(subset=['text'], keep='first')
    return prepared.reset_index(drop=True)

_PRODUCT_NOISE = re.compile(
    r"\b(review|best \w+ (mouse|keyboard|headset|webcam|speaker|headphone)|"
    r"buying guide|how to|deal|discount|coupon|sale|unboxing|hands.on|"
    r"vs\.?|comparison|gaming mouse|wireless mouse|mechanical keyboard|"
    r"top \d+|ranked|budget pick|editors? choice|gift guide)\b",
    re.IGNORECASE,
)
 
# Words that confirm the article is about the company as a financial entity
_FINANCIAL_SIGNAL = re.compile(
    r"\b(stock|share|investor|earnings|revenue|profit|loss|quarter|fiscal|"
    r"guidance|outlook|analyst|rating|upgrade|downgrade|target price|"
    r"dividend|buyback|acquisition|merger|CEO|CFO|annual report|"
    r"results|forecast|valuation|market cap|NYSE|NASDAQ|SIX|IPO|"
    r"EPS|beat|miss|consensus)\b",
    re.IGNORECASE,
)
 
 
def _is_financial_article(title: str, summary: str = "") -> bool:
    """
    Returns True if the article is about the company as a financial entity,
    False if it looks like a product review / buying guide.
 
    Logic:
      - If title matches product noise patterns → reject
      - If title OR summary contains financial signal → accept
      - Otherwise → reject (too ambiguous / likely product content)
    """
    text = f"{title} {summary}"
    if _PRODUCT_NOISE.search(title):
        return False
    if _FINANCIAL_SIGNAL.search(text):
        return True
    return False

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

    prepared_news = _prepare_scored_news(news)
    if prepared_news.empty:
        return {
            'score': 0.0,
            'has_data': False,
            'article_count': 0,
            'distinct_sources': 0,
            'latest_pub_date': None,
        }

    text_list = prepared_news['text'].tolist()
    providers = prepared_news['provider']
    distinct_sources = int((providers[providers != '']).nunique())

    latest_pub_date = None
    pub_dates = prepared_news['pub_date'].dropna()
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
                'fetched_article_count': int(len(news)),
                'error': r['error'],
            }

    component_pub_dates = prepared_news['pub_date'].tolist()
    component_sources = providers.tolist()
    components = {
        str(idx): {
            'score': result['score'],
            'pub_date': pub_date.isoformat() if not pd.isna(pub_date) else None,
            'source': source,
        }
        for idx, (result, pub_date, source) in enumerate(
            zip(results, component_pub_dates, component_sources)
        )
    }
    return {
        'score': score,
        'has_data': True,
        'article_count': len(text_list),
        'distinct_sources': distinct_sources,
        'latest_pub_date': latest_pub_date,
        'fetched_article_count': int(len(news)),
        'components' : components
    }

def score_news_for_ticker(
    ticker: str,
    data_mode: str = 'fast'
) -> dict[str, Any]:
    """Download news and return aggregate sentiment score for one ticker."""

    if data_mode not in ['fast', 'full']:
        return {
            'ticker': ticker,
            'score': 0.0,
            'has_data': False,
            'article_count': 0,
            'distinct_sources': 0,
            'latest_pub_date': None,
            'error': f"Invalid data_mode '{data_mode}'",
        }
    try:
        ticker = ticker.upper()

        if data_mode == 'fast':
            news = fetch_news(ticker)
        else:
            news = NewsParser().get_news(ticker)
            
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

class NewsParser:
    def __init__(
            self,
            ):
        self.sources = ['yfinance', 'seeking_alpha', 'google_news', 'finnhub']

        self.fetching_metods = {
            'yfinance' : self.fetch_yfinance,
            'seeking_alpha' : self.scrape_seeking_alpha,
            'google_news' : self.scrape_google_news,
            'finnhub' : self.scrape_finnhub,
        }
    
    def fetch_yfinance(self,ticker : str):
        t = yf.Ticker(ticker)
        raw_news = t.news or []
        if not raw_news:
            return _empty_parser_news_df()

        df = pd.DataFrame(raw_news)
        if df.empty or 'content' not in df.columns:
            return _empty_parser_news_df()

        content = pd.json_normalize(df['content'])
        if content.empty:
            return _empty_parser_news_df()

        news = pd.DataFrame(index=content.index)
        news['title'] = content['title'] if 'title' in content.columns else ''
        news['summary'] = content['summary'] if 'summary' in content.columns else ''
        news['published'] = content['pubDate'] if 'pubDate' in content.columns else pd.NaT
        news['url'] = content['canonicalUrl.url']
        news['ticker'] = ticker.upper()
        if 'provider' in content.columns:
            news['source'] = content['provider'].map(_extract_provider_name)
        elif 'provider.displayName' in content.columns:
            news['source'] = content['provider.displayName'].fillna('').astype(str).str.strip()
        elif 'provider.name' in content.columns:
            news['source'] = content['provider.name'].fillna('').astype(str).str.strip()
        elif 'provider.url' in content.columns:
            news['source'] = content['provider.url'].fillna('').astype(str).str.strip()
        else:
            news['source'] = ''

        return news

    def scrape_google_news(self,ticker : str):
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        company_long_name = info.get('longName') or info.get('shortName') or info.get('displayName') or ticker
        company_short_name = info.get('displayName') or info.get('shortName') or info.get('longName') or ticker

        queries = [
        f'"{company_long_name}" stock',
        f'"{company_long_name}" earnings',
        f'"{company_long_name}" investor',
        f'"{company_long_name}" revenue OR profit OR results',
        f'"{company_long_name}" financials',
        f'"{company_long_name}" prediction OR forecast OR outlook',
        f'"{ticker.upper()}" stock',
        f'Should I buy "{ticker.upper()}"',
        f'"{company_short_name}" stock',
        f'"{company_short_name}" earnings',
        f'"{company_short_name}" investor',
        f'"{company_short_name}" revenue OR profit OR results',
        f'"{company_short_name}" financials',
        f'"{company_short_name}" prediction OR forecast OR outlook',
        ]
        articles = []
        for query in queries:
            encoded = requests.utils.quote(query)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    
            try:
                feed = feedparser.parse(url, request_headers=HEADERS)
            except Exception as e:
                continue
    
            for entry in feed.entries:
                published = _parse_date(entry.get("published_parsed") or entry.get("published"))
                if not _is_recent(published):
                    continue
    
                title   = entry.get("title", "").strip()
                summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text().strip()[:300]
    
                if not _is_financial_article(title, summary):
                    continue
    
                articles.append({
                    "title":     title,
                    "summary":   summary,
                    "url":       entry.get("link", ""),
                    "source":    entry.get("source", {}).get("title", "Google News") if hasattr(entry.get("source", {}), "get") else "Google News",
                    "published": published,
                    "ticker":    ticker.upper(),
                })
    
        return pd.DataFrame(articles)
    
    def scrape_seeking_alpha(self,ticker: str) -> list[dict]:
        """
        Seeking Alpha RSS feed for a specific ticker.
        Contains analyst articles + news items – no auth needed for RSS.
        """
        url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
    
        try:
            feed = feedparser.parse(url, request_headers=HEADERS)
        except Exception as e:
            return []
    
        articles = []
        for entry in feed.entries:
            published = _parse_date(entry.get("published_parsed") or entry.get("updated_parsed"))
            if not _is_recent(published):
                continue
            title   = entry.get("title", "").strip()
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text().strip()[:300]
            if not _is_financial_article(title, summary):
                continue
            articles.append({
                "title":     title,
                "summary":   summary,
                "url":       entry.get("link", ""),
                "source":    "Seeking Alpha",
                "published": published,
                "ticker":    ticker.upper(),
            })
    
        return pd.DataFrame(articles)
    
    def scrape_finnhub(self,ticker : str):
        try:
            client = finnhub.Client(_require_env('FINNHUB_API_KEY'))
            ticker = ticker.upper()
            today = datetime.now(timezone.utc).date()
            start_date = today - timedelta(days=DAYS_BACK)

            raw_news = client.company_news(
                ticker,
                start_date.isoformat(),
                today.isoformat(),
            )

            if not raw_news:
                return _empty_parser_news_df()
            
            articles = []
            for news in raw_news:
                if not _is_financial_article(news['headline'],news['summary']):
                    continue

                articles.append({
                    'title' : news['headline'],
                    'summary' : news['summary'],
                    'url' : news['url'],
                    'source' : news['source'],
                    'published' : datetime.fromtimestamp(news['datetime'], tz=timezone.utc),
                    'ticker' : ticker
                    })

            return pd.DataFrame(articles)
        except:
            return _empty_parser_news_df()
    
         
    def get_news(self,ticker : str):
        news = pd.DataFrame()

        for source in self.sources:
            func = self.fetching_metods[source]
            news = pd.concat([news,func(ticker)])

        return news