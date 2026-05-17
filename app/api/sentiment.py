from concurrent.futures import ThreadPoolExecutor
import time

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import numpy as np

from app.core.confidence import (
    compute_fundamental_confidence,
    compute_news_confidence,
    compute_overall_confidence,
    compute_technical_confidence_breakdown,
)
from app.data.market import fetch_stock_data,compute_technical
from app.core.scoring_engine import technical_score,fundamental_score,get_final_score
from app.data.fundamentals import MassiveTicker
from app.data.news import score_news_for_ticker

router = APIRouter()
FUNDAMENTAL_CACHE_TTL = 60 * 60 * 24
fundamental_cache: dict[str, tuple[dict, float]] = {}
fundamental_last_fetch: dict[str, float] = {}


def _load_fundamentals_with_confidence(ticker: str) -> tuple[dict, float]:
    ticker_key = ticker.upper()
    now = time.time()

    if ticker_key in fundamental_cache and ticker_key in fundamental_last_fetch:
        if now - fundamental_last_fetch[ticker_key] < FUNDAMENTAL_CACHE_TTL:
            return fundamental_cache[ticker_key]

    fundamentals_ticker = MassiveTicker(ticker_key)
    fundamentals = fundamental_score(
        ticker_key,
        fundamentals_ticker=fundamentals_ticker,
    )
    confidence = compute_fundamental_confidence(
        fundamentals_ticker=fundamentals_ticker,
        fund_score=fundamentals.get("Score"),
    )
    result = (fundamentals, confidence)
    fundamental_cache[ticker_key] = result
    fundamental_last_fetch[ticker_key] = now
    return result

@router.get("/{ticker}/{data_mode}")
def get_sentiment(ticker: str, data_mode: str):
    """Endpoint to get a sentiment score for a given stock ticker."""
    if data_mode not in ['fast', 'full']:
        raise HTTPException(status_code=400, detail="Invalid data_mode. Must be 'fast' or 'full'.")
    with ThreadPoolExecutor(max_workers=3) as executor:
        market_future = executor.submit(fetch_stock_data, ticker)
        fund_future = executor.submit(_load_fundamentals_with_confidence, ticker)
        news_future = executor.submit(score_news_for_ticker, ticker, data_mode=data_mode)

        data = market_future.result()
    
        if isinstance(data, dict) and "error" in data:
            raise HTTPException(status_code=429, detail=data["error"])
        
        if not isinstance(data, list) or not data:
            return {"error": "No market data available."}

        if all(getattr(frame, "empty", True) for frame in data if not isinstance(frame, dict)):
            return {"error": "Ticker not found or no data available."}
        
        dfs = compute_technical(data)
        tech = technical_score(dfs)

        fund, fund_confidence = fund_future.result()
        news = news_future.result()

    overall_sentiment = get_final_score(tech['Score'], news['score'], fund['Score'])
    technical_confidence = compute_technical_confidence_breakdown(
        dfs=dfs,
        tech_score=tech.get("Score", 0.0),
        tech_payload=tech,
    )
    news_confidence = compute_news_confidence(
        news_score=news.get("score"),
        has_data=bool(news.get("has_data")),
        article_count=news.get("article_count"),
        distinct_sources=news.get("distinct_sources"),
        latest_pub_date=news.get("latest_pub_date"),
    )
    overall_confidence = compute_overall_confidence(
        {
            "technical": technical_confidence.get("Overall"),
            "fundamentals": fund_confidence,
            "news": news_confidence,
        }
    )

    payload = {
        'ticker' : ticker,
        'Score' : overall_sentiment,
        'Components' : {
            'Technical' : tech,
            'Fundamental' : fund,
            'News' : news
        },
        'Confidence': {
            'Overall': overall_confidence,
            'Technical': technical_confidence,
            'Fundamental': fund_confidence,
            'News': news_confidence,
        }
    }
    return jsonable_encoder(payload, custom_encoder={np.generic: lambda value: value.item()})
