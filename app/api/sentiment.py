from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from app.data.market import fetch_stock_data,compute_technical
from app.core.scoring_engine import technical_score,fundamental_score,get_final_score
from app.data.news import score_news_for_ticker

router = APIRouter()

@router.get("/{ticker}")
def get_sentiment(ticker: str,):
    """Endpoint to get a sentiment score for a given stock ticker."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        market_future = executor.submit(fetch_stock_data, ticker)
        fund_future = executor.submit(fundamental_score, ticker)
        news_future = executor.submit(score_news_for_ticker, ticker)

        data = market_future.result()
    
        if isinstance(data, dict) and "error" in data:
            raise HTTPException(status_code=429, detail=data["error"])
        
        if not isinstance(data, list) or not data:
            return {"error": "No market data available."}

        if all(getattr(frame, "empty", True) for frame in data if not isinstance(frame, dict)):
            return {"error": "Ticker not found or no data available."}
        
        dfs = compute_technical(data)
        tech = technical_score(dfs)

        fund = fund_future.result()
        news = news_future.result()

    overall_sentiment = get_final_score(tech['Score'], news['score'], fund['Score'])

    return {
        'ticker' : ticker,
        'Score' : overall_sentiment,
        'Components' : {
            'Technical' : tech,
            'Fundamental' : fund,
            'News' : news
        }
    }
