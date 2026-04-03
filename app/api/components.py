from fastapi import APIRouter, Depends, HTTPException

from app.data.market import fetch_stock_data,compute_technical
from app.core.scoring_engine import technical_score,fundamental_score
from app.data.news import score_news_for_ticker
from app.security.rate_limiting import rate_limiter

router = APIRouter()

@router.get('/technical/{ticker}')
def get_technical_sentiment(ticker: str, _ : None = Depends(rate_limiter)):
    """Endpoint to get a technical sentiment score for a given stock ticker."""
    data = fetch_stock_data(ticker)

    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=429, detail=data["error"])
    
    if data.empty:
        return {"error": "Ticker not found or no data available."}
    
    df = compute_technical(data)
    score = technical_score(df)
    return {"ticker": ticker, "Technical Score": score}

@router.get('/fundamental/{ticker}')
def get_fundamental_sentiment(ticker: str, _ : None = Depends(rate_limiter)):
    """Endpoint to get a fundamental sentiment score for a given stock ticker."""
    score = fundamental_score(ticker)
    return {"ticker": ticker, "Fundamental Score": score}

@router.get('/news/{ticker}')
def get_news_sentiment(ticker: str, _ : None = Depends(rate_limiter)):
    """Endpoint to get a news sentiment score for a given stock ticker."""
    score = score_news_for_ticker(ticker)
    return {"ticker": ticker, "News Score": score}