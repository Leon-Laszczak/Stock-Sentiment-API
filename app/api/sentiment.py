from fastapi import APIRouter, Depends, HTTPException

from app.data.market import fetch_stock_data,compute_technical
from app.core.scoring_engine import technical_score,fundamental_score,get_final_score
from app.data.news import score_news_for_ticker
from app.security.rate_limiting import rate_limiter

router = APIRouter()

@router.get("/{ticker}")
def get_sentiment(ticker: str, _ : None = Depends(rate_limiter)):
    """Endpoint to get a sentiment score for a given stock ticker."""
    data = fetch_stock_data(ticker)
    
    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=429, detail=data["error"])
    
    if data.empty:
        return {"error": "Ticker not found or no data available."}
    
    df = compute_technical(data)
    tech = technical_score(df)

    fund = fundamental_score(ticker)

    news = score_news_for_ticker(ticker)

    overall_sentiment = get_final_score(tech['Score'], news['Score'], fund['Score'])

    return {
        'ticker' : ticker,
        'Score' : overall_sentiment,
        'Components' : {
            'Technical' : tech,
            'Fundamental' : fund,
            'News' : news
        }
    }
