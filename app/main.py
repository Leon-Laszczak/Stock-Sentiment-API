from fastapi import FastAPI

from app.api.sentiment import router as sentiment_router
from app.api.components import router as components_router

app = FastAPI(title="Stock Sentiment API", description="API to compute stock sentiment scores.")
app.include_router(sentiment_router, prefix="/sentiment")
app.include_router(components_router, prefix="/components")

@app.get("/")
def status():
    """Basic endpoint to check if the API is running."""
    return {"status": "API is running. Endpoints: /sentiment/{ticker}, /components/technical/{ticker}, /components/fundamental/{ticker}"}