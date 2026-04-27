import logging
import os

from fastapi import FastAPI

from app.api.sentiment import router as sentiment_router
from app.models.sentiment_model import warm_up_sentiment_model

log = logging.getLogger(__name__)

app = FastAPI(title="Stock Sentiment API", description="API to compute stock sentiment scores.")
app.include_router(sentiment_router, prefix="/sentiment")

def _env_truthy(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


@app.on_event("startup")
def preload_news_model():
    if not _env_truthy("NEWS_SENTIMENT_EAGER_LOAD", "1"):
        return

    try:
        model_dir = warm_up_sentiment_model()
        log.info("News sentiment model warmed up from %s", model_dir)
    except Exception as exc:
        log.warning("News sentiment warmup failed: %s", exc)

@app.get("/")
def status():
    """Basic endpoint to check if the API is running."""
    return {"status": "API is running. Endpoints: /sentiment/{ticker}, /components/technical/{ticker}, /components/fundamental/{ticker}"}
