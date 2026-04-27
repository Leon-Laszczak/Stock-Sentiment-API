# Stock Sentiment API

Local FastAPI application for stock sentiment analysis. The project combines:

- technical indicators
- company fundamentals
- recent news scored by a locally loaded news sentiment model

The news model is now part of this application flow, so you do not need a separate News API service to use it.

## What It Does

For a ticker like `AAPL`, the API combines:

- Technical score from `RSI`, `MACD`, and `EMA`
- Fundamental score from revenue, margins, EPS, and analyst expectations
- News score from recent headlines and summaries analyzed inside this app

Final score weights:

- `40%` fundamentals
- `30%` technicals
- `30%` news

## Example

`GET /sentiment/AAPL`

Response:

```json
{
  "ticker": "AAPL",
  "Score": 0.31,
  "Components": {
    "Technical": {
      "Score": 0.42,
      "RSI": 0.18,
      "MACD": 0.61,
      "EMA": 0.47
    },
    "Fundamental": {
      "Score": 0.25,
      "Score Breakdown": {
        "Revenue Growth": 0.2,
        "Income and Margins": 0.3,
        "EPS Growth": 0.1,
        "Analyst Predictions": 0.4
      }
    },
    "News": {
      "score": 0.27,
      "has_data": true,
      "article_count": 8,
      "distinct_sources": 4,
      "latest_pub_date": "2026-03-30T12:00:00+00:00"
    }
  }
}
```

## Endpoints

`GET /`

Basic local status check.

`GET /sentiment/{ticker}`

Returns the full sentiment score with technical, fundamental, and news components.

`GET /components/technical/{ticker}`

Returns only the technical score and its breakdown.

`GET /components/fundamental/{ticker}`

Returns only the fundamental score and its breakdown.

`GET /components/news/{ticker}`

Returns only the news sentiment component.

## Local Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open locally:

- Docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/`
- Example: `http://127.0.0.1:8000/sentiment/AAPL`

## News Model

The app loads the news sentiment model directly inside `app/models/sentiment_model.py`.

- Preferred local path: `models/news_classification_free_quantized`
- Optional override: set `NEWS_SENTIMENT_MODEL_PATH`
- Fallback: if no local folder is present, the configured Hugging Face model identifier is used and cached locally by the libraries on first load

This means the sentiment inference runs locally in the same process as the API instead of calling a separate hosted sentiment endpoint.

## Project Structure

- `app/api` - API routes
- `app/data` - market, fundamentals, and news data collection
- `app/core` - scoring logic and helper functions
- `app/models` - local news sentiment model wrapper

## Notes

- Data comes from Yahoo Finance
- The app uses simple in-memory caching to reduce repeated fetches
- Rate limit is `10 request / second / IP`
- If Yahoo Finance is rate limited or data is missing, some components may return fallback values
- This project is intended for local use
- This is an MVP and not financial advice

## License

MIT
