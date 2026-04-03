# Stock Sentiment API (MVP)

A simple API that estimates stock sentiment from three sources:

- technical indicators
- company fundamentals
- recent news

Provide a ticker -> get one overall score plus a full breakdown.

---

## What It Does

For a ticker like `AAPL`, the API combines:

- Technical score from `RSI`, `MACD`, and `EMA`
- Fundamental score from revenue, margins, EPS, and analyst expectations
- News score from recent headlines and summaries using the local model in `models/free`

Final score weights:

- `40%` fundamentals
- `30%` technicals
- `30%` news

---

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

---

## Endpoints

`GET /`

Basic health check.

`GET /sentiment/{ticker}`

Returns the full sentiment score with technical, fundamental, and news components.

`GET /components/technical/{ticker}`

Returns only the technical score and its breakdown.

`GET /components/fundamental/{ticker}`

Returns only the fundamental score and its breakdown.

---

## Getting Started

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- Docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/`
- Example: `http://127.0.0.1:8000/sentiment/AAPL`

---

## Project Structure

- `app/api` - API routes
- `app/data` - market, fundamentals, and news data collection
- `app/core` - scoring logic and helper functions
- `app/models` - sentiment model wrapper

---

## Notes

- Data comes from Yahoo Finance
- The app uses simple in-memory caching to reduce repeated fetches
- Rate limit is `10 request / second / IP`
- If Yahoo Finance is rate limited or data is missing, some components may return fallback values
- This is an MVP and not financial advice

---

## Feedback

I am looking for feedback on:

- Is the output easy to understand?
- Does the score breakdown feel useful?
- Which component matters most to you: technicals, fundamentals, or news?
- What would make this API worth using again?

---

## License

MIT
