"""FinBERT utilities for continuous news sentiment scoring.

This module trains and serves a regression model (not hard classes).
Expected CSV columns:
- title
- summary
- sentiment (float, e.g. in [-1, 1])
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

import requests
from pydantic import BaseModel


def predict_sentiment_scores(
    texts: Iterable[str],
    tier: str = 'free',
) -> tuple[float, list[dict]]:
    """Predict sentiment scores for a list of texts using the specified model tier."""
    try:
        if isinstance(texts, str):
            texts = [texts]

        results = requests.post(
            'https://news-model-for-api.onrender.com/predict',
            json = {'texts' : texts, 'tier' : tier},
        )
        scores = [res["score"] for res in results]
        avg_score = round(float(np.mean(scores)), 4)

        return avg_score, results
    except Exception as e:
        return 0.0, [{"error": str(e)}]
