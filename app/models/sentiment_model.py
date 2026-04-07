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
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import logging as log

MAX_LENGTH = 256
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
torch.set_num_threads(1)

class SentimentAnalyzer:
    """
    Wraps a fine-tuned model for production inference.

    Sentiment score = P(positive) - P(negative)  →  [-1.0, +1.0]
    A score close to +1 is strongly bullish; close to -1 is strongly bearish.

    Usage:
        analyzer = SentimentAnalyzer("./models/premium")
        result = analyzer.analyze("Apple beats earnings expectations by 15%")
        # → {"label": "positive", "score": 0.87, "probabilities": {...}}
    """

    def __init__(self, model_dir: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True
            )

        self.model = torch.quantization.quantize_dynamic(
        self.model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

        self.model.to(self.device)
        self.model.eval()
        log.info("SentimentAnalyzer loaded from '%s' on %s", model_dir, self.device)

    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()

        p_negative = float(probs[LABEL2ID["negative"]])
        p_neutral  = float(probs[LABEL2ID["neutral"]])
        p_positive = float(probs[LABEL2ID["positive"]])

        sentiment_score = round(p_positive - p_negative, 4)
        predicted_label = ID2LABEL[int(np.argmax(probs))]

        return {
            "label": predicted_label,
            "score": sentiment_score,           # Core API field: [-1, +1]
            "probabilities": {
                "negative": round(p_negative, 4),
                "neutral":  round(p_neutral,  4),
                "positive": round(p_positive, 4),
            },
        }

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Batch inference for Premium / Pro tiers (multiple headlines at once)."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs_batch = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for probs in probs_batch:
                p_neg = float(probs[LABEL2ID["negative"]])
                p_neu = float(probs[LABEL2ID["neutral"]])
                p_pos = float(probs[LABEL2ID["positive"]])
                results.append({
                    "label": ID2LABEL[int(np.argmax(probs))],
                    "score": round(p_pos - p_neg, 4),
                    "probabilities": {
                        "negative": round(p_neg, 4),
                        "neutral":  round(p_neu, 4),
                        "positive": round(p_pos, 4),
                    },
                })
        return results

FREE_MODEL = None

def get_free_model():
    global FREE_MODEL
    if FREE_MODEL is None:
        FREE_MODEL = SentimentAnalyzer(model_dir='lokajko3/news_classification_free')
    return FREE_MODEL

def predict_sentiment_scores(
    texts: Iterable[str],
    tier: str = 'free',
    model_path: str | Path | None = None,
) -> tuple[float, list[dict]]:
    """Placeholder function to predict sentiment scores for a list of texts."""
    return 0.0,[]
