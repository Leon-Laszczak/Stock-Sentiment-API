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
FREE_MODEL_DIR = 'Leon-Laszczak/news_classification_free_quantized'
torch.set_num_threads(1)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class FastSentimentAnalyzer:
    def __init__(self, model_dir=FREE_MODEL_DIR):
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_dir,
            file_name="model_quantized.onnx"
        )

        import json
        import os

        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.label2id = {v: k for k, v in self.id2label.items()}

        log.info("Model loaded (ONNX INT8)")

    def analyze(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="np",  # 🔥 NO TORCH
            truncation=True,
            max_length=128,
            padding="max_length"
        )

        logits = self.model(**inputs).logits[0]
        probs = softmax(logits)
        p_negative = float(probs[self.label2id["negative"]])
        p_neutral = float(probs[self.label2id["neutral"]])
        p_positive = float(probs[self.label2id["positive"]])

        return {
            "label": self.id2label[int(np.argmax(probs))],
            "score": round(p_positive - p_negative, 4),
            "probabilities": {
                "negative": round(p_negative, 4),
                "neutral": round(p_neutral, 4),
                "positive": round(p_positive, 4),
            }
        }

    def analyze_batch(self, texts, batch_size=32):
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="np",
                truncation=True,
                max_length=128,
                padding="max_length"
            )

            logits_batch = self.model(**inputs).logits

            for logits in logits_batch:
                probs = softmax(logits)
                p_negative = float(probs[self.label2id["negative"]])
                p_neutral  = float(probs[self.label2id["neutral"]])
                p_positive = float(probs[self.label2id["positive"]])

                sentiment_score = round(p_positive - p_negative, 4)
                predicted_label = self.id2label[int(np.argmax(probs))]

                results.append({
                    "label": predicted_label,
                    "score": sentiment_score,           # Core API field: [-1, +1]
                    "probabilities": {
                        "negative": round(p_negative, 4),
                        "neutral":  round(p_neutral,  4),
                        "positive": round(p_positive, 4),
                    },
                })

        return results

FREE_MODEL = None

def get_free_model():
    global FREE_MODEL
    if FREE_MODEL is None:
        FREE_MODEL = FastSentimentAnalyzer()
    return FREE_MODEL

def predict_sentiment_scores(
    texts: Iterable[str],
    tier: str = 'free',
) -> tuple[float, list[dict]]:
    """Predict sentiment scores for a list of texts using the specified model tier."""
    try:
        if tier == 'free':
            model = get_free_model()
        else:
            raise ValueError(f"Unsupported tier: {tier}")

        if isinstance(texts, str):
            texts = [texts]

        results = model.analyze_batch(list(texts))
        scores = [res["score"] for res in results]
        avg_score = round(float(np.mean(scores)), 4)

        return avg_score, results
    except Exception as e:
        return 0.0, [{"error": str(e)}]
