"""Local FinBERT-based utilities for news sentiment scoring."""

from __future__ import annotations

import json
import logging as log
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MAX_LENGTH = 128
DEFAULT_MODEL_ID = "Leon-Laszczak/news_classification_free_quantized"
WARMUP_TEXT = "Stocks are stable today."


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_model_dir(model_path: str | Path | None = None) -> str:
    if model_path is not None:
        return str(model_path)

    configured_path = os.getenv("NEWS_SENTIMENT_MODEL_PATH")
    if configured_path:
        return configured_path

    local_model_dir = _project_root() / "models" / "news_classification_free_quantized"
    if local_model_dir.exists():
        return str(local_model_dir)

    return DEFAULT_MODEL_ID


class FastSentimentAnalyzer:
    """Loads the quantized news model and runs local batch inference."""

    def __init__(self, model_dir: str):
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_dir,
            file_name="model_quantized.onnx",
        )

        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
            self.id2label = {int(k): v for k, v in config["id2label"].items()}
        else:
            self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

        self.label2id = {value: key for key, value in self.id2label.items()}
        log.info("Loaded local news sentiment model from %s", model_dir)

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results: list[dict] = []

        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="np",
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
            )

            logits_batch = self.model(**inputs).logits

            for logits in logits_batch:
                probs = _softmax(logits)
                p_negative = float(probs[self.label2id["negative"]])
                p_neutral = float(probs[self.label2id["neutral"]])
                p_positive = float(probs[self.label2id["positive"]])

                results.append(
                    {
                        "label": self.id2label[int(np.argmax(probs))],
                        "score": round(p_positive - p_negative, 4),
                        "probabilities": {
                            "negative": round(p_negative, 4),
                            "neutral": round(p_neutral, 4),
                            "positive": round(p_positive, 4),
                        },
                    }
                )

        return results


@lru_cache(maxsize=4)
def _get_analyzer(model_dir: str) -> FastSentimentAnalyzer:
    return FastSentimentAnalyzer(model_dir=model_dir)


def predict_sentiment_scores(
    texts: Iterable[str],
    tier: str = "free",
    model_path: str | Path | None = None,
) -> tuple[float, list[dict]]:
    """Predict sentiment scores locally and return the mean score with per-text details."""
    try:
        if tier != "free":
            raise ValueError(f"Unsupported tier: {tier}")

        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = [str(text).strip() for text in texts if str(text).strip()]

        if not text_list:
            return 0.0, []

        resolved_model_dir = _resolve_model_dir(model_path=model_path)
        analyzer = _get_analyzer(resolved_model_dir)
        results = analyzer.analyze_batch(text_list)
        scores = [result["score"] for result in results]
        avg_score = round(float(np.mean(scores)), 4) if scores else 0.0

        return avg_score, results
    except Exception as exc:
        return 0.0, [{"error": str(exc)}]


def warm_up_sentiment_model(model_path: str | Path | None = None) -> str:
    """Load the analyzer eagerly so the first request does not pay model startup cost."""
    resolved_model_dir = _resolve_model_dir(model_path=model_path)
    analyzer = _get_analyzer(resolved_model_dir)
    analyzer.analyze_batch([WARMUP_TEXT], batch_size=1)
    return resolved_model_dir
