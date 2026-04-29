"""Local news sentiment scoring with a Qwen3 GGUF primary model and FinBERT fallback."""

from __future__ import annotations

import logging as log
import os
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MAX_LENGTH = 256
DEFAULT_PRIMARY_MODEL_ID = "Ayansk11/FinSenti-Qwen3-8B-GGUF"
DEFAULT_PRIMARY_MODEL_FILENAME = "FinSenti-Qwen3-8B.Q4_K_M.gguf"
DEFAULT_FALLBACK_MODEL_ID = "ProsusAI/finbert"
WARMUP_TEXT = "Stocks are stable today."
QWEN_MAX_TOKENS = 160
DEFAULT_QWEN_LOCAL_DIRS = (
    "finsenti-qwen3-8b-gguf",
    "news_sentiment_qwen3_8b_gguf",
    "qwen3_news_sentiment",
)
DEFAULT_FINBERT_LOCAL_DIRS = (
    "finbert",
    "news_sentiment_finbert",
)
SCORE_BY_LABEL = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}
QWEN_SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. For each headline you receive, "
    "write a short reasoning chain inside <reasoning>...</reasoning> tags, "
    "then give a single label inside <answer>...</answer> tags. The label "
    "must be exactly one of: positive, negative, neutral."
)
QWEN_ANSWER_PATTERN = re.compile(
    r"<answer>\s*(positive|negative|neutral)\s*</answer>",
    re.IGNORECASE,
)
LABEL_PATTERN = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)
UNAVAILABLE_PRIMARY_MODELS: set[str] = set()


def _softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _local_model_root() -> Path:
    return _project_root() / "models"


def _resolve_local_dir(candidates: tuple[str, ...]) -> str | None:
    model_root = _local_model_root()
    for candidate in candidates:
        path = model_root / candidate
        if path.exists():
            return str(path)
    return None


def _resolve_primary_model_source(model_path: str | Path | None = None) -> str:
    if model_path is not None:
        return str(model_path)

    configured_path = os.getenv("NEWS_SENTIMENT_MODEL_PATH") or os.getenv(
        "NEWS_SENTIMENT_PRIMARY_MODEL_PATH"
    )
    if configured_path:
        return configured_path

    local_dir = _resolve_local_dir(DEFAULT_QWEN_LOCAL_DIRS)
    if local_dir:
        return local_dir

    direct_file = _local_model_root() / DEFAULT_PRIMARY_MODEL_FILENAME
    if direct_file.exists():
        return str(direct_file)

    return os.getenv("NEWS_SENTIMENT_PRIMARY_MODEL_ID", DEFAULT_PRIMARY_MODEL_ID)


def _resolve_fallback_model_source() -> str:
    configured_path = os.getenv("NEWS_SENTIMENT_FALLBACK_MODEL_PATH")
    if configured_path:
        return configured_path

    local_dir = _resolve_local_dir(DEFAULT_FINBERT_LOCAL_DIRS)
    if local_dir:
        return local_dir

    return os.getenv("NEWS_SENTIMENT_FALLBACK_MODEL_ID", DEFAULT_FALLBACK_MODEL_ID)


def _result_from_label(label: str, model_name: str) -> dict:
    label = _normalize_label(label)
    probabilities = {
        "negative": 0.0,
        "neutral": 0.0,
        "positive": 0.0,
    }
    probabilities[label] = 1.0
    return {
        "label": label,
        "score": SCORE_BY_LABEL[label],
        "probabilities": probabilities,
        "model": model_name,
    }


def _result_from_probabilities(probabilities: dict[str, float], model_name: str) -> dict:
    rounded = {
        "negative": round(float(probabilities["negative"]), 4),
        "neutral": round(float(probabilities["neutral"]), 4),
        "positive": round(float(probabilities["positive"]), 4),
    }
    label = max(rounded, key=rounded.get)
    return {
        "label": label,
        "score": round(rounded["positive"] - rounded["negative"], 4),
        "probabilities": rounded,
        "model": model_name,
    }


def _extract_qwen_label(output_text: str) -> str:
    answer_match = QWEN_ANSWER_PATTERN.search(output_text)
    if answer_match:
        return _normalize_label(answer_match.group(1))

    loose_matches = LABEL_PATTERN.findall(output_text)
    if loose_matches:
        return _normalize_label(loose_matches[-1])

    raise ValueError(f"Could not parse sentiment label from Qwen output: {output_text!r}")


class QwenGgufSentimentAnalyzer:
    """Loads a local or Hugging Face GGUF file and runs local inference via llama.cpp."""

    def __init__(self, model_source: str):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        self.model_name = model_source
        self.model_path = self._resolve_model_path(
            model_source=model_source,
            hf_hub_download=hf_hub_download,
        )
        self._lock = Lock()
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=int(os.getenv("NEWS_SENTIMENT_QWEN_CTX", "2048")),
            n_threads=int(os.getenv("NEWS_SENTIMENT_QWEN_THREADS", str(os.cpu_count() or 4))),
            n_gpu_layers=int(os.getenv("NEWS_SENTIMENT_QWEN_GPU_LAYERS", "0")),
            verbose=False,
        )
        log.info("Loaded Qwen GGUF news sentiment model from %s", self.model_path)

    def _resolve_model_path(self, model_source: str, hf_hub_download) -> str:
        requested_path = Path(model_source)
        gguf_filename = os.getenv("NEWS_SENTIMENT_PRIMARY_MODEL_FILENAME", DEFAULT_PRIMARY_MODEL_FILENAME)

        if requested_path.is_file():
            return str(requested_path)

        if requested_path.is_dir():
            direct_candidate = requested_path / gguf_filename
            if direct_candidate.exists():
                return str(direct_candidate)

            gguf_candidates = sorted(requested_path.glob("*.gguf"))
            if gguf_candidates:
                return str(gguf_candidates[0])

            raise FileNotFoundError(f"No .gguf file found inside {requested_path}")

        if requested_path.suffix.lower() == ".gguf":
            raise FileNotFoundError(f"Configured GGUF file does not exist: {requested_path}")

        download_dir = os.getenv("NEWS_SENTIMENT_PRIMARY_DOWNLOAD_DIR")
        return hf_hub_download(
            repo_id=model_source,
            filename=gguf_filename,
            local_dir=download_dir or None,
        )

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        del batch_size
        results: list[dict] = []

        for text in texts:
            response = self._create_chat_completion(text)
            label = _extract_qwen_label(response)
            results.append(_result_from_label(label=label, model_name=self.model_name))

        return results

    def _create_chat_completion(self, text: str) -> str:
        with self._lock:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=int(os.getenv("NEWS_SENTIMENT_QWEN_MAX_TOKENS", str(QWEN_MAX_TOKENS))),
                temperature=0.0,
            )

        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen model returned an empty response.")
        return content


class FinBertSentimentAnalyzer:
    """Loads FinBERT and runs standard sequence classification inference."""

    def __init__(self, model_source: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_source
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self._resolve_id2label()
        self.label2id = {value: key for key, value in self.id2label.items()}
        log.info("Loaded FinBERT fallback sentiment model from %s", model_source)

    def _resolve_id2label(self) -> dict[int, str]:
        raw_id2label = getattr(self.model.config, "id2label", {}) or {}
        resolved: dict[int, str] = {}

        for key, value in raw_id2label.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue

            label = _normalize_label(str(value))
            if label in SCORE_BY_LABEL:
                resolved[idx] = label

        if set(resolved.values()) == set(SCORE_BY_LABEL):
            return resolved

        return {
            0: "positive",
            1: "negative",
            2: "neutral",
        }

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        import torch

        results: list[dict] = []

        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.inference_mode():
                logits_batch = self.model(**inputs).logits.detach().cpu().numpy()

            probabilities_batch = _softmax(logits_batch)
            for probabilities in probabilities_batch:
                resolved_probabilities = {
                    "negative": float(probabilities[self.label2id["negative"]]),
                    "neutral": float(probabilities[self.label2id["neutral"]]),
                    "positive": float(probabilities[self.label2id["positive"]]),
                }
                results.append(
                    _result_from_probabilities(
                        probabilities=resolved_probabilities,
                        model_name=self.model_name,
                    )
                )

        return results


@lru_cache(maxsize=4)
def _get_fallback_analyzer(model_source: str) -> FinBertSentimentAnalyzer:
    return FinBertSentimentAnalyzer(model_source=model_source)


def _mark_primary_model_unavailable(model_source: str) -> None:
    UNAVAILABLE_PRIMARY_MODELS.add(model_source)
    _get_preferred_analyzer.cache_clear()


@lru_cache(maxsize=4)
def _get_preferred_analyzer(
    primary_model_source: str,
    fallback_model_source: str,
) -> QwenGgufSentimentAnalyzer | FinBertSentimentAnalyzer:
    if primary_model_source in UNAVAILABLE_PRIMARY_MODELS:
        return _get_fallback_analyzer(fallback_model_source)

    try:
        return QwenGgufSentimentAnalyzer(model_source=primary_model_source)
    except Exception as exc:
        _mark_primary_model_unavailable(primary_model_source)
        log.warning(
            "Could not load primary Qwen sentiment model from %s: %s. Falling back to %s.",
            primary_model_source,
            exc,
            fallback_model_source,
        )
        return _get_fallback_analyzer(fallback_model_source)


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

        primary_model_source = _resolve_primary_model_source(model_path=model_path)
        fallback_model_source = _resolve_fallback_model_source()
        analyzer = _get_preferred_analyzer(primary_model_source, fallback_model_source)

        try:
            results = analyzer.analyze_batch(text_list)
        except Exception:
            if isinstance(analyzer, QwenGgufSentimentAnalyzer):
                _mark_primary_model_unavailable(primary_model_source)
                log.warning(
                    "Primary Qwen sentiment inference failed for %s. Retrying with fallback model %s.",
                    primary_model_source,
                    fallback_model_source,
                    exc_info=True,
                )
                results = _get_fallback_analyzer(fallback_model_source).analyze_batch(text_list)
            else:
                raise

        scores = [result["score"] for result in results]
        avg_score = round(float(np.mean(scores)), 4) if scores else 0.0
        return avg_score, results
    except Exception as exc:
        return 0.0, [{"error": str(exc)}]


def warm_up_sentiment_model(model_path: str | Path | None = None) -> str:
    """Load the analyzer eagerly so the first request does not pay model startup cost."""
    primary_model_source = _resolve_primary_model_source(model_path=model_path)
    fallback_model_source = _resolve_fallback_model_source()
    analyzer = _get_preferred_analyzer(primary_model_source, fallback_model_source)

    try:
        analyzer.analyze_batch([WARMUP_TEXT], batch_size=1)
        return analyzer.model_name
    except Exception:
        if isinstance(analyzer, QwenGgufSentimentAnalyzer):
            _mark_primary_model_unavailable(primary_model_source)
            log.warning(
                "Primary Qwen warmup failed for %s. Switching to fallback model %s.",
                primary_model_source,
                fallback_model_source,
                exc_info=True,
            )
            fallback = _get_fallback_analyzer(fallback_model_source)
            fallback.analyze_batch([WARMUP_TEXT], batch_size=1)
            return fallback.model_name
        raise
