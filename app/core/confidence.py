from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

TECH_WEIGHT = 0.3
NEWS_WEIGHT = 0.3
FUND_WEIGHT = 0.4


def _clip01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, value))


def _weighted_avg(values: list[float], weights: list[float]) -> float:
    total_weight = 0.0
    total = 0.0

    for value, weight in zip(values, weights):
        total += float(value) * float(weight)
        total_weight += float(weight)

    if total_weight == 0.0:
        return 0.0

    return total / total_weight


def _weighted_std(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return 0.0

    mean = _weighted_avg(values, weights)
    total_weight = 0.0
    variance = 0.0

    for value, weight in zip(values, weights):
        variance += float(weight) * ((float(value) - mean) ** 2)
        total_weight += float(weight)

    if total_weight == 0.0:
        return 0.0

    return float((variance / total_weight) ** 0.5)


def _score_direction(score: float | None, threshold: float = 0.05) -> int:
    try:
        score = float(score)
    except Exception:
        return 0

    if score >= threshold:
        return 1
    if score <= -threshold:
        return -1
    return 0


def _statement_present(df: pd.DataFrame | None) -> bool:
    return df is not None and hasattr(df, "empty") and not df.empty


def _latest_statement_date(dfs: list[pd.DataFrame | None]) -> pd.Timestamp | None:
    latest = None
    for df in dfs:
        if not _statement_present(df):
            continue
        dates = pd.to_datetime(df.columns, errors="coerce")
        if dates.isna().all():
            continue
        current_max = dates.max()
        if latest is None or current_max > latest:
            latest = current_max
    return latest


def compute_interval_technical_confidence(df: pd.DataFrame, interval_score: float) -> float:
    required_cols = [
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "EMA_50",
        "EMA_100",
        "EMA_200",
        "BBL",
        "BBM",
        "BBU",
        "DL",
        "DU",
        "ATR",
        "ADX",
        "Stoch",
    ]
    available_cols = [col for col in required_cols if col in df.columns]
    if not available_cols:
        return 0.1

    total_values = len(df) * len(available_cols)
    if total_values == 0:
        return 0.1

    missing = df[available_cols].isna().sum().sum()
    coverage = max(0.0, 1.0 - (missing / total_values))
    signal_strength = _clip01(abs(interval_score))

    agreement = 0.0
    votes = 0
    last = df.iloc[-1]

    if "RSI" in available_cols and not pd.isna(last["RSI"]):
        votes += 1
        if last["RSI"] <= 30 or last["RSI"] >= 70:
            agreement += 1.0

    if "MACD_Hist" in available_cols and not pd.isna(last["MACD_Hist"]):
        votes += 1
        if last["MACD_Hist"] > 0 or last["MACD_Hist"] < 0:
            agreement += 1.0

    if all(col in available_cols for col in ["EMA_50", "EMA_100", "EMA_200"]):
        if not (pd.isna(last["EMA_50"]) or pd.isna(last["EMA_100"]) or pd.isna(last["EMA_200"])):
            votes += 1
            if last["EMA_50"] > last["EMA_100"] > last["EMA_200"]:
                agreement += 1.0
            elif last["EMA_50"] < last["EMA_100"] < last["EMA_200"]:
                agreement += 1.0

    if "Close" in df.columns and all(col in available_cols for col in ["BBL", "BBU"]):
        price = last.get("Close")
        lower_band = last.get("BBL")
        upper_band = last.get("BBU")
        if not (pd.isna(price) or pd.isna(lower_band) or pd.isna(upper_band)):
            votes += 1
            if price < lower_band or price > upper_band:
                agreement += 1.0

    if "Close" in df.columns and all(col in available_cols for col in ["DL", "DU"]):
        price = last.get("Close")
        lower_channel = last.get("DL")
        upper_channel = last.get("DU")
        if not (pd.isna(price) or pd.isna(lower_channel) or pd.isna(upper_channel)):
            votes += 1
            if price < lower_channel or price > upper_channel:
                agreement += 1.0

    if "Stoch" in available_cols and not pd.isna(last["Stoch"]):
        votes += 1
        if last["Stoch"] <= 20 or last["Stoch"] >= 80:
            agreement += 1.0

    if "ADX" in available_cols and not pd.isna(last["ADX"]):
        votes += 1
        if last["ADX"] >= 20:
            agreement += 1.0

    agreement = agreement / votes if votes > 0 else 0.0

    volatility = 1.0
    if "Close" in df.columns and len(df) > 2:
        returns = df["Close"].pct_change().dropna()
        if len(returns) > 0:
            vol = returns.std()
            vol_ref = 0.03
            volatility = max(0.0, 1.0 - min(1.0, vol / vol_ref))

    confidence = (
        0.10
        + 0.35 * signal_strength
        + 0.25 * agreement
        + 0.20 * coverage
        + 0.10 * volatility
    )
    return _clip01(confidence)


def compute_technical_confidence_breakdown(
    dfs: list[pd.DataFrame],
    tech_score: float,
    tech_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not dfs:
        return {
            "Overall": 0.1,
            "Intervals": {},
            "Drivers": {
                "Signal Strength": _clip01(abs(tech_score)),
                "Interval Confidence": 0.0,
                "Timeframe Breadth": 0.0,
                "Cross-Interval Agreement": 0.5,
                "Consistency": 0.0,
            },
        }

    interval_specs = [
        ("1 day", 0, 8.0),
        ("1 min", 1, 1.0),
        ("5 min", 2, 2.0),
        ("15 min", 3, 3.0),
        ("1 hour", 4, 5.0),
        ("1 week", 5, 13.0),
    ]

    payload_intervals = {}
    if isinstance(tech_payload, dict):
        maybe_intervals = tech_payload.get("Intervals")
        if isinstance(maybe_intervals, dict):
            payload_intervals = maybe_intervals

    interval_confidences: list[float] = []
    interval_weights: list[float] = []
    interval_scores: list[float] = []
    score_weights: list[float] = []
    interval_breakdown: dict[str, float] = {}

    available_weight = 0.0
    total_possible_weight = sum(weight for _, _, weight in interval_specs)

    for interval_name, index, weight in interval_specs:
        if index >= len(dfs):
            continue

        df = dfs[index]
        if df is None or getattr(df, "empty", True):
            continue

        available_weight += weight
        interval_payload = payload_intervals.get(interval_name, {})
        interval_score = tech_score
        if isinstance(interval_payload, dict):
            interval_score = interval_payload.get("Score", tech_score)

        interval_confidence = compute_interval_technical_confidence(df, interval_score)
        interval_confidences.append(interval_confidence)
        interval_weights.append(weight)
        interval_breakdown[interval_name] = interval_confidence

        try:
            interval_scores.append(float(interval_score))
            score_weights.append(weight)
        except Exception:
            continue

    if not interval_confidences:
        return {
            "Overall": 0.1,
            "Intervals": {},
            "Drivers": {
                "Signal Strength": _clip01(abs(tech_score)),
                "Interval Confidence": 0.0,
                "Timeframe Breadth": 0.0,
                "Cross-Interval Agreement": 0.5,
                "Consistency": 0.0,
            },
        }

    interval_confidence = _weighted_avg(interval_confidences, interval_weights)
    timeframe_breadth = available_weight / total_possible_weight if total_possible_weight > 0 else 0.0

    overall_direction = _score_direction(tech_score, threshold=0.03)
    if overall_direction == 0 and interval_scores:
        overall_direction = _score_direction(_weighted_avg(interval_scores, score_weights), threshold=0.03)

    decisive_weight = 0.0
    agreeing_weight = 0.0
    for interval_score, weight in zip(interval_scores, score_weights):
        direction = _score_direction(interval_score, threshold=0.08)
        if direction == 0:
            continue
        decisive_weight += weight
        if overall_direction == 0 or direction == overall_direction:
            agreeing_weight += weight

    cross_interval_agreement = (agreeing_weight / decisive_weight) if decisive_weight > 0 else 0.5

    consistency = 1.0
    if len(interval_scores) >= 2:
        dispersion = _weighted_std(interval_scores, score_weights)
        consistency = max(0.0, 1.0 - min(1.0, dispersion / 0.6))

    signal_strength = _clip01(abs(tech_score))
    confidence = (
        0.05
        + 0.20 * signal_strength
        + 0.35 * interval_confidence
        + 0.15 * cross_interval_agreement
        + 0.15 * timeframe_breadth
        + 0.10 * consistency
    )

    return {
        "Overall": _clip01(confidence),
        "Intervals": interval_breakdown,
        "Drivers": {
            "Signal Strength": signal_strength,
            "Interval Confidence": interval_confidence,
            "Timeframe Breadth": timeframe_breadth,
            "Cross-Interval Agreement": cross_interval_agreement,
            "Consistency": consistency,
        },
    }


def compute_fundamental_confidence(
    ticker: str | None = None,
    ticker_obj: yf.Ticker | None = None,
    fund_score: float | None = None,
    financials: pd.DataFrame | None = None,
    balance_sheet: pd.DataFrame | None = None,
    cash_flow: pd.DataFrame | None = None,
) -> float:
    if ticker_obj is None:
        if not ticker:
            return 0.1
        ticker_obj = yf.Ticker(ticker)

    if financials is None:
        financials = ticker_obj.financials
    if balance_sheet is None:
        balance_sheet = ticker_obj.balance_sheet
    if cash_flow is None:
        cash_flow = ticker_obj.cash_flow

    statements_present = (
        int(_statement_present(financials))
        + int(_statement_present(balance_sheet))
        + int(_statement_present(cash_flow))
    ) / 3.0

    total = 0
    available = 0

    if _statement_present(financials):
        needed_rows = ["Total Revenue", "Net Income", "Gross Profit", "Operating Income", "Basic EPS"]
        total += len(needed_rows)
        available += sum(1 for row in needed_rows if row in financials.index)

    if _statement_present(balance_sheet):
        total += 2
        if "Stockholders Equity" in balance_sheet.index:
            available += 1
        has_total_debt = "Total Debt" in balance_sheet.index
        has_split_debt = "Long Term Debt" in balance_sheet.index and "Short Term Debt" in balance_sheet.index
        if has_total_debt or has_split_debt:
            available += 1

    if _statement_present(cash_flow):
        total += 1
        if "Total Cash From Operating Activities" in cash_flow.index:
            available += 1

    coverage = (available / total) if total > 0 else 0.0

    recency_score = 0.5
    latest = _latest_statement_date([financials, balance_sheet, cash_flow])
    if latest is not None:
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")
        days_old = (pd.Timestamp.now(tz="UTC") - latest).days
        if days_old <= 365:
            recency_score = 1.0
        elif days_old <= 730:
            recency_score = 0.7
        elif days_old <= 1095:
            recency_score = 0.4
        else:
            recency_score = 0.2

    signal_strength = _clip01(abs(fund_score)) if fund_score is not None else 0.0
    confidence = (
        0.10
        + 0.35 * signal_strength
        + 0.25 * coverage
        + 0.20 * statements_present
        + 0.10 * recency_score
    )
    return _clip01(confidence)


def compute_news_confidence(
    news_score: float | None = None,
    has_data: bool = False,
    article_count: int | None = None,
    distinct_sources: int | None = None,
    latest_pub_date: str | None = None,
) -> float:
    if not has_data:
        return 0.0

    signal_strength = _clip01(abs(news_score)) if news_score is not None else 0.0

    volume_score = 0.5
    if article_count is not None:
        try:
            volume_score = _clip01(max(0, int(article_count)) / 8.0)
        except Exception:
            volume_score = 0.5

    source_score = 0.5
    if distinct_sources is not None:
        try:
            source_score = _clip01(max(0, int(distinct_sources)) / 4.0)
        except Exception:
            source_score = 0.5

    freshness_score = 0.5
    if latest_pub_date is not None:
        ts = pd.to_datetime(latest_pub_date, errors="coerce", utc=True)
        if not pd.isna(ts):
            now = pd.Timestamp.now(tz="UTC")
            age_hours = max(0.0, (now - ts).total_seconds() / 3600.0)
            if age_hours <= 12:
                freshness_score = 1.0
            elif age_hours <= 24:
                freshness_score = 0.9
            elif age_hours <= 72:
                freshness_score = 0.75
            elif age_hours <= 168:
                freshness_score = 0.5
            elif age_hours <= 336:
                freshness_score = 0.3
            else:
                freshness_score = 0.15

    confidence = (
        0.05
        + 0.35 * signal_strength
        + 0.30 * volume_score
        + 0.15 * source_score
        + 0.15 * freshness_score
    )
    return _clip01(confidence)


def compute_overall_confidence(components: dict[str, float | None]) -> float:
    weights = {
        "technical": TECH_WEIGHT,
        "news": NEWS_WEIGHT,
        "fundamentals": FUND_WEIGHT,
    }

    total_weight = 0.0
    total = 0.0
    for key, confidence in components.items():
        if confidence is None:
            continue
        weight = weights.get(key, 0.0)
        if weight <= 0:
            continue
        total_weight += weight
        total += weight * _clip01(confidence)

    if total_weight == 0.0:
        return 0.0

    return _clip01(total / total_weight)
