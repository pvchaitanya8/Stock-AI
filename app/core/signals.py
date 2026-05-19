"""
Composite Buy / Hold / Sell signal engine.

Combines the model's 30-day forecast with the current technical-indicator
snapshot to produce an actionable signal plus a human-readable list of the
reasons behind it.
"""

LABELS = {
    3:  "STRONG BUY",
    2:  "BUY",
    1:  "BUY",
    0:  "HOLD",
    -1: "SELL",
    -2: "SELL",
    -3: "STRONG SELL",
}


def get_signal(current_price: float, forecast: list[dict], indicators: dict) -> dict:
    """
    Parameters
    ----------
    current_price : latest close from data layer
    forecast      : list of {date, low, median, high} from trainer.predict
    indicators    : dict from features.get_indicator_snapshot

    Returns
    -------
    {"signal": str, "score": int, "reasons": [str], "predicted_return_pct": float}
    """
    score   = 0
    reasons = []

    # 1 ─ Predicted 30-day return (the heaviest weight)
    final_pred = forecast[-1]["median"]
    pct = ((final_pred - current_price) / current_price) * 100

    if pct > 3:
        score += 2
        reasons.append(f"Predicted 30-day return of +{pct:.2f}% — strong upside.")
    elif pct > 0:
        score += 1
        reasons.append(f"Predicted 30-day return of +{pct:.2f}% — mild upside.")
    elif pct < -3:
        score -= 2
        reasons.append(f"Predicted 30-day return of {pct:.2f}% — strong downside.")
    elif pct < 0:
        score -= 1
        reasons.append(f"Predicted 30-day return of {pct:.2f}% — mild downside.")
    else:
        reasons.append("Forecast roughly flat over 30 days.")

    # 2 ─ RSI (oversold = bullish, overbought = bearish)
    rsi = indicators["rsi"]
    if rsi < 35:
        score += 1
        reasons.append(f"RSI at {rsi} — oversold territory (bullish).")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI at {rsi} — overbought territory (bearish).")

    # 3 ─ MACD cross
    cross = indicators["macd_cross"]
    if cross == "bullish":
        score += 1
        reasons.append("MACD bullish crossover detected.")
    elif cross == "bearish":
        score -= 1
        reasons.append("MACD bearish crossover detected.")

    # 4 ─ Bollinger Bands (%B position)
    pband = indicators["bb_pband"]
    if pband < 0.05:
        score += 1
        reasons.append("Price near/below lower Bollinger Band (bullish).")
    elif pband > 0.95:
        score -= 1
        reasons.append("Price near/above upper Bollinger Band (bearish).")

    # 5 ─ Stochastic
    stoch_k = indicators["stoch_k"]
    if stoch_k < 20:
        score += 1
        reasons.append(f"Stochastic %K at {stoch_k} — oversold (bullish).")
    elif stoch_k > 80:
        score -= 1
        reasons.append(f"Stochastic %K at {stoch_k} — overbought (bearish).")

    # 6 ─ Trend confirmation: price vs EMA-50 vs EMA-200
    close = indicators["close"]
    if close > indicators["ema_50"] > indicators["ema_200"]:
        reasons.append("Uptrend confirmed: price > EMA-50 > EMA-200.")
    elif close < indicators["ema_50"] < indicators["ema_200"]:
        reasons.append("Downtrend confirmed: price < EMA-50 < EMA-200.")

    # Map composite score → label
    label = LABELS.get(max(min(score, 3), -3), "HOLD")

    return {
        "signal":               label,
        "score":                score,
        "reasons":              reasons,
        "predicted_return_pct": round(pct, 2),
    }
