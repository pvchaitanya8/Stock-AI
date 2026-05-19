import logging
from fastapi import APIRouter, HTTPException

from app.core.data import fetch_ohlcv
from app.core.features import compute_features

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/indicators/{ticker}")
def indicators(ticker: str, days: int = 180):
    """
    Return the last `days` rows of OHLCV + every technical indicator
    so the frontend can render full historical charts.
    """
    ticker = ticker.upper()
    try:
        df   = fetch_ohlcv(ticker)
        feat = compute_features(df).tail(days)

        return {
            "ticker": ticker,
            "history": [
                {
                    "date":    str(idx.date()),
                    "open":    round(float(row["Open"]),    2),
                    "high":    round(float(row["High"]),    2),
                    "low":     round(float(row["Low"]),     2),
                    "close":   round(float(row["Close"]),   2),
                    "volume":  int(row["Volume"]),
                    "rsi":         round(float(row["rsi"]),         2),
                    "macd":        round(float(row["macd"]),        4),
                    "macd_signal": round(float(row["macd_signal"]), 4),
                    "macd_diff":   round(float(row["macd_diff"]),   4),
                    "bb_upper":    round(float(row["bb_upper"]),    2),
                    "bb_lower":    round(float(row["bb_lower"]),    2),
                    "bb_mid":      round(float(row["bb_mid"]),      2),
                    "ema_20":      round(float(row["ema_20"]),      2),
                    "ema_50":      round(float(row["ema_50"]),      2),
                    "ema_200":     round(float(row["ema_200"]),     2),
                    "stoch_k":     round(float(row["stoch_k"]),     2),
                    "stoch_d":     round(float(row["stoch_d"]),     2),
                    "atr":         round(float(row["atr"]),         4),
                    "obv":         int(row["obv"]),
                }
                for idx, row in feat.iterrows()
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Indicators failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Indicators error: {e}")
