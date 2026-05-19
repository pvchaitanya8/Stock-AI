import logging
from fastapi import APIRouter, HTTPException

from app.core.data import fetch_ohlcv, get_current_price
from app.core.features import compute_features, get_indicator_snapshot
from app.core.signals import get_signal
from app.core.trainer import predict as predict_model, is_trained

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/predict/{ticker}")
def predict(ticker: str):
    """Return 30-day forecast, current indicators, and Buy/Hold/Sell signal."""
    ticker = ticker.upper()

    if not is_trained(ticker):
        raise HTTPException(
            status_code=409,
            detail=f"No trained model for '{ticker}'. POST /api/train/{ticker} first.",
        )

    try:
        df       = fetch_ohlcv(ticker)
        feat     = compute_features(df)
        forecast = predict_model(ticker, feat)
        snap     = get_indicator_snapshot(feat)
        price    = get_current_price(ticker)
        signal   = get_signal(price, forecast, snap)

        return {
            "ticker":        ticker,
            "current_price": round(price, 2),
            "last_date":     str(feat.index[-1].date()),
            "forecast":      forecast,
            "indicators":    snap,
            "signal":        signal,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
