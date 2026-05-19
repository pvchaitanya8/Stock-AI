import logging
from fastapi import APIRouter, HTTPException

from app.core.data import fetch_ohlcv, invalidate
from app.core.features import compute_features
from app.core.trainer import train as train_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/train/{ticker}")
def train(ticker: str, refresh: bool = False):
    """Train (or retrain) the forecasting model for a ticker."""
    ticker = ticker.upper()
    try:
        if refresh:
            invalidate(ticker)
        df = fetch_ohlcv(ticker)
        feat = compute_features(df)
        metrics = train_model(ticker, feat)
        return {"status": "ok", **metrics}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Training failed for %s", ticker)
        raise HTTPException(status_code=500, detail=f"Training error: {e}")
