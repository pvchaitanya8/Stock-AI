import pickle
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

_memory: dict[str, pd.DataFrame] = {}


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.pkl"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload.get("date") == str(date.today())


def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Return daily OHLCV DataFrame for ticker, using cache where possible."""
    ticker = ticker.upper()

    if ticker in _memory:
        return _memory[ticker]

    path = _cache_path(ticker)
    if _is_fresh(path):
        with open(path, "rb") as f:
            df = pickle.load(f)["data"]
        _memory[ticker] = df
        logger.info("Loaded %s from cache.", ticker)
        return df

    logger.info("Downloading %s from yfinance...", ticker)
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data found for '{ticker}'. Verify the symbol.")

    # yfinance may return MultiIndex columns for a single ticker in newer versions
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    with open(path, "wb") as f:
        pickle.dump({"date": str(date.today()), "data": df}, f)

    _memory[ticker] = df
    logger.info("Downloaded %d rows for %s.", len(df), ticker)
    return df


def get_current_price(ticker: str) -> float:
    """Return the latest available closing price."""
    df = fetch_ohlcv(ticker)
    return float(df["Close"].iloc[-1])


def invalidate(ticker: str) -> None:
    """Clear in-memory and file cache for a ticker (forces re-download)."""
    ticker = ticker.upper()
    _memory.pop(ticker, None)
    path = _cache_path(ticker)
    if path.exists():
        path.unlink()
    logger.info("Cache cleared for %s.", ticker)
