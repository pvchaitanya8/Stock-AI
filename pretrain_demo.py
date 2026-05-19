"""Pre-train models for demo tickers so they load instantly during a live demo."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from app.core.data import fetch_ohlcv
from app.core.features import compute_features
from app.core.trainer import train, is_trained

DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]

for t in DEMO_TICKERS:
    if is_trained(t):
        print(f"[skip] {t} already trained")
        continue
    print(f"\n=== Training {t} ===")
    df = fetch_ohlcv(t)
    feat = compute_features(df)
    print(f"Rows: {len(feat)}")
    metrics = train(t, feat)
    print(f"Done: {metrics}")

print("\nAll demo tickers ready.")
