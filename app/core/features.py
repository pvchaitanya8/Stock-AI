import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Columns fed into the model (order matters — kept stable across train/infer)
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi",
    "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_lower", "bb_mid", "bb_pband",
    "ema_20", "ema_50", "ema_200",
    "stoch_k", "stoch_d",
    "atr",
    "obv",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all technical indicators and cyclical time features to an OHLCV
    DataFrame.  Rows with NaN (indicator warm-up period) are dropped.
    """
    f = df.copy()
    close  = f["Close"]
    high   = f["High"]
    low    = f["Low"]
    volume = f["Volume"]

    # --- Momentum ---
    f["rsi"] = RSIIndicator(close=close, window=14).rsi()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    f["stoch_k"] = stoch.stoch()
    f["stoch_d"] = stoch.stoch_signal()

    # --- Trend ---
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    f["macd"]        = macd_ind.macd()
    f["macd_signal"] = macd_ind.macd_signal()
    f["macd_diff"]   = macd_ind.macd_diff()

    f["ema_20"]  = EMAIndicator(close=close, window=20).ema_indicator()
    f["ema_50"]  = EMAIndicator(close=close, window=50).ema_indicator()
    f["ema_200"] = EMAIndicator(close=close, window=200).ema_indicator()

    # --- Volatility ---
    bb = BollingerBands(close=close, window=20, window_dev=2)
    f["bb_upper"] = bb.bollinger_hband()
    f["bb_lower"] = bb.bollinger_lband()
    f["bb_mid"]   = bb.bollinger_mavg()
    f["bb_pband"] = bb.bollinger_pband()

    f["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # --- Volume ---
    f["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # --- Cyclical time features ---
    f["day_sin"]   = np.sin(2 * np.pi * f.index.dayofweek / 7)
    f["day_cos"]   = np.cos(2 * np.pi * f.index.dayofweek / 7)
    f["month_sin"] = np.sin(2 * np.pi * f.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * f.index.month / 12)

    return f[FEATURE_COLS].dropna()


def get_indicator_snapshot(df_feat: pd.DataFrame) -> dict:
    """
    Return the most-recent values of all key indicators as a plain dict
    (used by the signal engine and the /indicators API response).
    """
    last = df_feat.iloc[-1]
    prev = df_feat.iloc[-2]
    return {
        "rsi":          round(float(last["rsi"]), 2),
        "macd":         round(float(last["macd"]), 4),
        "macd_signal":  round(float(last["macd_signal"]), 4),
        "macd_diff":    round(float(last["macd_diff"]), 4),
        "macd_cross":   _macd_cross(df_feat),
        "bb_upper":     round(float(last["bb_upper"]), 2),
        "bb_lower":     round(float(last["bb_lower"]), 2),
        "bb_mid":       round(float(last["bb_mid"]), 2),
        "bb_pband":     round(float(last["bb_pband"]), 4),
        "ema_20":       round(float(last["ema_20"]), 2),
        "ema_50":       round(float(last["ema_50"]), 2),
        "ema_200":      round(float(last["ema_200"]), 2),
        "stoch_k":      round(float(last["stoch_k"]), 2),
        "stoch_d":      round(float(last["stoch_d"]), 2),
        "atr":          round(float(last["atr"]), 4),
        "obv":          round(float(last["obv"]), 0),
        "close":        round(float(last["Close"]), 2),
    }


def _macd_cross(df: pd.DataFrame) -> str:
    """Detect the most recent MACD cross direction."""
    diff = df["macd_diff"]
    if len(diff) < 2:
        return "none"
    if diff.iloc[-2] < 0 and diff.iloc[-1] >= 0:
        return "bullish"
    if diff.iloc[-2] > 0 and diff.iloc[-1] <= 0:
        return "bearish"
    return "none"
