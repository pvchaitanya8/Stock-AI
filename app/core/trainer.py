import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from app.core.features import FEATURE_COLS
from app.core.model import LSTMAttentionForecaster

logger = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN    = 60     # days of history fed to the model
HORIZON    = 30     # days to forecast
HIDDEN     = 256
N_LAYERS   = 3
N_HEADS    = 4
DROPOUT    = 0.2
BATCH_SIZE = 32
LR         = 1e-3
MAX_EPOCHS = 100
PATIENCE   = 15     # early-stopping patience
MC_SAMPLES = 200    # MC Dropout forward passes at inference

MODELS_DIR = Path(__file__).parent.parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Paths ──────────────────────────────────────────────────────────────────────
def _model_path(ticker: str) -> Path:
    return MODELS_DIR / f"{ticker}.pt"

def _scaler_path(ticker: str) -> Path:
    return MODELS_DIR / f"{ticker}_scaler.pkl"

def is_trained(ticker: str) -> bool:
    return _model_path(ticker.upper()).exists()


# ── Dataset ────────────────────────────────────────────────────────────────────
class _SlideWindow(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        xs, ys = [], []
        n = len(features)
        for i in range(n - SEQ_LEN - HORIZON + 1):
            xs.append(features[i : i + SEQ_LEN])
            ys.append(targets[i + SEQ_LEN : i + SEQ_LEN + HORIZON])
        self.X = torch.tensor(np.array(xs), dtype=torch.float32)
        self.y = torch.tensor(np.array(ys), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training ───────────────────────────────────────────────────────────────────
def train(ticker: str, df_feat: pd.DataFrame) -> dict:
    ticker = ticker.upper()
    logger.info("Starting training for %s (%d rows).", ticker, len(df_feat))

    feat_arr  = df_feat[FEATURE_COLS].values.astype(np.float32)
    close_arr = df_feat[["Close"]].values.astype(np.float32)

    split = int(len(feat_arr) * 0.8)

    feat_scaler  = MinMaxScaler()
    close_scaler = MinMaxScaler()

    feat_scaled  = feat_scaler.fit_transform(feat_arr)
    close_scaled = close_scaler.fit_transform(close_arr).flatten()

    train_ds = _SlideWindow(feat_scaled[:split],  close_scaled[:split])
    val_ds   = _SlideWindow(feat_scaled[split:],  close_scaled[split:])

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Not enough data to build training windows. Need at least 5 years of history.")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = LSTMAttentionForecaster(
        n_features=len(FEATURE_COLS),
        hidden=HIDDEN, n_layers=N_LAYERS,
        n_heads=N_HEADS, dropout=DROPOUT,
        horizon=HORIZON,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    criterion  = nn.HuberLoss()

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        # ── validate ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                v_loss += criterion(model(xb.to(device)), yb.to(device)).item()

        t_loss /= len(train_dl)
        v_loss /= max(len(val_dl), 1)
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            logger.info("Epoch %3d | train=%.5f | val=%.5f | lr=%.2e",
                        epoch, t_loss, v_loss,
                        optimizer.param_groups[0]["lr"])

        if no_improve >= PATIENCE:
            logger.info("Early stop at epoch %d.", epoch)
            break

    # ── persist ──
    torch.save(
        {
            "state_dict": best_state,
            "n_features": len(FEATURE_COLS),
            "config": dict(hidden=HIDDEN, n_layers=N_LAYERS,
                           n_heads=N_HEADS, dropout=DROPOUT, horizon=HORIZON),
        },
        _model_path(ticker),
    )
    joblib.dump({"feat": feat_scaler, "close": close_scaler}, _scaler_path(ticker))
    logger.info("Model saved for %s. best_val=%.5f", ticker, best_val)

    return {"ticker": ticker, "epochs": epoch, "best_val_loss": round(best_val, 6)}


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(ticker: str, df_feat: pd.DataFrame) -> list[dict]:
    """
    Run MC Dropout inference and return 30 daily forecast rows with
    10th / 50th / 90th percentile price bands.
    """
    ticker = ticker.upper()

    ckpt    = torch.load(_model_path(ticker), map_location="cpu", weights_only=False)
    scalers = joblib.load(_scaler_path(ticker))
    feat_scaler  = scalers["feat"]
    close_scaler = scalers["close"]

    cfg   = ckpt["config"]
    model = LSTMAttentionForecaster(
        n_features=ckpt["n_features"], **cfg
    )
    model.load_state_dict(ckpt["state_dict"])
    model.train()   # keep dropout active for MC sampling

    feat_scaled = feat_scaler.transform(df_feat[FEATURE_COLS].values.astype(np.float32))
    seq = torch.tensor(feat_scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

    samples = []
    with torch.no_grad():
        for _ in range(MC_SAMPLES):
            samples.append(model(seq).squeeze(0).numpy())

    samples = np.array(samples)   # (MC_SAMPLES, HORIZON)

    def inv(arr):
        return close_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    p10 = inv(np.percentile(samples, 10, axis=0))
    p50 = inv(np.percentile(samples, 50, axis=0))
    p90 = inv(np.percentile(samples, 90, axis=0))

    last_date    = df_feat.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=HORIZON)

    return [
        {
            "date":   str(future_dates[i].date()),
            "low":    round(float(p10[i]), 2),
            "median": round(float(p50[i]), 2),
            "high":   round(float(p90[i]), 2),
        }
        for i in range(HORIZON)
    ]
