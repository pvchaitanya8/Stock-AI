# 🪙 Stock AI
<div align="center">
  <img src="https://res.cloudinary.com/dwco7vfgp/image/upload/v1784465662/6_h21rs2.png" alt="SignBridge Hero Image" width="100%" />
</div>

> An interactive forecaster's notebook for individual traders.

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  </p>
</div>


---

A complete rewrite of the original Flask + Keras stock predictor. The first version asked the user to type today's OHLCV by hand, supported only 14 hardcoded tickers, and produced a single next-day closing price. 

<div align="center">
  <img src="https://res.cloudinary.com/dwco7vfgp/image/upload/v1784473337/1_mldbmy.png" alt="UI Screenshot Placeholder" width="800" />
</div>

**This version does what a trader actually needs:**
- 🔍 **Any Ticker:** Live data is fetched automatically.
- 🔮 **30-Day Forecast:** Featuring a 10th / 90th percentile confidence band.
- 🚦 **Trade Signals:** Get a Buy / Hold / Sell signal with technical reasoning written out.
- 🎨 **Interactive UI:** Explore candlesticks and indicator panels in a hand-drawn *architect's notebook* theme.

---

## 📑 Table of Contents

1. [✨ What's New vs V1](#-whats-new-vs-v1)
2. [🏗️ Architecture](#️-architecture)
3. [📂 Project Layout](#-project-layout)
4. [🚀 Getting Started](#-getting-started)
5. [🔌 API Reference](#-api-reference)
6. [🧠 Model Details](#-model-details)
7. [📊 Signal Logic](#-signal-logic)
8. [🎯 Demo Guide](#-demo-guide)
9. [⚠️ Disclaimer](#️-disclaimer)

---

## ✨ What's New vs V1

| Area          | V1 (legacy)                   | **V2 (this repo)**                                                       |
| ------------- | ----------------------------- | ------------------------------------------------------------------------ |
| Backend       | Flask + Keras (TensorFlow)    | **FastAPI + PyTorch**                                                    |
| Tickers       | 14 hardcoded                  | **Any** ticker (live from yfinance)                                      |
| Input         | Manual OHLCV entry            | **Auto-fetched** live                                                    |
| Model         | Single-row LSTM, 1-day output | **3-layer LSTM + Multi-Head Attention**, 60-day window, 30-day forecast |
| Uncertainty   | None                          | **MC Dropout** with 200 samples → 10/50/90 percentile bands              |
| Features      | 5 (OHLCV)                     | **24** (OHLCV + 18 TA indicators + cyclical time)                        |
| Trade Signal  | None                          | **Composite Buy / Hold / Sell** with human-readable reasons              |
| Charts        | Static matplotlib PNG         | **Interactive Plotly** candlesticks + sub-panels                         |
| Visual design | Plain Bootstrap-ish form      | **Hand-drawn graph-paper notebook** theme                                |

> **Note:** The original V1 is preserved untouched in [`_legacy/`](./_legacy/).

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│  Browser (Plotly + vanilla JS)                                   │
└──────────────────────────────────┬───────────────────────────────┘
                                   │  REST (JSON)
┌──────────────────────────────────┴───────────────────────────────┐
│  FastAPI                                                         │
│  ┌──────────────────────────┬──────────────────────────────────┐ │
│  │  /api/predict            │  /api/indicators   /api/train    │ │
│  └──────────────┬───────────┴───────────────────────┬──────────┘ │
│                 │                                   │            │
│  ┌──────────────┴───────────┐         ┌─────────────┴──────────┐ │
│  │  core/signals            │         │  core/trainer          │ │
│  │  composite scorer        │         │  PyTorch model +       │ │
│  │  → BUY / HOLD / SELL     │         │  MC Dropout inference  │ │
│  └──────────────┬───────────┘         └─────────────┬──────────┘ │
│                 │                                   │            │
│  ┌──────────────┴───────────────────────────────────┴──────────┐ │
│  │  core/features  · 24 TA indicators via `ta` library         │ │
│  └──────────────────────────────┬──────────────────────────────┘ │
│  ┌──────────────────────────────┴──────────────────────────────┐ │
│  │  core/data      · yfinance fetch + daily disk cache         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Layout

```text
.
├── app/                        # FastAPI backend
│   ├── main.py                 # entry point: routes + static frontend mount
│   ├── routes/
│   │   ├── predict.py          # GET  /api/predict/{ticker}
│   │   ├── indicators.py       # GET  /api/indicators/{ticker}
│   │   └── train.py            # POST /api/train/{ticker}
│   ├── core/
│   │   ├── data.py             # yfinance fetch + daily file cache
│   │   ├── features.py         # 24 technical-analysis features
│   │   ├── model.py            # LSTM + Multi-Head Attention (PyTorch)
│   │   ├── trainer.py          # training loop + MC Dropout inference
│   │   └── signals.py          # composite Buy/Hold/Sell scoring
│   └── saved_models/           # per-ticker .pt + scaler (gitignored)
├── frontend/                   # single-page UI
│   ├── index.html
│   ├── app.js                  # Plotly charts + REST calls
│   └── styles.css              # pencil & graph-paper theme
├── _legacy/                    # untouched V1 (Flask + Keras)
├── pretrain_demo.py            # warm up models for demo tickers
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.11**
- Windows (PowerShell), macOS, or Linux
- ~2 GB free disk (PyTorch + dependencies)

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd Stock-AI
```

### 2. Create and activate a virtual environment

**Windows / PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks script execution, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-train demo tickers

This trains AAPL, MSFT, GOOGL, and TSLA up front so the demo loads instantly. Expect ~10-15 minutes total on CPU.

```bash
python pretrain_demo.py
```

*If you skip this step, the first analysis of any ticker will train its model inline (2-5 minutes per ticker).*

### 5. Run the server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in your browser.

The interactive API explorer lives at <http://127.0.0.1:8000/docs>.

<div align="center">
  <img src="https://res.cloudinary.com/dwco7vfgp/image/upload/v1784473337/2_tzyvdr.png" alt="API Docs Placeholder" width="600" />
</div>

---

## 🔌 API Reference

### `GET /health`

Liveness check.

```json
{ "status": "ok", "version": "2.0.0" }
```

### `POST /api/train/{ticker}?refresh=false`

Train or retrain a model. Pass `?refresh=true` to invalidate the cached yfinance data and force a fresh download.

```json
{ "status": "ok", "ticker": "MSFT", "epochs": 29, "best_val_loss": 0.007222 }
```

### `GET /api/predict/{ticker}`

Returns forecast, indicator snapshot, and signal. Returns `409` if the model has not been trained for that ticker yet.

```json
{
  "ticker": "MSFT",
  "current_price": 415.32,
  "last_date": "2026-05-18",
  "forecast": [
    { "date": "2026-05-19", "low": 408.1, "median": 419.7, "high": 432.4 }
  ],
  "indicators": {
    "rsi": 58.4, "macd": 1.23, "macd_signal": 0.98, "macd_diff": 0.25,
    "macd_cross": "bullish", "bb_pband": 0.62, "ema_50": 410.1
  },
  "signal": {
    "signal": "BUY",
    "score": 2,
    "predicted_return_pct": 18.96,
    "reasons": [
      "Predicted 30-day return of +18.96% — strong upside.",
      "MACD bullish crossover detected.",
      "Uptrend confirmed: price > EMA-50 > EMA-200."
    ]
  }
}
```

### `GET /api/indicators/{ticker}?days=180`

Returns the last `days` of OHLCV plus every indicator, used to render the historical portion of the charts.

---

## 🧠 Model Details

| Hyperparameter        | Value             |
| --------------------- | ----------------- |
| **Architecture**      | LSTM + MHA + MLP  |
| **LSTM layers**       | 3                 |
| **Hidden dim**        | 256               |
| **Attention heads**   | 4                 |
| **Lookback window**   | 60 days           |
| **Forecast horizon**  | 30 days           |
| **Input features**    | 24                |
| **Dropout** (active in inference) | 0.2   |
| **Loss**              | Huber             |
| **Optimizer**         | Adam (lr 1e-3)    |
| **LR scheduler**      | ReduceLROnPlateau |
| **Early-stopping patience** | 15 epochs   |
| **MC Dropout samples**| 200               |

**Features (24):** Open, High, Low, Close, Volume, RSI(14), MACD line / signal / histogram, Bollinger upper / lower / mid / %B, EMA 20 / 50 / 200, Stochastic %K / %D, ATR(14), OBV, day-of-week sin/cos, month sin/cos.

**Uncertainty:** Dropout layers remain active at inference; 200 forward passes yield a distribution of forecasts, from which the 10th / 50th / 90th percentiles produce the dashed median line and shaded band visible in the chart.

---

## 📊 Signal Logic

A composite score is computed from five weighted criteria:

| Criterion                       | Weight  |
| ------------------------------- | ------- |
| Predicted 30-day return > +3 %  | **+2**  |
| Predicted 30-day return > 0 %   | +1      |
| Predicted 30-day return < −3 %  | **−2**  |
| RSI < 35 (oversold)             | +1      |
| RSI > 70 (overbought)           | −1      |
| MACD bullish / bearish cross    | ±1      |
| Price near lower / upper BB     | ±1      |
| Stochastic %K < 20 / > 80       | ±1      |

| Composite Score | Label            |
| --------------- | ---------------- |
| ≥ 3             | **STRONG BUY**   |
| 1 to 2          | BUY              |
| 0               | HOLD             |
| −1 to −2        | SELL             |
| ≤ −3            | **STRONG SELL**  |

> The price-vs-EMA-50-vs-EMA-200 ordering adds a confirmation line to the reasoning but does not move the score.

---

## 🎯 Demo Guide

Suggested flow once the server is running:

1. **MSFT** → typically lands on BUY with healthy upside → shows the green signal card, candlestick + Bollinger band + EMA overlay, dashed blue forecast median, shaded confidence band.
2. **GOOGL** → swings to STRONG SELL (red) → demonstrates that the same pipeline produces dramatically different recommendations.
3. **AAPL** → STRONG SELL on overbought RSI — useful to talk through the "Field Notes" reasoning list.
4. **Any fresh ticker (e.g. NVDA)** → demonstrates the live training flow (the "Sketching a model…" loading state takes 2-5 minutes).

<div align="center">
  <img src="https://res.cloudinary.com/dwco7vfgp/image/upload/v1784473337/3_hrbhvb.png" alt="Demo Screenshot Placeholder" width="800" />
</div>

---

## ⚠️ Disclaimer

Forecasts are AI-generated estimates with inherent model uncertainty. 
The information here is for educational and research purposes only and is **not financial advice**.

---

## 💖 Support

Consider supporting by:

<p align="center">
  <a href="https://patreon.com/Chaitanya888"><img src="https://img.shields.io/badge/Patreon-FF424D?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon" /></a>
  &nbsp;
  <a href="https://buymeacoffee.com/chaitanya888"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee" /></a>
</p>

<br/>

---


## 📜 License
Distributed under the Apache-2.0 License. See [LICENSE](./LICENSE) for more information.

---
