// ─────────────────────────────────────────────────────────────────────────
//  StockAI V2 — frontend logic
//  Pencil-on-paper Plotly theme, identical to the surrounding CSS.
// ─────────────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const inp        = $('ticker');
const btnAnalyze = $('analyze');
const btnRetrain = $('retrain');
const statusBox  = $('status');
const results    = $('results');

btnAnalyze.addEventListener('click', () => analyze(false));
btnRetrain.addEventListener('click', () => analyze(true));
inp.addEventListener('keydown', e => { if (e.key === 'Enter') analyze(false); });

// ── Status helpers ──────────────────────────────────────────────────────────
function setStatus(msg, type = '') {
  statusBox.className = 'status ' + type;
  statusBox.textContent = msg;
}

// ── Main flow ───────────────────────────────────────────────────────────────
async function analyze(forceRetrain) {
  const ticker = inp.value.trim().toUpperCase();
  if (!ticker) { setStatus('Enter a ticker symbol.', 'error'); return; }

  results.classList.add('hidden');

  try {
    if (forceRetrain) {
      await trainAndWait(ticker, true);
    } else {
      try {
        await loadAndRender(ticker);
        return;
      } catch (e) {
        if (e.status === 409) {
          await trainAndWait(ticker, false);
        } else {
          throw e;
        }
      }
    }
    await loadAndRender(ticker);
  } catch (e) {
    setStatus(`Error: ${e.message || e}`, 'error');
  }
}

async function trainAndWait(ticker, refresh) {
  setStatus(`Sketching a model for ${ticker} — drafting takes 2-5 minutes on CPU…`, 'loading');
  const r = await fetch(`/api/train/${ticker}?refresh=${refresh}`, { method: 'POST' });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw Object.assign(new Error(err.detail || 'Training failed'), { status: r.status });
  }
  const data = await r.json();
  setStatus(`Model ready: ${data.epochs} epochs, val loss ${data.best_val_loss}. Plotting…`);
}

async function loadAndRender(ticker) {
  setStatus(`Plotting forecast and indicators for ${ticker}…`, 'loading');

  const [pRes, iRes] = await Promise.all([
    fetch(`/api/predict/${ticker}`),
    fetch(`/api/indicators/${ticker}`),
  ]);

  if (!pRes.ok) {
    const err = await pRes.json().catch(() => ({ detail: pRes.statusText }));
    throw Object.assign(new Error(err.detail || 'Prediction failed'), { status: pRes.status });
  }
  if (!iRes.ok) throw new Error('Indicators failed');

  render(await pRes.json(), (await iRes.json()).history);
  setStatus('');
}

// ── Renderers ───────────────────────────────────────────────────────────────
function render(pred, history) {
  results.classList.remove('hidden');

  $('ticker-display').textContent = pred.ticker;
  $('last-date').textContent      = `data through ${pred.last_date}`;
  $('current-price').textContent  = `$${pred.current_price}`;

  const pct = pred.signal.predicted_return_pct;
  const pctStr = (pct >= 0 ? '+' : '') + pct + '%';
  $('pred-return').textContent = `forecast 30-day · ${pctStr}`;

  const sigCard = $('signal-card');
  sigCard.className = 'card signal-card ' + pred.signal.signal.toLowerCase().replace(' ', '-');
  $('signal-label').textContent = pred.signal.signal;
  $('signal-score').textContent = `composite score · ${pred.signal.score >= 0 ? '+' : ''}${pred.signal.score}`;

  const ul = $('reasons');
  ul.innerHTML = '';
  pred.signal.reasons.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    ul.appendChild(li);
  });

  const grid = $('indicator-grid');
  grid.innerHTML = '';
  const order = [
    ['RSI', pred.indicators.rsi],
    ['MACD', pred.indicators.macd],
    ['MACD Cross', pred.indicators.macd_cross],
    ['BB %B', pred.indicators.bb_pband],
    ['BB Upper', pred.indicators.bb_upper],
    ['BB Lower', pred.indicators.bb_lower],
    ['EMA 20', pred.indicators.ema_20],
    ['EMA 50', pred.indicators.ema_50],
    ['EMA 200', pred.indicators.ema_200],
    ['Stoch %K', pred.indicators.stoch_k],
    ['Stoch %D', pred.indicators.stoch_d],
    ['ATR', pred.indicators.atr],
  ];
  order.forEach(([name, num]) => {
    const div = document.createElement('div');
    div.className = 'cell';
    div.innerHTML = `<div class="name">${name}</div><div class="num">${num}</div>`;
    grid.appendChild(div);
  });

  drawPriceChart(history, pred.forecast);
  drawRSI(history);
  drawMACD(history);
  drawStoch(history);
  drawATR(history);
  drawOBV(history);
}

// ── Pencil-on-paper Plotly theme ────────────────────────────────────────────
const PAPER  = '#f3efe4';
const INK    = '#2a2622';
const INK_M  = '#7a7368';
const INK_F  = '#a39c8e';
const GRID   = 'rgba(110, 90, 60, 0.18)';
const PENCIL = {
  blue:  '#2e4a6b',
  green: '#3d6b3d',
  red:   '#a83b3b',
  amber: '#a87526',
  plum:  '#6b3d6b',
};

const baseLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',     // let the cream paper show through
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: {
    family: 'Inter, -apple-system, Segoe UI, Roboto, sans-serif',
    color:  INK,
    size:   12,
  },
  margin: { l: 56, r: 16, t: 6, b: 38 },
  xaxis: {
    gridcolor: GRID,
    linecolor: INK_F,
    zerolinecolor: INK_F,
    tickfont: { color: INK_M, family: 'JetBrains Mono, Consolas, monospace', size: 11 },
  },
  yaxis: {
    gridcolor: GRID,
    linecolor: INK_F,
    zerolinecolor: INK_F,
    tickfont: { color: INK_M, family: 'JetBrains Mono, Consolas, monospace', size: 11 },
  },
  legend: {
    orientation: 'h', y: -0.22,
    font: { size: 11, color: INK_M, family: 'Inter, sans-serif' },
    bgcolor: 'rgba(255,252,244,0.4)',
    bordercolor: INK_F,
    borderwidth: 0,
  },
  hoverlabel: {
    bgcolor: '#fffcf4',
    bordercolor: INK,
    font: { family: 'JetBrains Mono, monospace', color: INK, size: 12 },
  },
};
const plotConfig = { displayModeBar: false, responsive: true };

// ── Charts ──────────────────────────────────────────────────────────────────
function drawPriceChart(history, forecast) {
  const dates  = history.map(r => r.date);
  const opens  = history.map(r => r.open);
  const highs  = history.map(r => r.high);
  const lows   = history.map(r => r.low);
  const closes = history.map(r => r.close);
  const bbUp   = history.map(r => r.bb_upper);
  const bbLo   = history.map(r => r.bb_lower);
  const ema50  = history.map(r => r.ema_50);
  const ema200 = history.map(r => r.ema_200);

  const fDates = forecast.map(r => r.date);
  const fMed   = forecast.map(r => r.median);
  const fLow   = forecast.map(r => r.low);
  const fHigh  = forecast.map(r => r.high);

  const traces = [
    {
      type: 'candlestick', x: dates, open: opens, high: highs, low: lows, close: closes,
      name: 'price',
      increasing: { line: { color: PENCIL.green, width: 1 }, fillcolor: 'rgba(61,107,61,0.35)' },
      decreasing: { line: { color: PENCIL.red,   width: 1 }, fillcolor: 'rgba(168,59,59,0.35)' },
    },
    { x: dates, y: bbUp,   name: 'BB upper',  type: 'scatter', mode: 'lines',
      line: { color: INK_F, width: 1, dash: 'dot' }, hoverinfo: 'skip' },
    { x: dates, y: bbLo,   name: 'BB band',  type: 'scatter', mode: 'lines',
      line: { color: INK_F, width: 1, dash: 'dot' },
      fill: 'tonexty', fillcolor: 'rgba(110, 90, 60, 0.06)', hoverinfo: 'skip' },
    { x: dates, y: ema50,  name: 'EMA 50',   type: 'scatter', mode: 'lines',
      line: { color: PENCIL.amber, width: 1.4 } },
    { x: dates, y: ema200, name: 'EMA 200',  type: 'scatter', mode: 'lines',
      line: { color: PENCIL.plum,  width: 1.4 } },
    { x: fDates, y: fHigh, name: 'forecast 90%', type: 'scatter', mode: 'lines',
      line: { color: 'rgba(46,74,107,0)' }, showlegend: false, hoverinfo: 'skip' },
    { x: fDates, y: fLow,  name: 'forecast band', type: 'scatter', mode: 'lines',
      line: { color: 'rgba(46,74,107,0)' },
      fill: 'tonexty', fillcolor: 'rgba(46,74,107,0.18)' },
    { x: fDates, y: fMed,  name: 'forecast median', type: 'scatter', mode: 'lines',
      line: { color: PENCIL.blue, width: 2.2, dash: 'dash' } },
  ];

  const layout = {
    ...baseLayout,
    height: 480,
    xaxis: { ...baseLayout.xaxis, rangeslider: { visible: false } },
  };
  Plotly.newPlot('price-chart', traces, layout, plotConfig);
}

function drawRSI(history) {
  const dates = history.map(r => r.date);
  const rsi   = history.map(r => r.rsi);
  const traces = [
    { x: dates, y: rsi, type: 'scatter', mode: 'lines', name: 'RSI',
      line: { color: PENCIL.blue, width: 1.6 } },
    { x: [dates[0], dates[dates.length-1]], y: [70, 70], mode: 'lines',
      line: { color: PENCIL.red,   dash: 'dot', width: 1 }, showlegend: false, hoverinfo: 'skip' },
    { x: [dates[0], dates[dates.length-1]], y: [30, 30], mode: 'lines',
      line: { color: PENCIL.green, dash: 'dot', width: 1 }, showlegend: false, hoverinfo: 'skip' },
  ];
  Plotly.newPlot('rsi-chart',
    traces,
    { ...baseLayout, height: 220, yaxis: { ...baseLayout.yaxis, range: [0, 100] } },
    plotConfig);
}

function drawMACD(history) {
  const dates = history.map(r => r.date);
  const traces = [
    { x: dates, y: history.map(r => r.macd),        type: 'scatter', mode: 'lines', name: 'MACD',
      line: { color: PENCIL.blue,  width: 1.6 } },
    { x: dates, y: history.map(r => r.macd_signal), type: 'scatter', mode: 'lines', name: 'signal',
      line: { color: PENCIL.amber, width: 1.4, dash: 'dot' } },
    { x: dates, y: history.map(r => r.macd_diff),   type: 'bar', name: 'histogram',
      marker: { color: history.map(r => r.macd_diff >= 0
                                    ? 'rgba(61,107,61,0.55)'
                                    : 'rgba(168,59,59,0.55)') } },
  ];
  Plotly.newPlot('macd-chart', traces, { ...baseLayout, height: 220 }, plotConfig);
}

function drawStoch(history) {
  const dates = history.map(r => r.date);
  const traces = [
    { x: dates, y: history.map(r => r.stoch_k), type: 'scatter', mode: 'lines', name: '%K',
      line: { color: PENCIL.blue,  width: 1.5 } },
    { x: dates, y: history.map(r => r.stoch_d), type: 'scatter', mode: 'lines', name: '%D',
      line: { color: PENCIL.amber, width: 1.3, dash: 'dot' } },
    { x: [dates[0], dates[dates.length-1]], y: [80, 80], mode: 'lines',
      line: { color: PENCIL.red,   dash: 'dot', width: 1 }, showlegend: false, hoverinfo: 'skip' },
    { x: [dates[0], dates[dates.length-1]], y: [20, 20], mode: 'lines',
      line: { color: PENCIL.green, dash: 'dot', width: 1 }, showlegend: false, hoverinfo: 'skip' },
  ];
  Plotly.newPlot('stoch-chart',
    traces,
    { ...baseLayout, height: 220, yaxis: { ...baseLayout.yaxis, range: [0, 100] } },
    plotConfig);
}

function drawATR(history) {
  const dates = history.map(r => r.date);
  Plotly.newPlot('atr-chart', [
    { x: dates, y: history.map(r => r.atr), type: 'scatter', mode: 'lines', name: 'ATR',
      line: { color: PENCIL.plum, width: 1.6 },
      fill: 'tozeroy', fillcolor: 'rgba(107,61,107,0.10)' },
  ], { ...baseLayout, height: 220 }, plotConfig);
}

function drawOBV(history) {
  const dates = history.map(r => r.date);
  Plotly.newPlot('obv-chart', [
    { x: dates, y: history.map(r => r.obv), type: 'scatter', mode: 'lines', name: 'OBV',
      line: { color: PENCIL.green, width: 1.6 },
      fill: 'tozeroy', fillcolor: 'rgba(61,107,61,0.10)' },
  ], { ...baseLayout, height: 220 }, plotConfig);
}
