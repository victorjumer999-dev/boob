# Momentum Trading — build, test, backtest

A small, auditable momentum research toolkit built from the "Quantitative
Finance Essentials 04/04 — Momentum Trading" reference sheet, plus a real-data
backtest of every claim on it.

**Short version of the result:** on this universe the cross-sectional
long/short momentum strategy the sheet advertises earns a Sharpe of **0.15**
(t = 0.86) after costs and loses to SPY by 8.4%/yr. The prior-return ladder is
monotone, but the spread is too small to survive the short leg and turnover.
Full write-up in [`REPORT.md`](REPORT.md).

## Layout

```
momentum/          the library
  data.py          loading, month-end normalisation, quality gates
  signals.py       12-1 momentum, price/SMA, rolling z-score, ranks, vol target
  backtest.py      weight application, costs, turnover, XS + TS engines
  metrics.py       CAGR, Sharpe, Sortino, drawdown, Calmar
  stats.py         Newey-West HAC, moving-block bootstrap, predictive regression
  synth.py         synthetic panels with a *known* planted edge, for the tests
tests/             83 tests, incl. explicit look-ahead traps
scripts/
  run_backtest.py  the full study -> results/
  make_charts.py   the chart panel  -> results/momentum_report.png
data/raw/          cached monthly adjusted closes (Alpha Vantage)
results/           CSVs, report.json, chart
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 83 passed
python scripts/run_backtest.py      # prints the study, writes results/
python scripts/make_charts.py       # writes results/momentum_report.png
```

## Data

Monthly split- and dividend-adjusted closes for 6 SPDR sector ETFs
(XLE, XLF, XLK, XLP, XLV, XLY) plus SPY, Dec-1999 to Aug-2026, from Alpha
Vantage. The daily endpoint is premium-gated on this key, so the study runs at
monthly frequency — which is the frequency Jegadeesh & Titman used anyway.

Two data hazards are handled explicitly and tested:

* **Partial final month.** The vendor stamps the latest trading day as if it
  were a completed bar (2026-09-11 on a 2026-09-14 run). It is dropped.
* **Splits.** XLK, XLE and XLY all split in Dec-2025. The adjusted-close
  column absorbs them; `tests/test_data.py` asserts no month shows the ~-50%
  move an unadjusted series would print.

## Design rules the tests enforce

1. **No look-ahead.** A signal indexed at `t` is shifted exactly once, in
   `run_weighted_backtest`, before meeting returns. The test suite feeds the
   engine a literal oracle signal (this bar's winner) and asserts the reported
   Sharpe stays below 1.0 — with a control proving the same oracle scores 4.6
   when the shift is removed.
2. **No silent neutral defaults.** Nothing returns `0.0`, `1.0`, `0.5` or an
   empty Series on a failure path. Too short a sample, a dead signal layer, an
   unadjusted split, a degenerate variance — each raises.
3. **Planted-edge recovery.** `synth.py` generates panels with a known
   cross-sectional momentum effect. The engine must find it, must find nothing
   on a random walk, and must rank a strong effect above a weak one.
