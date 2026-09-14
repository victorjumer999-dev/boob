# Momentum Trading — build, test, backtest

[![CI](https://github.com/victorjumer999-dev/boob/actions/workflows/ci.yml/badge.svg)](https://github.com/victorjumer999-dev/boob/actions/workflows/ci.yml)

A small, auditable momentum research toolkit built from the "Quantitative
Finance Essentials 04/04 — Momentum Trading" reference sheet, plus a real-data
backtest of every claim on it.

**Three studies, three null results.**

* **Monthly, 6 SPDR sectors + SPY (2002–2026)** — the cross-sectional
  long/short strategy the sheet advertises earns Sharpe **0.15** (t = 0.86)
  after costs and loses to SPY by 8.4%/yr. Write-up: [`REPORT.md`](REPORT.md).
* **Daily, XAUUSD (5 years)** — time-series momentum earns Sharpe **1.09**
  against buy & hold's **1.18**, gives zero crash protection (its five worst
  days are gold's five worst days), and 1 of 22 grid cells beats the
  benchmark. Write-up: [`REPORT_DAILY.md`](REPORT_DAILY.md).
* **Monthly, 13 assets across 4 classes (2001–2026)** — widening the universe
  cut average pairwise correlation from **0.59 to 0.28** and still did not
  rescue cross-sectional momentum (Sharpe **0.04** vs equal-weight's **0.72**);
  the pooled predictive coefficient is negative. Also tests whether silver
  leads gold: it does not (R² = 0.0003 over 15 years), though the two are
  0.78 correlated. Write-up: [`REPORT_WIDE.md`](REPORT_WIDE.md).

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
  run_backtest.py        monthly sector study  -> results/
  make_charts.py         monthly chart panel   -> results/momentum_report.png
  check_results.py       golden-result check: the study must reproduce REPORT.md
  run_daily_xauusd.py    daily XAUUSD study    -> results_daily/
  make_charts_daily.py   daily chart panel     -> results_daily/xauusd_daily.png
  run_gold_silver.py     gold/silver lead-lag  -> results_daily/
  run_wide_universe.py   13-asset study        -> results_wide/
  make_charts_wide.py    wide chart panel      -> results_wide/wide_report.png
data/raw/          cached monthly adjusted closes (sector ETFs + SPY)
data/raw_daily/    cached daily XAUUSD + XAGUSD spot, weekday bars only
data/raw_wide/     cached monthly panel, 13 assets across 4 asset classes
results/           monthly study output
results_daily/     daily study output
results_wide/      wide-universe study output
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 111 passed
python scripts/run_backtest.py      # prints the study, writes results/
python scripts/check_results.py     # asserts the headline figures still hold
python scripts/make_charts.py       # writes results/momentum_report.png

python scripts/run_daily_xauusd.py  # the daily XAUUSD study
python scripts/make_charts_daily.py # writes results_daily/xauusd_daily.png

python scripts/run_gold_silver.py   # gold/silver correlation and lead-lag
python scripts/run_wide_universe.py # the 13-asset study
python scripts/make_charts_wide.py  # writes results_wide/wide_report.png
```

## Position sizing

Equal weights are not equal risk. Across the wide universe annualised
volatility runs from 12% (XLP) to 71% (natural gas) — a 5.7x spread — and an
equal-weighted book is a natural-gas bet with equities attached. It produced a
-99.8% drawdown before `BacktestConfig(risk_parity=True)` was available.
Widening a universe and fixing position sizing are not independent changes:
breadth is what introduces the heterogeneity that breaks equal weighting.

## Frequency

The library defaults to monthly but **never assumes** a frequency when
annualising: every function that scales by time takes `periods_per_year`, and
`BacktestConfig` carries it. Using 12 where 261 belongs is a silent 4.66x
error in every Sharpe — it raises nothing and the number still looks
plausible, so `tests/test_frequency.py` pins the scaling law explicitly.

`data.infer_bars_per_year` measures the observed frequency from the index, and
the daily study aborts if it disagrees with the declared constant. That check
is what caught the XAUUSD feed shipping weekend bars (365/yr, with Saturday
carrying a full weekday's volatility — see `REPORT_DAILY.md` §7).

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

* **tests** — the full suite on Python 3.11, 3.12 and 3.13.
* **study** — reproduces the whole pipeline end to end on the cached data,
  then runs `check_results.py`, which asserts 19 headline figures still match
  the ones quoted in `REPORT.md`. A backtest that silently reports different
  numbers after a library upgrade is the same class of defect as the five
  listed in the report: it does not crash, it just quietly says something
  else. The rendered `results/` directory is uploaded as a build artifact.

If a golden result changes intentionally, update both `scripts/check_results.py`
and the figures quoted in `REPORT.md` — they are written into the prose.

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
