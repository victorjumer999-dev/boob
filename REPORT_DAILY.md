# XAUUSD, daily bars, 5 years — time-series momentum

**Instrument:** XAUUSD spot (Alpha Vantage gold history)
**Requested:** daily candles, 5 years — 1,305 weekday bars, 2021-09-13 → 2026-09-11
**Evaluated:** 952 bars (3.65 yrs) after a 273-bar warm-up for the 252-21 signal
**Costs:** 2bp one-way (swept to 25bp below)
**Annualisation:** 261 bars/yr, measured from the data, not assumed

---

## 1. The answer

**Buy and hold beat every momentum variant tested.**

| Strategy | CAGR | Vol | **Sharpe** | Max DD | Turnover | Cost drag |
|---|---:|---:|---:|---:|---:|---:|
| **Buy & hold gold** | 25.12% | 20.92% | **1.178** | −27.3% | 0 | 0 |
| TS long/flat (252-21) | 22.43% | 20.57% | 1.088 | −27.3% | 2.7/yr | 0.06%/yr |
| TS long/short | 19.63% | 20.93% | 0.962 | −27.3% | 5.5/yr | 0.12%/yr |
| Price > SMA(252) | 19.72% | 19.84% | 1.008 | −31.2% | 3.6/yr | 0.07%/yr |
| TS long/flat + 15% vol target | 23.82% | 17.33% | 1.321 | −23.2% | 8.8/yr | 0.19%/yr |

Across a 22-cell lookback grid (21–252 days × skip 0/21 × long-flat/long-short), **one cell** beat buy & hold — 42-day long/flat at Sharpe 1.191 vs 1.178. That is the best of 22 tries, which is what noise looks like.

---

## 2. Why momentum can't win here

Gold went from $1,793 to $4,319 — **+141%, 25%/yr** over the evaluated window. In a market like that a long/flat trend filter is long **95% of the time** (flat on 5% of bars). It cannot add anything; it can only occasionally step out and miss upside.

That is exactly what it does. Against a passive book held at the filter's *own* average 94.5% exposure:

```
constant-94.5%-exposure twin   Sharpe 1.178   CAGR 23.74%   maxDD -25.9%
TS long/flat                   Sharpe 1.088   CAGR 22.43%   maxDD -27.3%
timing excess                  -0.90%/yr,  t = -0.38
up-capture 94.6%   vs   down-capture 98.7%   ->  asymmetry +4.1pp
```

The asymmetry is **the wrong way round**. A working trend filter captures less downside than upside. This one captures *more* downside (98.7%) than upside (94.6%) — it is, marginally, worse than doing nothing.

---

## 3. It provided zero crash protection

The whole point of a trend filter is to be out before the break. It wasn't.

| Day | TS long/flat | Gold |
|---|---:|---:|
| 2026-02-02 (Mon) | **−13.09%** | −13.09% |
| 2025-10-22 (Wed) | −7.99% | −7.99% |
| 2026-03-23 (Mon) | −5.28% | −5.28% |
| 2026-02-06 (Fri) | −4.45% | −4.45% |

Every worst day is **identical to gold's**. The strategy was fully long through the January 2026 blow-off top ($5,478 on 29-Jan) and the −13.1% break that followed. Max drawdown −27.3% for both, to the basis point.

A 252-day lookback measured over a market that had tripled was still overwhelmingly positive while price was collapsing. The signal is far too slow to be a risk control at this horizon.

---

## 4. Costs are irrelevant here

| One-way cost | 0bp | 1bp | 2bp | 5bp | 10bp | 25bp |
|---|---:|---:|---:|---:|---:|---:|
| Sharpe | 1.091 | 1.090 | 1.088 | 1.084 | 1.076 | 1.054 |

At 2.7 turns/yr, even a punitive 25bp barely moves the result. This strategy does not fail because of friction — it fails because the signal has nothing to say. Worth stating plainly: cost assumptions are the usual place a backtest flatters itself, and here they are not doing any work.

---

## 5. The one thing that looked good, and why it isn't

The 15% vol-targeted variant posts the study's best Sharpe, 1.321 vs buy & hold's 1.178, with a shallower drawdown (−23.2%). Three reasons not to believe it:

1. **Its return is lower**, not higher: −1.75%/yr vs buy & hold, t = −0.42. The Sharpe gain is entirely a volatility reduction.
2. **It uses leverage** — up to the 2.0x cap, above 100% exposure on 44% of bars. It is not a like-for-like comparison with an unlevered book.
3. **Its Sharpe confidence interval is [0.397, 2.445]**, which swallows buy & hold's 1.178 whole.

Same conclusion as the monthly sector study: vol targeting times *volatility*, not *returns*. It is a risk-budgeting tool, not alpha.

---

## 6. Statistical power — read this before believing any number above

A 252-bar lookback over 952 bars is **3.8 independent momentum episodes.**

Daily sampling multiplies observations, not information. The 952 daily returns are not 952 independent tests of a 12-month signal; they are ~4 overlapping ones. That is why the HAC bandwidth is floored at 251 lags (the overlap minus one) in every t-statistic reported here.

Every inference in this document is **indicative, not decisive**. With 3.8 episodes, nothing short of an enormous effect could reach significance, and none of the effects here are enormous.

---

## 7. What the data needed before it could be used

Two defects in the vendor feed, both of which would have produced confident nonsense:

**The series is calendar-day, not trading-day.** 703 Saturdays and 703 Sundays across the full history; 365 bars/yr from 2014. Spot gold does not trade at the weekend.

The weekend bars are not carry-forwards, which is what makes this dangerous:

| | daily σ | exactly zero |
|---|---:|---:|
| Weekdays | 1.06% | 0.0% |
| **Saturday** | **1.09%** | 6.1% |
| Sunday | 0.15% | 3.4% |

Saturday carries a **full weekday's volatility** — real price movement stamped onto a closed market. Backtesting on it books P&L from trades that could never have been executed. Sunday's 0.15% is the thin spot re-open.

Fix: keep Mon–Fri only. Nothing is lost — the Friday→Monday return then spans the weekend, which is exactly what a holder experiences. Annualisation then uses the **measured** 261.3 bars/yr, not the US-equity 252 convention and not 365.

**The −13.09% day is real.** It looked like a bad print (gold's historical worst is ~−9%), but it is Fri 30-Jan close $5,335 → Mon 2-Feb close $4,637, spanning a weekend during a parabolic unwind. Genuine, and kept.

---

## 8. Code changes this required

The library was monthly-only: `PERIODS_PER_YEAR = 12` was a module constant imported by five modules.

- **Frequency is now a parameter**, not an assumption. Every function that annualises takes `periods_per_year`; `BacktestConfig` carries it. Defaults stay 12, so the monthly study is unchanged (all 83 prior tests still pass).
- **`momentum.data.infer_bars_per_year`** measures the observed frequency, and the study *refuses to run* if the declared constant disagrees with the data by more than 12 bars/yr — the check that catches a weekend-contaminated series.
- **`momentum.data.drop_weekends`** with the Fri→Mon semantics above.
- **14 new tests** (`tests/test_frequency.py`) pinning the scaling law, weekend handling, and — explicitly — that using 12 instead of 261 is a silent 4.66x error that raises nothing.

**One real bug found:** `quantile_weights` returned an **all-zero book** for a single-asset panel instead of raising. The per-row "fewer than 2 names" branch skipped every date, so a strategy that is *impossible to form* came back looking like a strategy that *chose not to trade*. On a one-instrument study that is precisely the wrong answer. It now raises. (Section 2 of the study output demonstrates the new behaviour.)

---

## 9. Conclusion

**Is the code correct?** Yes. 97 tests pass, including the look-ahead traps from the monthly study and 14 new frequency tests. The annualisation constant is measured from the data and cross-checked before any scaled statistic is reported.

**Is there an edge?** No. On XAUUSD daily over this window, time-series momentum **underperforms buy and hold** on return (−2.2%/yr), on Sharpe (1.088 vs 1.178), and on drawdown (identical, −27.3%). It gave no crash protection whatsoever — its five worst days are gold's five worst days. Its up/down capture is asymmetric in the wrong direction. One of 22 grid cells edged past the benchmark, which is what you expect from 22 tries.

The honest framing: **this window is hostile to trend-following by construction.** A 2.4x bull market with one sharp correction at the end is close to the worst case for a filter whose only lever is stepping aside — there was almost never a good time to be out. That is not evidence that trend-following never works on gold; it is evidence that it did not work *here*, and that a 3.65-year single-asset sample cannot settle the question either way.

**What would change the conclusion:** a sample spanning gold's 2012–2015 bear market (where a trend filter has something to do), a multi-asset trend portfolio where the cross-section supplies diversification, or a faster signal — the 21–63 day cells are the only ones that even react in time, and testing those properly needs intraday cost modelling this study does not have.
