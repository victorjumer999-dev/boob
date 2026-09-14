# Is there an edge?

Short answer: **not in returns. There is one in drawdowns, and it is smaller and less certain than it first looks.**

---

## The constraint that shapes this

Across three studies this project has now tested **97 configurations** on overlapping data. At that point, searching harder is not research — the best of 97 tries looks good by construction. The Bonferroni threshold for 97 trials is **|t| = 3.47**, and that is the bar anything new has to clear.

So this is **one** strategy, run once. Every design choice was dictated by what the earlier studies already established, not by looking for what scores well:

| Choice | Forced by |
|---|---|
| Time-series, not cross-sectional | XS momentum failed at 0.148 (7 equities), 0.041 (13 assets), pooled predictive β negative |
| Blend 3/6/12-month lookbacks | Grid Sharpe ranged 0.19–1.19; that spread is noise, and taking its maximum is how a backtest lies |
| Inverse-vol position sizing | Volatility spans 12%–71% here; equal weights produced a −99.8% drawdown |
| 10% portfolio vol target | The only lever that improved risk-adjusted return in *every* study |

13 assets, 4 asset classes, 297 months (2002-01 → 2026-09), 10bp one-way.

---

## The result

| Strategy | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| Trend long/flat | 9.22% | 12.01% | **0.797** | **−24.1%** |
| Trend long/short | 5.34% | 11.14% | 0.523 | −28.4% |
| Equal-weight basket | 10.99% | 14.48% | 0.796 | −45.3% |
| SPY | 9.94% | 14.79% | 0.717 | −50.8% |

Sharpe 0.797 against the basket's 0.796. That is not an edge, it is a rounding difference.

### The comparison that settles it

The fairest benchmark is not SPY. It is **the same basket, held at 83% weight, never traded** — which matches the trend portfolio's volatility with no signal, no model, and no turnover:

| | CAGR | Vol | Sharpe | Max DD | Turnover |
|---|---:|---:|---:|---:|---:|
| Trend following | 9.22% | 12.01% | 0.797 | **−24.1%** | 5.6/yr |
| **Hold 83% of the basket, never trade** | 9.20% | 12.01% | 0.796 | −39.1% | **0** |

**Excess return: +0.01%/yr. HAC t = 0.00.**

Same return, same volatility, same Sharpe — for 5.6 portfolio turns a year. On the return dimension, twenty-five years of trend following bought precisely nothing that de-levering a buy-and-hold did not already provide for free.

---

## The permutation test

The sharpest evidence. Shuffle the signal's *rows* — this destroys the alignment between signal and subsequent return while preserving the positions, the sizing, the asset returns and the exposure profile exactly. If the strategy's performance survives that, it never came from timing.

```
real strategy (unlevered)     Sharpe 0.716
randomly-timed same positions Sharpe 0.602 mean, sd 0.118, 95th pct 0.794

the real strategy sits at the 84th percentile of the null
empirical p-value = 0.16
```

**A randomly-timed version of the same book captures 84% of the strategy.** The timing contributes 0.11 Sharpe, comfortably inside noise. The performance is an exposure effect wearing a signal's clothes.

---

## The one thing that survives

At identical return and identical volatility, the maximum drawdown is **−24.1% vs −39.1%** — 15 points shallower. That is a real difference and it is worth having.

Three caveats, all of which cut against it:

1. **It is not statistically established.** Paired block-bootstrap of the drawdown difference: −10.1pp, 95% CI **[−32.4pp, +13.8pp]**. The probability that trend's drawdown is genuinely shallower is **0.795** — likely, not proven.
2. **The capture asymmetry is backwards, for the third time.** In the benchmark's 20 worst months, trend captures **55.1%** of the downside but only **45.8%** of the upside in the best 20. A working defensive strategy has it the other way round. The same inversion appeared in the monthly sector study (47.4% up vs 46.5% down) and the daily gold study (94.6% up vs 98.7% down).
3. **The mechanism is not crash avoidance.** Month by month, trend is no better at dodging bad months. It reduces drawdown *depth* by de-risking through extended declines — a path property, not a prediction. Which is exactly what a volatility-responsive position sizer does, with or without a momentum signal attached.

---

## Verdict

| Test | Result |
|---|---|
| Beats passive benchmark at Bonferroni \|t\| = 3.47 | **FAIL** (t = 0.00) |
| Beats the permutation null at p < 0.05 | **FAIL** (p = 0.16) |
| Positive Sharpe in both halves | PASS (0.912, 0.695) |
| Shallower drawdown than the benchmark | PASS (−24.1% vs −39.1%) |

The two failures are the two tests that ask whether there is a *return* edge. The two passes are risk properties — and note the benchmark was also positive in both halves, and beat trend in the second (0.834 vs 0.695).

**So: no return edge, in anything this project tested.** Four momentum formulations, three frequencies, four asset classes, 98 configurations. The single consistent finding across all of it is that every apparent edge resolved into reduced exposure once measured against a benchmark holding the same amount of risk.

**What is real:** volatility-responsive position sizing produces a shallower drawdown path at the same return. That is a genuine, repeatable property — it showed up in all three studies — but it is risk management, and it should be bought as cheaply as possible rather than dressed up as alpha.

## What would actually be worth trying next

Not more of this. The honest conclusion is that this data cannot support a return edge, and the remaining options are about changing the data, not the parameters:

- **A real futures universe with roll-adjusted total returns.** The commodity legs here are spot prices, which nobody can hold. Roll yield is where a large part of documented commodity trend returns actually live, and this project has never been able to see it.
- **Signals that are not price.** Everything tested here is a function of past prices alone. Carry, value, and positioning data are independent families with their own documented evidence; none of them are testable with this feed.
- **A cross-section wide enough to diversify idiosyncratic noise.** Thirteen assets is better than seven and still an order of magnitude short of what the momentum literature uses.

If none of those are available, the defensible position is the one the data supports: hold a diversified basket, size it by volatility, and don't pay turnover for a timing signal that a random permutation reproduces 84% of.
