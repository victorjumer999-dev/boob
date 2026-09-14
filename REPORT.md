# Momentum Trading — reproduction report

**Universe:** 6 SPDR sector ETFs (XLE, XLF, XLK, XLP, XLV, XLY) + SPY
**Frequency:** monthly, split/dividend-adjusted (Alpha Vantage)
**Sample:** 321 bars, Dec-1999 → Aug-2026; evaluation window Jan-2002 → Aug-2026 (296 months)
**Costs:** 10bp one-way on traded notional (~2-3bp is realistic for these ETFs, so this is conservative)
**Signal:** 12-1 compound momentum unless stated otherwise

---

## 1. What the image claims

The reference sheet asserts five things. Each was implemented and tested:

| # | Claim on the sheet | Verdict here |
|---|---|---|
| 1 | Return persistence: past winners keep winning | **Directionally yes, economically tiny.** Monotone ladder, 2.8%/yr winner−loser spread |
| 2 | Formation/holding: long top quantile, short bottom | **Fails.** Sharpe 0.15, t = 0.86, −8.4%/yr vs SPY |
| 3 | Lookbacks of 3/6/12 months | **12 is best, 3 is worst** — and no long/short lookback is significant |
| 4 | Time-series vs cross-sectional | **TS is the better construction**, but for risk, not return |
| 5 | Risk management / crash risk (2009, 2020) | **Confirmed, emphatically.** −16.6% in Feb-2021 while SPY was +2.8% |

The sheet's own equity chart shows the top decile at **+512%** versus the market's **+243%**, and a bottom decile at **−41%**. That picture does not reproduce.

---

## 2. Headline results

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover | Cost drag |
|---|---:|---:|---:|---:|---:|---:|
| **XS long/short (12-1)** | **1.09%** | 15.6% | **0.148** | −41.6% | 10.3x/yr | 1.03%/yr |
| XS long-only (top tercile) | 11.18% | 15.5% | 0.766 | −38.4% | 5.4x/yr | 0.55%/yr |
| XS L/S, 10% vol target | 1.49% | 11.1% | 0.189 | −25.3% | 9.0x/yr | 0.90%/yr |
| TS absolute (long/short) | 6.44% | 11.1% | 0.621 | −34.8% | 2.3x/yr | 0.23%/yr |
| **TS absolute (long/flat)** | 8.64% | 9.9% | **0.886** | **−19.2%** | 1.2x/yr | 0.12%/yr |
| Equal-weight sectors | 10.13% | 14.6% | 0.735 | −48.9% | 0 | 0 |
| SPY buy & hold | 9.99% | 14.8% | 0.720 | −50.8% | 0 | 0 |

All strategies are evaluated over the identical window so no benchmark gets a free extra year.

---

## 3. Is the cross-sectional effect real?

**The ladder is monotone.** Sorting sectors into terciles on prior 12-1 momentum and measuring the *next* month:

| Bucket | Next-month return | Annualised | t-stat |
|---|---:|---:|---:|
| 1 (losers) | 0.728% | 9.10% | 2.79 |
| 2 | 0.847% | 10.65% | 4.39 |
| 3 (winners) | 0.955% | 12.09% | 4.45 |

Winner − loser = **0.227%/month (2.76%/yr)**. The sign is right and the ordering is correct. But note what those t-stats are testing: that each bucket's return differs from *zero*. All three are long equity books, so all three inherit the equity risk premium. That is what makes bucket 1 — the **losers** — significant at t = 2.79.

**The spread does not survive implementation.** Harvesting 2.76%/yr requires a long/short book that:
- trades 10.3x/yr, costing 1.03%/yr at 10bp;
- carries a short leg whose own drawdowns are violent.

Result: Sharpe **0.148**, Newey-West t = **0.855** (11 lags, floored at the 12-month overlap), moving-block 95% CI on Sharpe **[−0.248, +0.461]** — straddling zero — and P(Sharpe > 0) = 0.73.

**The pooled predictive regression finds nothing.** Fitting the sheet's third formula, `R_{t+1} = α + β·M_t + ε`, across 1,848 asset-months:

```
α = 0.7997%/mo  (t = 2.51)      <- the equity premium, not the signal
β = 0.00479     (t = 0.31)      <- the signal
R² = 0.00027
```

β is indistinguishable from zero. The only statistically alive term is the intercept — i.e. "stocks went up".

---

## 4. The parameter grid, and why 11 "significant" cells prove nothing

24 configurations (6 lookbacks × skip 0/1 × long-only/long-short):

- Cells with |t| > 1.96: **12**. Chance alone predicts ~1.2, so this looks like a strong result.
- Bonferroni threshold for 24 tests: |t| > **3.08**. Cells clearing it: **11**.

Eleven cells clearing a multiple-testing-corrected threshold would normally be decisive. It is not, for one reason: **every single one of them is long-only.** Not one long/short cell reaches even |t| > 1.96 — the best is 12-0 at t = 1.33.

What the long-only t-stats measure is whether a long equity portfolio has a positive mean. It does. Tested against the right benchmark, the signal disappears:

| Strategy | Benchmark | Excess | t |
|---|---|---:|---:|
| XS long-only | Equal-weight sectors | +1.08%/yr | +0.74 |
| XS long/short | SPY | −8.35%/yr | −1.67 |
| TS long/flat | Equal-weight sectors | −1.95%/yr | −1.12 |
| TS long/flat | SPY | −1.85%/yr | −0.90 |

Not one is significant, and three of the four are negative. **Momentum selection adds no measurable return over simply holding the same six sectors.**

---

## 5. The one thing that does work — and what it actually is

Time-series momentum long/flat has the best Sharpe in the study (0.886 vs SPY's 0.720) and less than half the drawdown (−19.2% vs −50.8%). That is a large, real difference in path.

But it is **not return timing**. The filter is invested 76% of the time on average (fully invested 45% of months, fully flat 6%). Compared against a passive book held at that same constant 76% exposure:

```
constant-exposure twin   Sharpe 0.735   CAGR 7.82%   maxDD -39.5%
TS momentum long/flat    Sharpe 0.886   CAGR 8.64%   maxDD -19.2%
timing excess            +0.64%/yr,  t = +0.52        <- not significant
up-capture   47.4%   vs   down-capture  46.5%         <- asymmetry -0.9pp
```

The capture ratios are the tell. In the 20 worst months for the sector book the filter loses 4.09% instead of 8.79% — but in the 20 best months it makes 4.18% instead of 8.82%. It cuts both tails by the *same* proportion. That is what holding less equity does, not what foresight does.

What it genuinely delivers is **volatility timing**: it is out of the market during clustered-volatility regimes, so it reduces risk by more than the mechanical 76% scaling would (Sharpe 0.886 vs the twin's 0.735, max DD halved). That is a real and useful property for a risk budget. It is not alpha, and the +0.64%/yr return advantage is not statistically distinguishable from zero.

---

## 6. Momentum crash risk — the sheet's one unambiguous hit

The sheet flags crash risk in 2009 and 2020. Confirmed, and worse than advertised. The five worst months for the long/short book:

| Month | Strategy | SPY |
|---|---:|---:|
| 2021-02 | **−16.61%** | +2.78% |
| 2020-04 | **−14.58%** | +12.70% |
| 2009-04 | **−12.74%** | +9.93% |
| 2020-11 | **−11.73%** | +10.88% |
| 2016-11 | **−11.43%** | +3.68% |

Every one is a sharp market *rally*. Momentum's short leg is loaded with beaten-down names precisely when they rebound hardest. The strategy's deepest drawdown began Feb-2009 and had **not recovered by Aug-2026** — 154 months to the trough and no new high in 17 years.

Sub-period Sharpe:

| Strategy | 2001-08 | 2009-16 | 2017-26 |
|---|---:|---:|---:|
| XS long/short | 0.46 | −0.09 | 0.10 |
| XS long-only | 0.23 | 0.88 | 1.05 |
| TS long/flat | 0.66 | 0.99 | 0.94 |
| SPY | −0.04 | 1.03 | 1.00 |

The long/short book's only decent period is the one where the market went nowhere. Since 2009 it has been flat-to-negative throughout.

---

## 7. Bugs found and fixed while building

The test suite caught five defects. Four were in code I had just written; all four are the kind that produce plausible output rather than a crash.

1. **Warm-up months padded into the track record.** Before the 12-month signal existed the book was all zeros, and a zero-weight bar produces a hard 0.00% return. Those months were being counted as real observations — shrinking the standard deviation, inflating the month count, and back-dating the reported start (a strategy that first traded in 1997 showed a 1996 start and a flattered Sharpe). Now only the *leading* flat run is trimmed; a mid-sample step to cash is a genuine decision and stays in.

2. **`sd == 0` never fired.** `pd.Series([0.01]*30).std(ddof=1)` returns `1.76e-18`, not `0.0`. The exact-equality guard therefore never triggered and the Sharpe of a constant return stream came back as **~2e16** — finite, plausible-looking, and silently wrong. Replaced with a tolerance scaled to the data.

3. **`max_drawdown` missed first-bar losses.** The running peak was taken from the equity curve, which starts *after* the first return, so a strategy whose worst loss is immediate reported a max drawdown of 0.00%. The starting capital of 1.0 is now seeded into the peak.

4. **The partial-month guard was inverted.** It compared `asof` against the bar's own timestamp (2026-09-11) rather than the end of that bar's calendar month (2026-09-30), so on a 2026-09-14 run the incomplete September bar sailed straight through.

5. **`np.where(momentum > 0, 1, -1)`** — copied from the sheet's own code block — puts `NaN` in the `-1` bucket, because NaN comparisons are falsy. Applied literally, that holds a **maximally short book through the entire 12-month warm-up** on no information at all. `timeseries_signal` keeps NaN as NaN.

Two further corrections to the sheet's formulas, made deliberately rather than as bug fixes:

- `momentum = returns.rolling(lookback).sum()` is an arithmetic sum of returns. At a 12-month horizon on volatile assets that diverges badly from the actual return (12 months of +10% compounds to +214%, not +120%). `compound_momentum` is used throughout; the arithmetic version is retained as `cumulative_momentum` for fidelity to the sheet.
- The sheet's formula has no skip month. 12-1 (skip the most recent month, to dodge short-horizon reversal) is the standard convention and is the default here — though on this data it makes things slightly *worse*, not better (12-0 Sharpe 0.242 vs 12-1's 0.148).

---

## 8. Honest limitations

- **Six assets is a thin cross-section.** The sheet shows deciles; six sectors support terciles of two names each. A genuine decile sort needs hundreds of names. A real cross-sectional momentum effect is well documented in individual stocks — the null found here is a null **for sector rotation on this universe**, not a refutation of momentum as a phenomenon. The daily/individual-stock endpoints were premium-gated on this API key.
- **Monthly frequency** is faithful to Jegadeesh & Titman but cannot speak to intra-month or faster momentum.
- **One market, one asset class, 24 years.** Momentum's strongest published evidence is cross-asset and cross-country; neither is tested here.
- **Costs are modelled as proportional.** No market impact, no borrow cost on the short leg — both of which would make the long/short result *worse*, not better.
- **The 2002 start** excludes the dot-com bust's first two years, which the 12-month warm-up consumes.

---

## 9. Conclusion

Two separate questions, answered separately.

**Is the code correct?** Yes, with evidence. 83 tests pass, including an oracle-signal look-ahead trap with a control demonstrating it has teeth (Sharpe 4.6 unshifted vs <1.0 shifted), planted-edge recovery on synthetic panels, a no-edge check on random walks, and five fixed defects documented above. The engine finds a synthetic momentum effect of known sign and size, and reports nothing on a random walk.

**Is there an edge?** On this universe, no. The cross-sectional long/short strategy the reference sheet advertises earns a Sharpe of 0.148 with a t-statistic of 0.855 and a bootstrap confidence interval that straddles zero. It underperforms SPY by 8.4%/yr, has not made a new high since 2009, and loses 16% in a month when the market rallies. Its long-only cousin looks respectable only because it is a long equity portfolio; measured against the same six sectors held passively it adds 1.1%/yr with t = 0.74.

The one construction worth keeping — time-series momentum, long/flat — earns its Sharpe by holding less equity in volatile regimes, not by predicting returns. Its up- and down-capture are symmetric to within 1 percentage point. Use it as a risk control. Do not model it as alpha.

**What would change this conclusion:** an individual-stock universe of a few hundred names (where the published evidence actually lives), a cross-asset panel spanning bonds/FX/commodities, or a shorter holding period with proper transaction-cost modelling. The sheet's +512% chart is not reproducible on sector ETFs, and nothing in this sample suggests it would be.
