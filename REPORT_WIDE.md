# Wide universe + the gold/silver question

Two things were asked for: widen the universe and re-test, and check whether silver — correlated to gold — influences how gold moves.

---

## Part 1 — Gold and silver

**Sample:** 3,987 weekday bars, 2011-06-02 → 2026-09-11 (15.3 years), both weekend-cleaned.

### The correlation claim is right, and it matters

| | |
|---|---:|
| Daily return correlation | **0.781** |
| 1-yr rolling correlation | min 0.656, median 0.783, max 0.857 |
| Silver's beta to gold | **1.46** (t = 26.2), R² = 0.61 |
| Silver's volatility | **1.88x** gold's |

The rolling correlation never drops below 0.66 in fifteen years. This is a persistent structural link, not an episodic one — silver behaves like a high-beta version of gold.

### The influence claim doesn't survive testing

Does yesterday's move in one predict today's move in the other?

| Predictor | Target | β | HAC t | R² |
|---|---|---:|---:|---:|
| **XAG[t−1]** | **XAU[t]** | −0.0060 | **−0.42** | 0.00012 |
| XAU[t−1] | XAG[t] | +0.0131 | +0.35 | 0.00005 |
| XAU[t−1] | XAU[t] | −0.0005 | −0.02 | — |
| XAG[t−1] | XAG[t] | −0.0172 | −0.62 | — |

Joint test — does silver add anything beyond gold's own lag?

```
XAU[t] ~ XAU[t-1] + XAG[t-1]
   gold_lag1    +0.0210  (t +0.46)
   silver_lag1  -0.0147  (t -0.56)      R^2 = 0.00030
```

Every t-statistic is under 0.7; the Bonferroni threshold for these six tests is 2.64. R² of 0.0003 means silver's previous move explains **three hundredths of one percent** of gold's next move.

Traded directly — using silver's momentum to time gold — it is worse than useless:

| | CAGR | Sharpe |
|---|---:|---:|
| Buy & hold gold | 26.19% | **1.222** |
| Gold's own signal → gold | 22.13% | 1.081 |
| **Silver's signal → gold** | 17.62% | **0.914** |

**The distinction that matters:** gold and silver move together *contemporaneously* — that is real, large, and persistent. But moving together is not the same as one leading the other. If silver reliably predicted gold a day ahead, that would be an arbitrage in two of the most heavily traded instruments on earth, and it isn't there.

Where the correlation genuinely bites is **risk**, not prediction. An equal-weight gold+silver book has a diversification ratio of **0.957** — 1.00 means no benefit whatsoever. Two assets this correlated are close to one position, and that has a direct consequence for Part 2.

---

## Part 2 — Widening the universe

**13 instruments, 4 asset classes, 2001-01 → 2026-09 (309 months):** 7 equity ETFs, 3 energy (WTI, Brent, natural gas), 3 metals (copper, gold, silver).

### Breadth did what it was supposed to do

| | Average pairwise correlation |
|---|---:|
| Original 7 (all equities) | 0.593 |
| **Wide 13** | **0.284** |

Cross-class correlations are 0.17–0.21 against 0.59 within equities. The universe is genuinely more diversified — the precondition for cross-sectional momentum was met.

### The strategy still doesn't work

| Strategy | CAGR | Vol | **Sharpe** | Max DD |
|---|---:|---:|---:|---:|
| **Equal-weight (13)** | 9.88% | 14.57% | **0.722** | −45.3% |
| TS long/flat (13) | 7.12% | 10.43% | 0.713 | **−26.4%** |
| SPY buy & hold | 9.00% | 15.05% | 0.651 | −50.8% |
| XS long-only (13, risk parity) | 7.50% | 26.96% | 0.404 | −74.5% |
| XS long-only (13) | 4.84% | 30.29% | 0.307 | −76.2% |
| XS long/short (orig 7) | 1.10% | 16.09% | 0.149 | −39.5% |
| **XS long/short (13, RP)** | −6.67% | 33.65% | **0.041** | −89.7% |
| XS long/short (13) | −21.12% | 46.83% | −0.170 | −99.8% |

Widening the universe made the headline strategy **worse**, not better:

```
wide 13 RP    +1.38%/yr   HAC t +0.24    Sharpe  0.041  95% CI [-0.220, +0.474]
original 7    +2.40%/yr   HAC t +0.89    Sharpe  0.149  95% CI [-0.180, +0.492]
difference (13 RP - 7):  -1.02%/yr,  HAC t = -0.19
```

The pooled predictive regression is **negative**: β = −0.0168 (HAC t = −1.49, R² = 0.0028, n = 3,739). Across 24 grid cells, not one long/short configuration beat SPY, and the best long-only cell (0.404) is less than two thirds of the passive benchmark's 0.651.

### Why it got worse — and the fix that was needed anyway

The first run produced a −99.8% drawdown. The cause is visible in one statistic: **annualised volatility ranges from 12% (XLP) to 71% (natural gas), a 5.7x spread.** An equal-weighted book across assets that heterogeneous is a natural-gas bet with some equities attached.

So I added inverse-volatility position sizing (`signals.inverse_vol_weights`), which scales each position by trailing volatility and renormalises to the same gross exposure — reallocating risk without changing leverage. It helped materially:

| | Vol | Sharpe | Max DD |
|---|---:|---:|---:|
| XS long/short, equal weight | 46.8% | −0.170 | −99.8% |
| **XS long/short, risk parity** | 33.7% | **+0.041** | −89.7% |

It repaired the *risk* profile and could not manufacture *return*. That is the same pattern as both earlier studies.

**The lesson is that the two changes are not independent.** Widening a universe without fixing position sizing actively hurts, because breadth is what introduces the volatility heterogeneity that breaks equal weighting. Anyone who widens and doesn't re-size gets a worse strategy and may well conclude, wrongly, that breadth didn't help.

### The one result worth keeping

Time-series momentum across the wide universe is the only active strategy that is competitive:

| | CAGR | Vol | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| TS long/flat (13) | 7.12% | 10.43% | 0.713 | **−26.4%** |
| SPY buy & hold | 9.00% | 15.05% | 0.651 | −50.8% |
| TS scaled to SPY's volatility (1.44x) | 10.04% | 15.05% | 0.713 | −36.5% |

At equal volatility it earns +0.94%/yr more than SPY with a drawdown 14 points shallower. But the honest reading: **HAC t = +0.33**, and the Sharpe intervals overlap almost completely — TS [0.339, 1.183] vs SPY [0.232, 1.187]. The drawdown reduction is structural and believable (diversification plus the ability to step to cash). The return advantage is not statistically distinguishable from zero.

---

## Code changes

| Change | Why |
|---|---|
| `signals.inverse_vol_weights` | Equal weights are not equal risk across a 5.7x volatility spread. Causal by the library's convention; 7 new tests including a look-ahead check. |
| `BacktestConfig.risk_parity`, `.rp_window`, `.min_names` | Position sizing and a minimum-names floor for ragged panels, threaded through the engine. Defaults preserve existing behaviour. |
| `stats.ols_hac` | General HAC regression, factored out of `predictive_regression` (which now reuses it — all 19 golden results reproduce bit-identical). Needed for the lead-lag tests. |
| **`metrics.is_ruined`** | The grid produced "−103% drawdown" — a book compounded past zero. `(1+r).cumprod()` carries on into negative equity and keeps emitting CAGRs and drawdowns that look like numbers but describe nothing. Now detected and surfaced; wiped-out cells print `WIPED OUT` instead of a fake statistic. |
| `data.audit` empty-column guard | An all-NaN column raised an opaque `IndexError` from inside pandas. Now names the likely cause (month-start vs month-end index mismatch), which is exactly what had happened. |

Tests: 100 → **111**.

### Data caveat, stated plainly

**The commodity series are spot prices, and spot commodities are not investable.** A real position rolls futures and earns or pays the roll yield, which for energy frequently dominates the spot move. These results test whether the momentum *signal* ranks assets usefully. They are not an achievable P&L for the commodity legs, and the commodity CAGRs in particular should not be read as returns anyone could have earned.

---

## Conclusion

**On silver:** the correlation is real, strong, and persistent (ρ = 0.78, β = 1.46, never below 0.66 on a rolling year) — and it matters, because it means gold and silver are close to a single position rather than two. The predictive claim does not hold: silver's prior move explains 0.03% of gold's next move, with every t-statistic under 0.7 over fifteen years.

**On widening:** the universe is genuinely more diversified (correlation 0.593 → 0.284), and cross-sectional momentum still has no edge — it is slightly *worse* than on the original seven, and the pooled predictive coefficient is negative. Three studies now, three null results, on monthly equities, daily gold, and a 13-asset multi-class panel.

What keeps surviving across all three is the risk side: volatility targeting, inverse-vol sizing, and time-series momentum's drawdown reduction all work and are all repeatable. What never survives is return prediction. That is not a failure of the tests — it is a reasonably consistent finding, and it points at where any remaining effort should go.
