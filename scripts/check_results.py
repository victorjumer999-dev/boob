#!/usr/bin/env python3
"""Golden-result check: the study must reproduce its published numbers.

A backtest whose headline figures drift silently between library versions is
the same class of defect as the ones in REPORT.md section 7 -- it does not
crash, it just quietly reports something else. This pins the numbers quoted
in REPORT.md so a pandas/numpy upgrade cannot move them unnoticed.

Tolerances are loose enough to absorb last-place floating-point differences
across platforms and tight enough that any real change in behaviour fails.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "results", "report.json")

# (json path, expected, absolute tolerance)
EXPECTED = [
    ("data.months", 321, 0),
    ("data.dropped_partial_bars", 1, 0),
    ("summary.XS long/short.bars", 296, 0),
    ("summary.XS long/short.sharpe", 0.147884, 5e-4),
    ("summary.XS long/short.cagr_%", 1.0851, 5e-3),
    ("summary.XS long/short.max_dd_%", -41.5915, 5e-3),
    ("summary.XS long-only.sharpe", 0.765590, 5e-4),
    ("summary.TS absolute (long/flat).sharpe", 0.885990, 5e-4),
    ("summary.TS absolute (long/flat).max_dd_%", -19.1957, 5e-3),
    ("summary.SPY buy & hold.sharpe", 0.719775, 5e-4),
    ("summary.SPY buy & hold.max_dd_%", -50.7976, 5e-3),
    ("hac.t_stat", 0.854565, 5e-4),
    ("hac.lags_used", 11, 0),
    ("predictive_regression.beta_t", 0.308590, 5e-4),
    ("bucket_spread_monthly_pct", 0.2269, 5e-4),
    ("grid.n_trials", 24, 0),
    ("grid.bonferroni_hits", 11, 0),
    ("constant_exposure_control.up_capture_pct", 47.36, 5e-2),
    ("constant_exposure_control.down_capture_pct", 46.47, 5e-2),
]


def dig(blob, path):
    node = blob
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"{path}: missing at segment {key!r}")
        node = node[key]
    return node


def main() -> int:
    if not os.path.exists(REPORT):
        print(f"FAIL: {REPORT} not found -- run scripts/run_backtest.py first")
        return 1
    with open(REPORT) as fh:
        blob = json.load(fh)

    failures = []
    for path, expected, tol in EXPECTED:
        try:
            actual = dig(blob, path)
        except KeyError as exc:
            failures.append(str(exc))
            continue
        delta = abs(float(actual) - float(expected))
        if delta > tol:
            failures.append(
                f"{path}: expected {expected} +/- {tol}, got {actual} (delta {delta:.6g})"
            )
        else:
            print(f"  ok  {path} = {actual}")

    if failures:
        print("\nGOLDEN RESULT CHECK FAILED:")
        for line in failures:
            print(f"  - {line}")
        print(
            "\nIf this change is intentional, update EXPECTED here AND the "
            "corresponding figures in REPORT.md -- they are quoted in prose."
        )
        return 1

    print(f"\nall {len(EXPECTED)} golden results reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
