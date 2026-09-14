"""momentum: a small, auditable momentum research toolkit.

Implements the concepts laid out in the "Momentum Trading" reference sheet:
simple returns, cumulative-return momentum signals, price/SMA momentum,
rolling z-score signals, cross-sectional rank portfolios, time-series
(absolute) momentum, risk controls and the standard performance statistics.

Design rules enforced throughout (see tests/):
  * A signal computed from data up to and including bar t may only be acted
    on from bar t+1 onward. Every position series is shifted before being
    multiplied by returns.
  * Nothing returns a "safe looking" neutral default on a failure path.
    Bad input raises.
"""

from momentum import data, signals, backtest, metrics, stats, synth  # noqa: F401

__all__ = ["data", "signals", "backtest", "metrics", "stats", "synth"]
__version__ = "1.0.0"
