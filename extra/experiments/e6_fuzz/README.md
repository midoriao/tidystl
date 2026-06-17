# E6: Randomized Cross-Backend Divergence Sweep

> Not part of the paper reproduction; preserved as supplementary evidence.

## Purpose

The statistical sibling of `e1_divergence`. Where e1 uses curated diagnostic
cases compared against the real external tools, e6 fuzzes random
`(formula, signal)` pairs across all the registered tidystl backends and measures how often they
agree. Output is a divergence map: a pairwise agreement matrix plus
per-operator divergence rates, not a single headline percentage. Pure
Python, in-process, no real tools.
