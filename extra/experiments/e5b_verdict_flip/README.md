# E5b: Verdict-Flip Sweep

## Purpose

Measure how often two named semantic rules answer "satisfied" vs. "violated" oppositely on the same signal.

## Runner

Evaluate a spec on a signal under one backend; record the robustness and the Boolean verdict at a fixed time index.

| Kind           | Items                       |
| -------------- | --------------------------- |
| Conditions     | `spec`, `signal`, `backend` |
| Parameters     | verdict time index          |
| Recorded facts | robustness vector, verdict  |

## Batch and Aggregate

Batch varies the signal (sweeping the final sample over a range) and the backend.

Aggregate compares, for each fixed spec and signal, the verdicts of the two backends; reports flip counts and `flip_rate` (and `inversion_rate` against a reference signal).
