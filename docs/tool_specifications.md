# External Tool Specifications

This document records the current implementation status of `tidystl` against
external tools. For user-facing backend selection guidance and worked
examples of the semantic differences, see `docs/usage.md`
(Backends and Semantics).

## Breach

### Overview

Breach is a MATLAB toolbox for temporal-logic-based analysis of dynamical and
hybrid systems. In `tidystl`, the relevant reference point is Breach's STL
robustness monitoring behavior on piecewise-linear signals.

Reference URLs:

- https://github.com/decyphir/breach
- https://github.com/decyphir/breach/blob/master/Doc/BreachDoc.pdf

Notable characteristics:

- dense-time interpretation
- piecewise-linear interpolation between samples
- bounded-future STL robustness
- MATLAB-based ecosystem

### Feature coverage

- [x] Predicate robustness with `>=`, `>`, `<=`, `<`, `==`
- [x] Arithmetic predicates with `+`, `-`, `*`, `/`, `^`, `abs`, `sqrt`
- [x] Boolean STL operators `not`, `and`, `or`
- [x] Bounded `always` / `eventually`
- [x] Bounded `until`
- [x] Piecewise-linear interpolation inside temporal windows
- [x] Boundary behavior matched to the current Breach fixtures
- [x] Trace extraction for evaluated subformulas
- [ ] Full Breach language compatibility
  Current status: only the STL robustness fragment implemented in this repository is covered.
- [ ] MATLAB API compatibility
  Current status: no MATLAB interface is provided.
- [ ] Full Breach feature coverage outside STL robustness
  Current status: modeling, simulation, and other toolbox features are out of scope.

## RTAMT

### Overview

RTAMT is a runtime monitoring library for STL and IA-STL. In `tidystl`, the
relevant reference point is RTAMT's discrete-time bounded-future STL
robustness semantics.

Reference URLs:

- https://github.com/nickovic/rtamt
- https://github.com/nickovic/rtamt/blob/master/README.md

Notable characteristics:

- discrete-time semantics are explicitly documented
- offline and online monitoring modes exist upstream
- dense-time support also exists upstream
- end-of-trace behavior differs from Breach-style clamping

### Feature coverage

- [x] Predicate robustness with `>=`, `>`, `<=`, `<`, `==`
- [x] Arithmetic predicates with `+`, `-`, `*`, `/`, `^`, `abs`, `sqrt`
- [x] Boolean STL operators `not`, `and`, `or`
- [x] Bounded `always` / `eventually` on a uniform discrete grid
- [x] Bounded `until` on a uniform discrete grid
- [x] RTAMT-style end-of-trace behavior for bounded future operators
- [x] RTAMT-style witness-exclusive left prefix for bounded `until`
- [x] Trace extraction for evaluated subformulas
- [ ] Dense-time RTAMT compatibility
  Current status: `RtamtBackend` only implements the discrete-time path.
- [ ] Online monitor compatibility
  Current status: no RTAMT-style online monitor is implemented.
- [ ] RTAMT parser / syntax compatibility
  Current status: formulas still go through the `tidystl` parser.
- [ ] RTAMT past-time operators
  Current status: operators such as `since`, `once`, and `historically` are not implemented.
- [ ] RTAMT-specific extra operators
  Current status: operators such as `next`, `prev`, `rise`, `fall`, `xor`, `->`, `<->`, `!==`, and `exp` are not implemented.
- [ ] Non-uniform discrete sampling compatibility
  Current status: `RtamtBackend` requires a strictly increasing uniform time grid and aligned interval bounds.

## STLCG++

### Overview

STLCG++ is a PyTorch/JAX toolbox for differentiable STL robustness evaluation.
In `tidystl`, the relevant reference point is the PyTorch package
`stlcgpp` operating over uniform discrete traces.

Reference URLs:

- https://github.com/UW-CTRL/stlcg-plus-plus
- https://uw-ctrl.github.io/stlcg/

Notable characteristics observed from `stlcgpp==0.0.2`:

- discrete-time semantics over sample indices
- differentiable masking-based implementations of bounded future operators
- default end-of-trace padding emits large sentinels (`-1e9`) rather than
  compatibility-friendly infinities
- `padding="last"` produces a more practical last-value extension target
- bounded `until` uses a witness-inclusive left prefix, which matches the
  current Breach-style discrete fixtures rather than RTAMT's witness-exclusive
  rule

### Feature coverage

- [x] Ground-truth fixture generation for the current discrete compatibility cases
  Current status: `extra/other_tools/stlcgpp/generate_ground_truth.py` emits CSV fixtures under
  `tests/stlcgpp_ground_truth` using `padding="last"`.
- [x] Runtime backend over uniform discrete traces
  Current status: `StlcgppBackend` implements sample-indexed bounded-future STL
  with last-value extension for `always` / `eventually` and a witness-inclusive
  left prefix for bounded `until`.
- [ ] Dense-time compatibility
  Current status: current fixture generation is index-based and does not model dense-time interpolation.
- [ ] Explicit sampling-grid validation
  Current status: the upstream package operates on sample positions only; a future backend should define and enforce `tidystl`'s grid requirements explicitly.
