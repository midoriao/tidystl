# tidystl User Manual

tidystl computes quantitative robustness of Signal Temporal Logic (STL)
specifications over batches of time series. It is numpy-native, batch-first,
and puts the evaluation semantics under explicit user control: the default
backend implements principled piecewise-linear (PL) dense-time semantics,
and additional backends reproduce the runtime behavior of Breach, RTAMT,
and STLCG++ for validation and migration.

*Robustness* refers to the quantitative semantics of STL [1, 2]: instead of
a Boolean verdict, each formula evaluates to a real number whose sign
indicates satisfaction (positive) or violation (negative) and whose
magnitude indicates the margin. This manual assumes basic familiarity with
STL; see the [References](#references) for background.

Every Python code block in this manual is self-contained and executable;
the test suite extracts and runs all of them (`tests/test_usage_doc.py`;
blocks that need torch are executed when the `torch` extra is installed
and syntax-checked otherwise).

**Document map**

| Document | Content |
|---|---|
| This manual | Installation, concepts, semantics, recipes, troubleshooting |
| [`reference.md`](reference.md) | API reference: grammar, signatures, contracts |
| [`design.md`](design.md) | Architecture and design rationale |
| [`tool_specifications.md`](tool_specifications.md) | Per-tool compatibility scope and conformance status |
| [`../examples/`](../examples/) | Runnable demo scripts |

**Contents**

- [Getting Started](#getting-started):
  [overview](#overview), [installation](#installation),
  [quick start](#quick-start)
- [Signals](#signals)
- [Specification Language](#specification-language)
- [Evaluating Robustness](#evaluating-robustness):
  [API](#robustness-and-evaluate), [output](#interpreting-the-output),
  [static checks](#matching-formulas-to-signals),
  [sub-formula traces](#sub-formula-traces)
- [Backends and Semantics](#backends-and-semantics):
  [choosing](#choosing-a-backend), [native semantics](#what-nativebackend-computes),
  [background](#semantics-background), [torch](#differentiable-robustness-with-torch)
- [Recipes](#recipes)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Getting Started

### Overview

A tidystl workflow has three ingredients:

1. a **signal**: a batch of multi-variable time series with shape
   `(N, S, T)` (N traces, S variables, T timesteps);
2. a **formula**: an STL specification parsed from a string into a `Node`
   tree by `parse()`;
3. a **backend**: the evaluation semantics, selected per call.

`robustness(formula, signal, backend)` returns an `(N, T)` array:
entry `[n, t]` is the robustness of trace `n` for the formula evaluated at
time `times[t]` (that is, over the suffix of the trace starting at
`times[t]`). The rest of this manual walks through each ingredient in
that order, then covers diagnosis and extension.

### Installation

```bash
pip install tidystl
```

Requires Python 3.11+. Core dependencies are numpy and lark only.
For [torch-backed differentiable
evaluation](#differentiable-robustness-with-torch), install the
torch extra:

```bash
pip install "tidystl[torch]"
```

To verify the installation, run the bundled demo from a checkout of the
repository:

```bash
python examples/minimal_demo.py
```

For development installs (running the test suite, regenerating fixtures),
see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

### Quick Start

End to end: build a signal, parse a specification, evaluate.

```python
import numpy as np
from tidystl import Signal, parse, robustness

# One trace of a vehicle speed, sampled 200 times over 10 seconds.
times = np.linspace(0.0, 10.0, 200)
speed = 20.0 + 5.0 * np.sin(times)

sig = Signal.from_dict(times=times, values={"speed": speed})

# "The speed always stays below 30 during the first 10 seconds."
phi = parse("G[0,10](speed <= 30)")

rho = robustness(phi, sig)      # default backend: NativeBackend
print(rho.shape)                # (1, 200): (N traces, T timesteps)
print(float(rho[0, 0]))         # ~5.0: satisfied with margin about 5

assert rho[0, 0] > 0.0          # positive robustness: satisfied
```

The sign of `rho[n, t]` tells you whether trace `n` satisfies the formula
at time `times[t]`; the magnitude is the margin. Here the speed peaks at
25, so the specification holds with robustness 30 - 25 = 5.

## Signals

A `Signal` holds a batch of multi-variable traces sampled on a shared time
grid:

- `values`: float array of shape `(N, S, T)`; N traces, S variables,
  T timesteps;
- `times`: float array of shape `(T,)`, strictly increasing;
- `labels`: dict mapping variable names to indices on the S axis.

The batch axis is first-class: evaluating one formula over thousands of
candidate traces is a single call, which is the common pattern in
falsification and parameter-synthesis loops.

`Signal.from_dict` is the usual entry point. Each variable maps to a
`(N, T)` array; a 1D `(T,)` array is auto-expanded to `(1, T)`:

```python
import numpy as np
from tidystl import Signal

times = np.linspace(0.0, 10.0, 100)
x_batch = np.random.default_rng(0).normal(size=(50, 100))   # (N, T)

sig = Signal.from_dict(times=times, values={"x": x_batch})
assert sig.values.shape == (50, 1, 100)

v_single = np.cos(times)                  # (T,) auto-expands to (1, T)
solo = Signal.from_dict(times=times, values={"v": v_single})
assert solo.values.shape == (1, 1, 100)

sig["x"]        # (N, T) array for one variable
sig["x", 3]     # (N,) values at timestep index 3
```

All variables in one signal must share the batch size and the number of
timesteps. The exact construction and indexing contracts, including error
conditions and the direct `Signal(...)` constructor, are in the
[API reference](reference.md#signals).

Two things to know about time grids:

- `times` is a physical time vector; temporal intervals such as `G[0,10]`
  refer to these units, not to sample indices.
- `NativeBackend` and `BreachBackend` accept any strictly increasing grid
  (values between samples are linearly interpolated). The discrete-time
  backends (`rtamt`, `stlcgpp`, `stlcgpp_torch`) require a uniform grid
  with interval bounds aligned to the sampling period; see
  [Backends and Semantics](#backends-and-semantics).

`TorchSignal` is the torch counterpart: same layout and API, with `values`
held as a torch tensor so [gradients flow through
it](#differentiable-robustness-with-torch).

## Specification Language

Specifications are written inline; predicates are comparisons between
arithmetic expressions over signal variables:

```python
from tidystl import parse

phi = parse("G[0,10]((x >= 0) and F[0,3](velocity - 0.5 * ref >= 0))")
assert phi.kind == "always"
```

Six operators are available: `G[a,b]` (always), `F[a,b]` (eventually),
`U[a,b]` (until), `and`, `or`, `not`. All intervals are bounded, `G` and
`F` require parentheses around their argument, and a predicate's
robustness is its signed margin (`x >= c` evaluates to `x - c`).

The full grammar (operator table, precedence, predicate arithmetic and
robustness rules, reserved keywords) and the legacy sectioned format are
in the [API reference](reference.md#formula-syntax).

## Evaluating Robustness

### robustness() and evaluate()

`robustness(formula, signal, backend=None)` returns the `(N, T)` array
directly (a numpy array for `Signal` input, a torch tensor for
`TorchSignal` input). The `backend` argument takes a registered name such
as `"native"` or `"breach"`, a backend instance, or `None` for the default
(`NativeBackend`); full signatures and argument contracts are in the
[API reference](reference.md#evaluation-api). `evaluate()` takes the
same arguments and returns the full backend result, which adds
sub-formula traces:

```python
import numpy as np
from tidystl import Signal, evaluate, parse

t = np.linspace(0.0, 10.0, 100)
sig = Signal.from_dict(times=t, values={"x": np.sin(t)})

result = evaluate(parse("G[0,10](x >= -1.5)"), sig, backend="breach")
rho = result.robustness            # same array robustness() would return
assert result.has_trace            # sub-formula traces: see below
```

### Interpreting the output

`rho[n, t]` is the robustness of the formula for trace `n` evaluated at
time `times[t]`, that is, over the trace suffix starting there. Most
applications use `rho[:, 0]` (robustness at the start of the trace).

- `rho[n, t] > 0`: satisfied; larger means more margin.
- `rho[n, t] < 0`: violated; smaller means more severe.
- `rho[n, t] == 0`: boundary case; Boolean satisfaction is not determined
  by quantitative robustness alone.

Values at later `t` evaluate temporal windows that extend beyond the
sampled trace; each backend has its own end-of-trace rule there, so
trailing entries should be interpreted with care.

### Matching formulas to signals

Two static analyses help check that a signal is long and dense enough for
a formula before evaluating:

```python
import math
from tidystl import horizon, parse, required_max_gap

phi = parse("G[0,5](F[0,3](x >= 0))")
assert horizon(phi) == 8.0               # 5 + 3: nesting adds up
assert required_max_gap(phi) == 3.0      # min over window upper bounds

pred = parse("x >= 0")
assert horizon(pred) == 0.0
assert math.isinf(required_max_gap(pred))
```

`horizon(phi)` is the future reach: make the trace at least this much
longer than the last evaluation time you care about. `required_max_gap(phi)`
is a conservative sampling bound: it guarantees a sample within each
temporal operator's look-ahead `b`, but not inside an offset window
`[t+a, t+b]` with `a > 0`; to keep every offset window sampled (which also
keeps the dense-time backends in agreement, see
[Backends and Semantics](#backends-and-semantics)), keep the gap at or below the
smallest window width `b - a` instead.

### Sub-formula Traces

`evaluate()` exposes the robustness of every evaluated sub-formula, which
is the primary tool for diagnosing *why* a specification is violated.

`trace_for(node)` is keyed by node object identity: pass the very `Node`
objects reachable from your parsed formula (`phi.children`, attributes) or
the objects returned by `traced_nodes()`.

```python
import numpy as np
from tidystl import Signal, evaluate, parse

times = np.linspace(0.0, 10.0, 100)
sig = Signal.from_dict(
    times=times,
    values={"x": np.full((1, 100), 2.0), "y": np.sin(times)[np.newaxis, :]},
)

# Which conjunct is responsible for the violation?
phi = parse("G[0,10](x >= 0) and F[0,10](y >= 100)")
result = evaluate(phi, sig)

assert result.robustness[0, 0] < 0          # violated overall

left, right = phi.children                  # the two conjuncts
assert result.trace_for(left)[0, 0] > 0     # G[0,10](x >= 0): satisfied
assert result.trace_for(right)[0, 0] < 0    # F[0,10](y >= 100): violated
```

Each trace has the same `(N, T)` shape as the top-level robustness, and
`traced_nodes()` lists every observable node, including predicates and
arithmetic subexpressions. Backends without trace support set
`has_trace = False` and raise `NotImplementedError` from `trace_for` /
`traced_nodes`; all built-in backends support traces. Passing a node that
is not part of the evaluated formula (for example, a node from a second
`parse()` call of the same string) raises `KeyError`.

## Backends and Semantics

### Choosing a backend

Use **`NativeBackend`** (the default) unless you need to match
another tool's numbers. Use **`BreachBackend`** to validate against or
migrate from Breach (MATLAB), **`RtamtBackend`** / **`StlcgppBackend`**
for those tools' discrete-time semantics, and **`StlcgppTorchBackend`**
when you need [gradients](#differentiable-robustness-with-torch).

The dense-time backends (`native`, `breach`) accept any strictly
increasing grid; the discrete-time backends require a uniform grid with
interval bounds aligned to the sampling period. The full comparison table
is in the [API reference](reference.md#backends); all five backends
support sub-formula traces.

### What NativeBackend computes

`NativeBackend` interprets the samples as a piecewise-linear continuous
signal: predicates are evaluated exactly on each linear segment, and
`G[a,b]` / `F[a,b]` compute the exact minimum / maximum of the PL
robustness function over the window `[t+a, t+b]`, including interpolated
values at window boundaries that fall between samples. Until follows the
Maler-Nickovic sliding-window construction [1]. Windows reaching past the
final sample are clamped to the sampled portion of the trace. This is the
principled reference semantics; the other backends reproduce their
reference tools' runtime behavior, including tool-specific deviations.

One example of where this matters: a `G` window that contains no sample
points. NativeBackend takes the true PL minimum over the window;
BreachBackend reproduces Breach's shortcut (the value at `t+a` for the
first sample, the value at `t` afterwards), and the two diverge:

```python
import numpy as np
from tidystl import Signal, parse, robustness

# Sparse samples: t = [0, 2, 4], x = [-1, 10, -1]. The window
# [t+0.5, t+1.5] never contains a sample.
sig = Signal.from_dict(
    times=np.array([0.0, 2.0, 4.0]),
    values={"x": np.array([[-1.0, 10.0, -1.0]])},
)
phi = parse("G[0.5,1.5](x >= 0)")

rho_native = robustness(phi, sig, backend="native")
rho_breach = robustness(phi, sig, backend="breach")

# Native: exact PL minimum, attained at interpolated boundary points.
# Breach: rho(t) at t=2, missing the window minimum entirely.
np.testing.assert_allclose(rho_native[0], [1.75, 1.75, -1.0], atol=1e-10)
np.testing.assert_allclose(rho_breach[0], [1.75, 10.0, -1.0], atol=1e-10)
```

The other documented deviations are of the same flavor but smaller:
BreachBackend copies the penultimate value to the final timestep of
top-level `and`/`or`, and the discrete-time backends differ from each
other in end-of-trace and until-witness details. Each compatibility
backend is regression-tested against ground-truth data generated by its
reference tool; the complete per-tool lists live in
[`tool_specifications.md`](tool_specifications.md), and
[`../examples/falsification_demo.py`](../examples/falsification_demo.py)
shows how the choice affects an optimization loop in practice.

### Semantics background

The base quantitative semantics follows the standard space-robustness
definitions [1, 2]; `NativeBackend`'s exact PL window computation is in
the spirit of Breach's dense-time interpretation [3] computed without
tool-specific shortcuts. Discrete-time backends follow the semantics of
RTAMT [4] and STLCG++ [5] respectively.

### Differentiable Robustness with Torch

`StlcgppTorchBackend` evaluates robustness on torch tensors with full
autograd support, enabling gradient-based falsification and
specification-guided learning. It requires the `torch` extra
([Installation](#installation)) and a `TorchSignal`.

Exact `min`/`max` have piecewise-constant gradients: only the
argmin/argmax sample receives gradient. The backend therefore offers
smooth approximations (`approx_method="softmax"` or `"logsumexp"`, with a
`temperature` knob; constructor options are in the
[API reference](reference.md#stlcgpptorchbackend-options)). Smoothed
robustness is an approximation and its sign near zero is not a reliable
verdict.

```python
import numpy as np
import torch
from tidystl import StlcgppTorchBackend, TorchSignal, parse, robustness

x = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
sig = TorchSignal.from_dict(times=np.arange(3, dtype=float), values={"x": x})
phi = parse("F[0,2](x >= 0)")

# Exact semantics: gradient flows to the argmax sample only.
rho = robustness(phi, sig, backend=StlcgppTorchBackend())
rho.sum().backward()
assert x.grad is not None

# Smooth variant: gradient reaches every sample in the window.
x.grad = None
smooth = StlcgppTorchBackend(approx_method="logsumexp", temperature=10.0)
robustness(phi, sig, backend=smooth).sum().backward()
assert torch.all(x.grad != 0)
```

The discrete-time semantics is shared with `StlcgppBackend`
([Choosing a backend](#choosing-a-backend)): uniform grid, aligned
intervals, last-value extension at end of trace.

## Recipes

### Batch evaluation and falsification ranking

Evaluate a formula over many candidate traces at once and rank them by
robustness; the minimizer is the most violating candidate.

```python
import numpy as np
from tidystl import Signal, parse, robustness

N, T = 1000, 200
rng = np.random.default_rng(7)
velocity = 20.0 + rng.normal(scale=4.0, size=(N, T)).cumsum(axis=1) * 0.1

sig = Signal.from_dict(
    times=np.linspace(0.0, 10.0, T),
    values={"velocity": velocity},
)
phi = parse("G[0,10](velocity <= 30)")

rho0 = robustness(phi, sig)[:, 0]      # (N,) robustness at t=0
worst = velocity[rho0.argmin()]        # most violating candidate
print(f"worst robustness: {rho0.min():.2f}")
```

### Comparing backends on the same signal

When validating a migration, evaluate once per backend and inspect the
difference:

```python
import numpy as np
from tidystl import Signal, parse, robustness

t = np.linspace(0.0, 10.0, 101)        # uniform: works for all backends
sig = Signal.from_dict(times=t, values={"x": np.sin(t)})
phi = parse("G[0.5,1.5](x >= -2)")

rhos = {
    name: robustness(phi, sig, backend=name)
    for name in ["native", "breach", "rtamt", "stlcgpp"]
}
diff = np.abs(rhos["native"] - rhos["breach"]).max()
print(f"max |native - breach| = {diff:.3g}")
```

### Demo scripts and paper benchmarks

```bash
python examples/minimal_demo.py          # smallest end-to-end run
python examples/falsification_demo.py    # semantics choice vs optimization
```

Benchmark scripts (batch speedup, scaling, semantics comparison) live in
`extra/experiments/`; see
[`../extra/experiments/README.md`](../extra/experiments/README.md) for run
instructions. Ground-truth fixtures for the compatibility suites are
regenerated with the scripts under `extra/other_tools/`, documented in
[`../extra/README.md`](../extra/README.md).

## Troubleshooting

**`TypeError: ... backend requires a Signal, got TorchSignal`** (or vice
versa). Numpy backends (`native`, `breach`, `rtamt`, `stlcgpp`) require
`Signal`; `stlcgpp_torch` requires `TorchSignal`. Convert your data to the
matching container.

**`ValueError: rtamt backend requires a strictly increasing uniform time
grid`** (similarly for `stlcgpp`). The discrete-time backends need
equidistant timestamps. Resample your signal, or use `native`/`breach`,
which accept non-uniform grids.

**`ValueError: interval start ... is not aligned to sampling period ...`**
Discrete-time backends require every `[a,b]` bound to be an integer
multiple of the sampling period; adjust the interval or the grid.

**`KeyError` from `trace_for`.** Traces are keyed by node object identity
([Sub-formula Traces](#sub-formula-traces)). Use sub-nodes of the exact formula
object you evaluated, not a re-parsed copy.

**`NotImplementedError` from `trace_for`.** The backend reports
`has_trace = False`; check the flag before requesting traces.

**Native and Breach disagree on my trace.** Expected for top-level binary
`and`/`or` and for `G`/`F` windows containing no samples; see
[What NativeBackend computes](#what-nativebackend-computes) and keep the sampling gap
at or below the smallest window width `b - a`
([Matching formulas to signals](#matching-formulas-to-signals)).

**Robustness at the last timesteps looks off.** Windows extending past the
end of the trace are resolved by backend-specific end-of-trace rules;
prefer `rho[:, 0]` and make sure the trace is at least `horizon(phi)` long
([Matching formulas to signals](#matching-formulas-to-signals)).

## References

1. O. Maler and D. Nickovic. *Monitoring Temporal Properties of Continuous
   Signals.* FORMATS/FTRTFT 2004.
2. G. Fainekos and G. Pappas. *Robustness of Temporal Logic Specifications
   for Continuous-Time Signals.* Theoretical Computer Science, 2009.
3. A. Donze and O. Maler. *Robust Satisfaction of Temporal Logic over
   Real-Valued Signals.* FORMATS 2010; A. Donze. *Breach, A Toolbox for
   Verification and Parameter Synthesis of Hybrid Systems.* CAV 2010.
4. D. Nickovic and T. Yamaguchi. *RTAMT: Online Robustness Monitors from
   STL.* ATVA 2020.
5. P. Kapoor, K. Leung, et al. *STLCG++: A Masking Approach for
   Differentiable Signal Temporal Logic Specification.* 2025.
   [uw-ctrl.github.io/stlcg](https://uw-ctrl.github.io/stlcg/)

To cite tidystl itself, see [`../CITATION.cff`](../CITATION.cff).
