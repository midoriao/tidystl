# tidystl User Manual

tidystl computes quantitative robustness of Signal Temporal Logic (STL)
specifications over batches of time series. It is numpy-native, batch-first,
and puts the evaluation semantics under explicit user control: the default
backend implements principled piecewise-linear (PL) dense-time semantics,
and additional backends reproduce the runtime behavior of existing tools such as Breach for validation and migration.

*Robustness* refers to the quantitative semantics of STL [1, 2]: instead of
a Boolean verdict, each formula evaluates to a real number whose sign
indicates satisfaction (positive) or violation (negative) and whose
magnitude indicates the margin. This manual assumes basic familiarity with
STL; see the [References](#references) for background.

**Document map**

| Document | Content |
|---|---|
| This manual | Installation, concepts, semantics, recipes, troubleshooting |
| [`language.md`](language.md) | Specification language: grammar and robustness rules |
| [`api.md`](api.md) | API reference: signatures and contracts |
| [`design.md`](design.md) | Architecture and design rationale |
| [examples/](https://github.com/midoriao/tidystl/tree/main/examples) | Runnable demo scripts |

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
  [background](#semantics-background), [extending](#extending-with-a-custom-backend)
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

To verify the installation, run the bundled demo from a checkout of the
repository:

```bash
python examples/minimal_demo.py
```

For development installs (running the test suite, regenerating fixtures),
see [CONTRIBUTING.md](https://github.com/midoriao/tidystl/blob/main/CONTRIBUTING.md).

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
{py:class}`API reference <tidystl.Signal>`.

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
robustness rules, reserved keywords) is
on the [Specification Language](language.md) page.

## Evaluating Robustness

### robustness() and evaluate()

`robustness(formula, signal, backend=None)` returns the `(N, T)` array
directly (a numpy array for `Signal` input, a torch tensor for `TorchSignal` input). The `backend` argument takes a registered name such
as `"native"` or `"breach"`, a backend instance, or `None` for the default
(`NativeBackend`); full signatures and argument contracts are in the
{py:func}`API reference <tidystl.evaluate>`. `evaluate()` takes the
same arguments and returns the full backend result, which adds
sub-formula traces:

> **The compat backends (e.g. `breach`) are not built in.** Install the
> separate package (`pip install tidystl-compat`) and call
> `tidystl.use(tidystl_compat)` once at startup to register them with the
> tidystl registry.
> The only built-in backend is `native`; call `list_backends()` to see the full
> set registered in your environment.

```python
import numpy as np
import tidystl
import tidystl_compat
from tidystl import Signal, evaluate, parse

tidystl.use(tidystl_compat)

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
another tool's numbers.
Use backends such as **`BreachBackend`** from `tidystl_compat` to validate against or
migrate from existing tools.

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
import tidystl
import tidystl_compat
from tidystl import Signal, parse, robustness

tidystl.use(tidystl_compat)

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

### Semantics background

The base quantitative semantics follows the standard robustness
definitions [2, 3]; `NativeBackend`'s exact PL window computation is in
the spirit of Breach's dense-time interpretation [3] computed without
tool-specific shortcuts.

### Extending with a custom backend

A backend is any object with a `name` string and an `evaluate(formula, signal)`
method returning an `EvaluationResult`. Register backends in a `BackendRegistry`
and pass it explicitly, or activate a plugin module with
{py:func}`tidystl.use` (see the [design notes](design.md) for
the plugin protocol). The following private registry adds a boolean backend
alongside the native default:

```python
import numpy as np
from tidystl import BackendRegistry, NativeBackend, Signal, parse, robustness


class BooleanBackend:
    """Sign of the native robustness: +1 satisfied, -1 violated."""

    name = "boolean"

    def evaluate(self, formula, signal):
        result = NativeBackend().evaluate(formula, signal)
        result.robustness = np.sign(result.robustness)
        return result


registry = BackendRegistry()
registry.register(NativeBackend(), default=True)
registry.register(BooleanBackend())

sig = Signal.from_dict(
    times=np.arange(3, dtype=float),
    values={"x": np.array([[1.0, -2.0, 3.0]])},
)
verdict = robustness(parse("x >= 0"), sig, backend="boolean", registry=registry)
np.testing.assert_allclose(verdict[0], [1.0, -1.0, 1.0])
```

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
import tidystl
import tidystl_compat
from tidystl import Signal, parse, robustness

tidystl.use(tidystl_compat)

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

### Localizing where two backends diverge

A whole-trace difference says *that* two backends disagree, not *where*. Because
every backend exposes per-node traces over the same formula, `localize` aligns
them node by node and returns the **minimal divergent nodes**: the lowest
sub-formulas whose own output differs while every descendant still agrees, so the
divergence originates there rather than being inherited from below.

`localize` takes one formula, one signal, and the two backends to compare — the
same-formula/same-trace setup the difference is about — and reports where they
part:

```python
import numpy as np
import tidystl_compat  # noqa: F401 -- registers BreachBackend
from tidystl import Signal, localize, parse

t = np.linspace(0.0, 10.0, 101)
sig = Signal.from_dict(times=t, values={"x": np.sin(t)})
phi = parse("G[0.5,1.5](x >= -2)")

for d in localize(phi, sig, "native", "breach"):  # empty if they agree everywhere
    scope = "root" if d.is_root else "sub-formula"
    print(f"diverges at {d.node.kind} ({scope}), first at t-index {d.first_divergent_index}")
```

Each result is a `DivergentNode` carrying the offending `node`, whether it
`is_root`, and the `first_divergent_index` (the time step where the two traces
first part). Several independent origins yield several entries. If you already
hold evaluated results (or only per-node traces), `localize_results(phi,
result_a, result_b)` does the same diff without re-evaluating.

### Demo scripts and paper benchmarks

```bash
python examples/minimal_demo.py          # smallest end-to-end run
python examples/falsification_demo.py    # semantics choice vs optimization
```

Benchmark scripts live in
`extra/experiments/`; see [the experiments README](https://github.com/midoriao/tidystl/blob/main/extra/experiments/README.md) for run instructions. 

## Troubleshooting

**`KeyError` from `trace_for`.** Traces are keyed by node object identity
([Sub-formula Traces](#sub-formula-traces)). Use sub-nodes of the exact formula
object you evaluated, not a re-parsed copy.

**`NotImplementedError` from `trace_for`.** The backend reports
`has_trace = False`; check the flag before requesting traces.

## References

1. O. Maler and D. Nickovic. *Monitoring Temporal Properties of Continuous
   Signals.* FORMATS/FTRTFT 2004.
2. G. Fainekos and G. Pappas. *Robustness of Temporal Logic Specifications
   for Continuous-Time Signals.* Theoretical Computer Science, 2009.
3. A. Donze and O. Maler. *Robust Satisfaction of Temporal Logic over
   Real-Valued Signals.* FORMATS 2010; A. Donze. *Breach, A Toolbox for
   Verification and Parameter Synthesis of Hybrid Systems.* CAV 2010.

To cite tidystl, see [CITATION.cff](https://github.com/midoriao/tidystl/blob/main/CITATION.cff).
