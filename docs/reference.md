# tidystl API Reference

Lookup material for the public API: grammar tables, call signatures, and
contracts. For a guided introduction with worked examples, start with the
[user manual](usage.md). Python blocks in this document are executable and
CI-tested, like the manual's (`tests/test_usage_doc.py`).

**Contents**

- [Formula Syntax](#formula-syntax)
- [Signals](#signals)
- [Evaluation API](#evaluation-api)
- [Backends](#backends)
- [Extending tidystl](#extending-tidystl)

## Formula Syntax

### Operators

| Operator | Syntax | Node kind |
|---|---|---|
| Always (globally) | `G[a,b](phi)` | `always` |
| Eventually (finally) | `F[a,b](phi)` | `eventually` |
| Until | `phi U[a,b] psi` | `until` |
| Conjunction | `phi and psi` | `and` |
| Disjunction | `phi or psi` | `or` |
| Negation | `not phi` | `not` |

All temporal intervals `[a,b]` are bounded with `0 <= a <= b`; unbounded
operators are not supported. `G` and `F` require parentheses around their
argument: `G[0,5](x >= 0)`.

Binding strength, from loosest to tightest:

```text
or  <  and  <  U  <  not, G, F  <  atoms
```

So `not p and q` parses as `(not p) and q`, and `p and q or r` parses as
`(p and q) or r`. Operands of `U` are unary-level expressions; use
parentheses to put `and`/`or` under an until.

### Predicate arithmetic

Predicate operands are arithmetic expressions over variables and numeric
constants:

- binary: `+`, `-`, `*`, `/`, exponentiation `^`;
- unary: `-x` (negation), `abs(...)`, `sqrt(...)`;
- comparisons: `>=`, `>`, `<=`, `<`, `==`.

Robustness of an atomic predicate is its signed margin:

| Predicate | Robustness |
|---|---|
| `lhs >= rhs`, `lhs > rhs` | `lhs - rhs` |
| `lhs <= rhs`, `lhs < rhs` | `rhs - lhs` |
| `lhs == rhs` | `-abs(lhs - rhs)` |

Note two consequences of quantitative semantics: strict and non-strict
comparisons have identical robustness, and `==` never yields positive
robustness (0 at exact equality, negative otherwise).

The keywords `and`, `or`, `not`, `abs`, `sqrt` are reserved and cannot be
used as variable names.

### Legacy sectioned format and errors

The older two-section format with named predicates is still accepted; a
string containing a `[predicates]` or `[stl]` header is parsed as
sectioned, everything else as inline. New code should prefer inline.

```python
from tidystl import parse

phi = parse("""
[predicates]
safe : x >= 0

[stl]
G[0,10](safe)
""")
assert phi.kind == "always"
```

Syntax errors raise lark exceptions (subclasses of
`lark.exceptions.UnexpectedInput`); referencing an undefined predicate
name in the sectioned format raises `ValueError`. Bare names are not
valid inline formulas; predicates must be comparisons.

## Signals

### Signal

A `Signal` holds a batch of multi-variable traces on a shared time grid:

- `values`: float array of shape `(N, S, T)`; N traces, S variables,
  T timesteps;
- `times`: float array of shape `(T,)`, strictly increasing, in physical
  time units (temporal intervals such as `G[0,10]` refer to these units,
  not to sample indices);
- `labels`: dict mapping variable names to indices on the S axis.

Construction:

- `Signal(values=..., times=..., labels=...)`: direct, for an already
  stacked `(N, S, T)` array.
- `Signal.from_dict(times=..., values={name: array})`: each variable maps
  to an `(N, T)` array; a 1D `(T,)` array is auto-expanded to `(1, T)`.
  All variables must share the batch size and the number of timesteps.

Indexing: `sig["x"]` returns the `(N, T)` array for one variable;
`sig["x", 3]` returns the `(N,)` values at timestep index 3.

Errors: a shape mismatch (between variables, or against `times`) raises
`ValueError` at construction; an unknown variable name raises `KeyError`.

### TorchSignal

The torch counterpart of `Signal`: same layout, constructors, and indexing,
with `values` held as a torch tensor so gradients flow through it.
Requires the `torch` extra. Accepted by `stlcgpp_torch` only; numpy
backends raise `TypeError` for `TorchSignal` input and vice versa.

## Evaluation API

### robustness() and evaluate()

```text
robustness(formula, signal, backend=None, *, registry=None) -> (N, T) array
evaluate(formula, signal, backend=None, *, registry=None)   -> EvaluationResult
```

- `formula`: `Node`, the return value of `parse()`;
- `signal`: `Signal` (numpy backends) or `TorchSignal` (torch backends);
- `backend`: a registered name (`"native"`, `"breach"`, `"rtamt"`,
  `"stlcgpp"`, `"stlcgpp_torch"`), a backend instance, or `None` for the
  default (`NativeBackend`);
- `registry`: a custom `BackendRegistry`
  (see [Extending tidystl](#extending-tidystl)).

`robustness()` returns the `(N, T)` array directly (a numpy array for
`Signal` input, a torch tensor for `TorchSignal` input); entry `[n, t]` is
the robustness of trace `n` for the formula evaluated at time `times[t]`.
`evaluate()` returns the full backend result ([EvaluationResult](#evaluationresult)).

### EvaluationResult

- `robustness`: the `(N, T)` array `robustness()` would return;
- `has_trace`: whether sub-formula traces are available (`True` for all
  built-in backends);
- `trace_for(node)`: the `(N, T)` robustness trace of one sub-formula.
  Keyed by node object identity: pass `Node` objects reachable from the
  evaluated formula, not a re-parsed copy. Raises `KeyError` for nodes
  outside the evaluated formula and `NotImplementedError` when
  `has_trace` is `False`;
- `traced_nodes()`: every observable node, including predicates and
  arithmetic subexpressions.

### Static analyses

```text
horizon(formula) -> float
required_max_gap(formula) -> float
```

`horizon` is the future reach of the formula (nested window bounds add
up); `required_max_gap` is a conservative sampling bound (the minimum over
temporal operators' look-ahead upper bounds; infinite for predicates).
Usage guidance and the offset-window caveat are in the
[user manual](usage.md#matching-formulas-to-signals).

## Backends

### Comparison

| Backend | Name | Time model | Signal type | Grid requirement | Gradients |
|---|---|---|---|---|---|
| `NativeBackend` (default) | `"native"` | dense (PL) | `Signal` | strictly increasing | no |
| `BreachBackend` | `"breach"` | dense (PL) | `Signal` | strictly increasing | no |
| `RtamtBackend` | `"rtamt"` | discrete | `Signal` | uniform, aligned intervals | no |
| `StlcgppBackend` | `"stlcgpp"` | discrete | `Signal` | uniform, aligned intervals | no |
| `StlcgppTorchBackend` | `"stlcgpp_torch"` | discrete | `TorchSignal` | uniform, aligned intervals | yes |

All five backends support sub-formula traces (`has_trace = True`).

"Uniform, aligned intervals" means equidistant timestamps with every
interval bound `[a,b]` an integer multiple of the sampling period;
violations raise `ValueError`. Semantics guidance and worked divergence
examples are in the [user manual](usage.md#backends-and-semantics);
the per-tool conformance scope is in
[`tool_specifications.md`](tool_specifications.md).

### StlcgppTorchBackend options

```text
StlcgppTorchBackend(*, approx_method="true", temperature=1.0)
```

- `approx_method="true"` (default): exact `min`/`max`;
- `approx_method="softmax"`: softmax-weighted average,
  `sum_i softmax(k * r)_i * r_i` with temperature `k`;
- `approx_method="logsumexp"`: `logsumexp(k * r) / k`.

Higher `temperature` means closer to the exact value; smoothed robustness
is an approximation and its sign near zero is not a reliable verdict.

## Extending tidystl

Backends are resolved through a `BackendRegistry`. A backend is any object
with a `name` string and an `evaluate(formula, signal)` method returning
an `EvaluationResult` (`robustness`, `has_trace`, `trace_for`,
`traced_nodes`; see [EvaluationResult](#evaluationresult)). Importing
`tidystl` populates a module-level default registry with the five built-in
backends; you can also build a private registry and pass it explicitly:

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

(This sketch reuses the native result object; a standalone backend would
implement the `EvaluationResult` protocol itself.) Third-party packages
can instead call `tidystl.get_default_backend_registry().register(...)`
at import time to make their backend available by name in every call; see
[`design.md`](design.md).
