# tidystl API Reference

Lookup material for the public API: grammar tables, call signatures, and
contracts. For a guided introduction with worked examples, start with the
[user manual](usage.md). 

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

## Evaluation API

### robustness() and evaluate()

```text
robustness(formula, signal, backend=None, *, registry=None) -> (N, T) array
evaluate(formula, signal, backend=None, *, registry=None)   -> EvaluationResult
```

- `formula`: `Node`, the return value of `parse()`;
- `signal`: `Signal` (numpy backends) or `TorchSignal` (torch backends);
- `backend`: a registered name (e.g. `"native"` or `"breach"`; call
  `list_backends()` for the full set), a backend instance, or `None` for the
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
  backends in `tidystl` and `tidystl-compat`);
- `trace_for(node)`: the `(N, T)` robustness trace of one sub-formula.
  Keyed by node object identity: pass `Node` objects reachable from the
  evaluated formula, not a re-parsed copy. Raises `KeyError` for nodes
  outside the evaluated formula and `NotImplementedError` when
  `has_trace` is `False`;
- `traced_nodes()`: every observable node, including predicates and
  arithmetic subexpressions.

### localize()

```text
localize(formula, signal, backend_a, backend_b, *, atol=1e-9, registry=None)
    -> list[DivergentNode]
localize_results(formula, source_a, source_b, *, atol=1e-9) -> list[DivergentNode]
```

Finds where two backends first diverge on the same `formula` and `signal`.
`localize` takes the two backends (a registered name or a backend instance, as
for `robustness()`), evaluates `formula` over `signal` under each, and localizes
the difference — passing one signal keeps both sides on the same trace by
construction. `localize_results` is the lower-level form for when you already
have the two `EvaluationResult`s (or only bare `node -> trace` callables, e.g.
record-based aggregation that does not re-evaluate); the two sources must come
from the same parsed `formula`, so their per-node traces align by node identity.

A post-order walk returns the **minimal divergent nodes** (lowest first): nodes
whose own trace differs by more than `atol` while every descendant still agrees,
so the divergence originates there. Equal infinities count as agreement; an
empty list means the two agree on every node. Predicates are leaves (their
internal arithmetic is not descended into).

Each `DivergentNode` exposes `node` (the AST node), `index` (its post-order
position), `is_root`, and `first_divergent_index` (the flat trace index where
the two first differ; the time step for a single-trace signal). See
[Localizing where two backends diverge](usage.md#localizing-where-two-backends-diverge)
for a worked example.

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

### list_backends()

```text
list_backends(registry=None) -> list[str]
```

Sorted names of the currently registered backends, using the module-level
default registry unless `registry` is given. After
`tidystl.use(tidystl_compat)` the result includes the compat backends
(e.g. `"breach"`); before that it is just `["native"]`.

### use()

```text
use(plugin_or_backends, *, registry=None) -> None
```

Register a backend plugin with the module-level default registry unless
`registry` is given. A plugin module should expose `register(registry)`.
You can also pass one backend instance or an iterable of backend instances.

## Extending tidystl

Backends are resolved through a `BackendRegistry`. A backend is any object
with a `name` string and an `evaluate(formula, signal)` method returning
an `EvaluationResult` (`robustness`, `has_trace`, `trace_for`,
`traced_nodes`; see [EvaluationResult](#evaluationresult)). Importing
`tidystl` populates a module-level default registry with `NativeBackend` (the
one built-in backend); call `tidystl.use(tidystl_compat)` to register the
compat backends (e.g. `breach`). Call `list_backends()` for the full
registered set.
You can also build a private registry and pass it explicitly:

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

Third-party packages
can expose a `register(registry)` function so callers can activate them with
`tidystl.use(plugin_module)`; see [`design.md`](design.md).

For a further example, see the `tidystl_compat` package, which implements a backend for each of the external tools.
