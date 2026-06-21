# API Reference

The public API re-exported from the top-level `tidystl` package. For the
formula grammar see [Specification Language](language.md); for guided
examples see the [user manual](usage.md).

## Parsing

Turn a formula string into an evaluable AST and inspect its temporal extent.

```{eval-rst}
.. autofunction:: tidystl.parse
.. autofunction:: tidystl.horizon
.. autofunction:: tidystl.required_max_gap
```

## Signals

Containers for the time-series data that formulas are evaluated against.

```{eval-rst}
.. autoclass:: tidystl.Signal
   :members:
.. autoclass:: tidystl.TorchSignal
   :members:
```

## Evaluation

Compute robustness values and structured evaluation results for a formula.

```{eval-rst}
.. autofunction:: tidystl.robustness
.. autofunction:: tidystl.evaluate
.. autoclass:: tidystl.EvaluationResult
   :members:
```

## Localization

Diagnose which subformula and time interval drive a (dis)satisfaction.

```{eval-rst}
.. autofunction:: tidystl.localize
.. autofunction:: tidystl.localize_results
.. autoclass:: tidystl.DivergentNode
   :members:
```

## Backends

Select and register the engines that perform robustness computation.

```{eval-rst}
.. autofunction:: tidystl.list_backends
.. autofunction:: tidystl.use
.. autoclass:: tidystl.BackendRegistry
   :members:
.. autofunction:: tidystl.get_default_backend_registry
.. autoclass:: tidystl.NativeBackend
   :members:
```

## AST

The formula syntax tree produced by {py:func}`tidystl.parse`.

```{eval-rst}
.. autoclass:: tidystl.Node
   :members:
```
