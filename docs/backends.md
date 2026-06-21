# Compatibility Backends

The only backend built into `tidystl` is `native` (see
[What NativeBackend computes](usage.md#what-nativebackend-computes)). The
*compatibility* backends each reproduce, in pure numpy, the robustness semantics
of an external STL tool, so results can be cross-checked against that tool
without installing it. They ship in the separate
[`tidystl-compat`](https://github.com/midoriao/tidystl/tree/main/packages/tidystl-compat)
package:

```bash
pip install tidystl-compat
```

Register them once at startup, then select by name:

```python
import tidystl
import tidystl_compat

tidystl.use(tidystl_compat)
print(tidystl.list_backends())   # native, breach, rtamt, ...
```

Each backend module is written to be read as documentation on its own: the
module docstring states the exact semantics it targets, including the
tool-specific deviations it reproduces. The sections below surface those
docstrings (click the source links to read the implementation) and add
cross-references. For a worked side-by-side comparison and a divergence example,
see [Backends and Semantics](usage.md#backends-and-semantics) in the user
manual.

## Breach

Registered name: `breach`. Targets the runtime semantics of
[Breach](https://github.com/decyphir/breach) (MATLAB), reproducing
implementation-specific behaviors that diverge from principled dense-time
semantics. Reach for it to validate against or migrate from an existing Breach
setup; for clean piecewise-linear semantics use `native` instead.

```{eval-rst}
.. automodule:: tidystl_compat.breach
   :no-members:

.. autoclass:: tidystl_compat.breach.BreachBackend
   :members:
```

## RTAMT

Registered name: `rtamt`. Targets the *discrete-time* semantics of
[RTAMT](https://github.com/nickovic/rtamt), which treats the trace as a sequence
of samples on a uniform grid and shifts bounded operators by whole sample
indices. It requires a strictly increasing, uniform time grid with intervals
aligned to the sampling period. For RTAMT's dense-time (piecewise-constant)
interpretation use the `rtamt-dense` backend instead.

```{eval-rst}
.. automodule:: tidystl_compat.rtamt
   :no-members:

.. autoclass:: tidystl_compat.rtamt.RtamtBackend
   :members:
```

## RTAMT (dense-time)

Registered name: `rtamt_dense`. The dense-time counterpart to `rtamt`: it models
the trace as a right-continuous piecewise-constant (sample-and-hold) signal and
resolves bounded operators in real time, so the grid need not be uniform. Use it
to match RTAMT's dense-time interpretation; use `rtamt` for its discrete-time
interpretation.

```{eval-rst}
.. automodule:: tidystl_compat.rtamt_dense
   :no-members:

.. autoclass:: tidystl_compat.rtamt_dense.RtamtDenseBackend
   :members:
```

## py-metric-temporal-logic

Registered name: `pymtl`. Targets the
[`metric-temporal-logic`](https://github.com/mvcisback/py-metric-temporal-logic)
package (`import mtl`): zero-order-hold (piecewise-constant) interpolation over a
possibly non-uniform grid, with right-half-open temporal windows evaluated on a
`dt` pivot grid. The pivot spacing is a backend parameter (`PymtlBackend(dt=...)`,
default `0.1`); construct an instance and pass it as `backend=` to override it.

```{eval-rst}
.. automodule:: tidystl_compat.pymtl
   :no-members:

.. autoclass:: tidystl_compat.pymtl.PymtlBackend
   :members:
```

## TaLiRo

Registered name: `taliro`. Targets TaLiRo's `dp_taliro` robustness in pure
Python. Predicates are linear half-spaces and robustness is the Euclidean signed
distance (normalized by `||A||`), so multi-variable predicates differ from the
raw signed margin used by most other backends. Equality and non-linear
predicates are not half-spaces and raise.

```{eval-rst}
.. automodule:: tidystl_compat.taliro
   :no-members:

.. autoclass:: tidystl_compat.taliro.TaliroBackend
   :members:
```

## STLCG++

Registered names: `stlcgpp` (numpy) and `stlcgpp_torch` (torch, differentiable).
Both target [STLCG++](https://github.com/UW-CTRL/stlcgpp) over a uniform grid,
sharing one discrete-time lowering whose distinguishing feature is the
last-sample end-of-trace extension. The torch backend requires a `TorchSignal`,
produces a robustness tensor that supports `.backward()`, and exposes STLCG++'s
smooth `min`/`max` approximations via
`StlcgppTorchBackend(approx_method=..., temperature=...)` for gradient-based
optimization. Install the torch extra to use it:

```bash
pip install "tidystl-compat[torch]"
```

```{eval-rst}
.. automodule:: tidystl_compat.stlcgpp
   :no-members:

.. autoclass:: tidystl_compat.stlcgpp.StlcgppBackend
   :members:

.. autoclass:: tidystl_compat.stlcgpp.StlcgppTorchBackend
   :members:
```

## Generic (parametrized)

Registered name: `generic`. Not a faithful single-tool backend: it exposes the
implicit-semantics choices that distinguish STL tools as six independent knobs,
so one engine can span the behaviors of the others. Its default configuration
reproduces `native`. Construct `GenericBackend(GenericConfig(...))` and pass it
as `backend=` to select a point in the configuration space.

```{eval-rst}
.. automodule:: tidystl_compat.generic
   :no-members:

.. autoclass:: tidystl_compat.generic.GenericBackend
   :members:

.. autoclass:: tidystl_compat.generic.GenericConfig
   :members:
```
