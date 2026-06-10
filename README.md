# tidystl

> STL robustness evaluation for Python. Explicit semantics, batch-first, extensible backends.

```python
import numpy as np
from tidystl import Signal, parse, robustness

phi = parse("""
G[0,5](F[0,2](velocity >= 10))
""")

velocity_batch = np.full((100, 200), 12.0)  # N=100 traces, T=200 samples

sig = Signal.from_dict(
    times=np.linspace(0, 10, 200),
    values={"velocity": velocity_batch},  # shape: (N, T)
)

rho = robustness(phi, sig)  # shape: (N, T)
```

## Features

- **Minimal dependencies** — numpy and lark only; no C++ build, no MATLAB
- **Explicit semantics** — boundary handling, interpolation, and end-of-trace behavior are documented and switchable per backend
- **Cross-validated** — regression suites against Breach, RTAMT, and STLCG++ ground-truth data
- **Batch-first** — native `(N, S, T)` tensor layout
- **Backend-centric** — swap evaluation strategy without changing formula or signal
- **Full STL** — arbitrary nesting including the Until operator

## Use cases

tidySTL is useful when you want to evaluate STL robustness in a small, explicit, and testable way.

- **Evaluate STL robustness**
  Compute robustness values of STL formulas over real-valued signals.

- **Test STL semantics**
  Check boundary-sensitive behavior such as interpolation, sampling, interval endpoints, and temporal aggregation, and compare the results with other STL tools when needed.

- **Experiment with semantics**
  Implement alternative robustness semantics by changing only the relevant evaluation components.

- **Build research prototypes**
  Use tidySTL as a lightweight STL evaluation layer inside monitoring, falsification, learning, or control workflows.

## Installation

```bash
pip install tidystl
```

Requires Python 3.11+, numpy, lark.

For autograd-enabled STLCG++-style execution, install the torch extra
(`pip install "tidystl[torch]"`) and use `TorchSignal` with `StlcgppTorchBackend`.

## Quick Start

```python
import numpy as np
from tidystl import Signal, parse, robustness

phi = parse("G[0,5](F[0,2](x >= 0))")

sig = Signal.from_dict(
    times=np.linspace(0, 10, 100),
    values={"x": np.ones((10, 100))},  # batch of 10 traces, 100 timesteps
)

rho = robustness(phi, sig)  # shape (10, 100)
```

For the full user manual — installation, signals, specification language,
backend semantics, and worked examples — see **[docs/usage.md](docs/usage.md)**;
for grammar tables, call signatures, and API contracts, see
**[docs/reference.md](docs/reference.md)**.

## Repository Layout

```
src/tidystl/        library source
tests/              test suite and ground-truth CSV fixtures
examples/           user-facing runnable demos
docs/
  usage.md          user manual: concepts, semantics, and examples
  reference.md      API reference: grammar, signatures, contracts
  design.md         architecture and cross-tool validation notes
extra/
  experiments/      benchmark scripts and paper-experiment code
  other_tools/      ground-truth generators (Breach/RTAMT/STLCG++)
  tidystl_simd/     optional Rust/SIMD extension (separate sub-project)
```

## License

MIT License.
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST). Developed by Sota Sato.
