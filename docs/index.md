# tidySTL

```{toctree}
:hidden:
:maxdepth: 2

usage
language
api
design
```

STL robustness computation for Python: explicit semantics, multiple
backends, batch-first.

- **[Usage](usage.md)** -- user manual: concepts, backend selection, worked examples.
- **[Specification Language](language.md)** -- formula grammar and robustness rules.
- **[API Reference](api.md)** -- generated reference for the public API.
- **[Design](design.md)** -- architecture and design rationale.

## Install

```bash
pip install tidystl
```

## Quick example

```python
import numpy as np
from tidystl import Signal, parse, robustness

sig = Signal.from_dict(
    times=np.linspace(0, 10, 200),
    values={"speed": np.random.default_rng(0).uniform(0, 25, (1, 200))},
)
phi = parse("G[0,10](speed <= 30)")
rho = robustness(phi, sig)
print(rho[0, 0])
```
