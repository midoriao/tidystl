# tidySTL

STL robustness evaluation for Python. This repository contains the core
`tidystl` package, optional compatibility and SIMD packages, package documentation,
examples, and experiment/reproduction code.

## Routing

- Installable Python packages live under `packages/`.
- User-facing docs for the core package live under `docs/`.
- Runnable examples live under `examples/`.
- Experiment, reproduction, external-tool handoff, and generated-record
  material lives under `extra/`.

For package usage, start with `packages/tidystl/README.md` or
`docs/usage.md`.

## Layout

```text
packages/
  tidystl/          core tidystl package
  tidystl-compat/   compatibility backends package
  tidystl-simd/     optional Rust/SIMD backend package
examples/           user-facing runnable demos
extra/
  experiments/      benchmark scripts and paper-experiment code
  other_tools/      ground-truth generators and external-tool handoff files
```

## License

MIT License.
Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST). Developed by Sota Sato.
