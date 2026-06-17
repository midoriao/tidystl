# Contributing

Bug reports, semantics questions, and pull requests are welcome.

## Development

```bash
uv sync --all-packages --all-groups
```

CI runs these on every pull request; run them locally first:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run --package tidystl pytest packages/tidystl/tests
```
