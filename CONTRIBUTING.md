# Contributing

Bug reports, semantics questions, and pull requests are welcome.

## Development

```bash
uv sync --dev
```

CI runs these on every pull request; run them locally first:

```bash
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run pyright src tests examples
uv run pytest
```
