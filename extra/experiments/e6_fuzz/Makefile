export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report smoke test
all: run report

run:
	./batch.sh

report:
	uv run python aggregate.py --result-dir "$(OUTPUTS_ROOT)/e6_fuzz" --emit-tex "$(OUTPUTS_ROOT)/e6_fuzz/divergence_matrix.tex"

smoke:
	OUTPUTS_ROOT=/tmp/e6_smoke uv run python run.py --n-pairs 12 --seed 0

test:
	uv run python -m pytest tests/ -q
