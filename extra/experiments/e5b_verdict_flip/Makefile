export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report smoke
all: run report

run:
	./batch.sh

report:
	uv run python aggregate.py --reference-values "5.0,-1.0,5.0" --result-dir "$(OUTPUTS_ROOT)/e5b_verdict_flip"

smoke:
	OUTPUTS_ROOT=/tmp/smoke \
	  uv run python run.py --backend=native --signal.s=1.0
