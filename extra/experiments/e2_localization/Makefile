export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report smoke
all: run report

run:
	./batch.sh

report:
	uv run python aggregate.py --result-dir "$(OUTPUTS_ROOT)/e2_localization"

smoke:
	OUTPUTS_ROOT=/tmp/smoke \
	  uv run python run.py --backend=native \
	  --spec=div_terminal_and --signal=div_terminal_and
