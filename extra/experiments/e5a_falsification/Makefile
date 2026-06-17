export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report smoke
all: run report

run:
	./batch.sh

report:
	uv run --no-sync python aggregate.py --result-dir "$(OUTPUTS_ROOT)/e5a_falsification" \
	  --ecdf "$(OUTPUTS_ROOT)/e5a_falsification/falsification_ecdf.pdf" \
	  --difficulty "$(OUTPUTS_ROOT)/e5a_falsification/difficulty_sweep.pdf"
	uv run --no-sync python agreement.py --n-samples 500 \
	  --output "$(OUTPUTS_ROOT)/e5a_falsification/agreement_table.tex"

smoke:
	OUTPUTS_ROOT=.tmp/smoke \
	  uv run --no-sync \
	  python run.py --backend rtamt --benchmark m2_mass_spring --optimizer anneal --n_repetitions=3 --eval_budget=60

