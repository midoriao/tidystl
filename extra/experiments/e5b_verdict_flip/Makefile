export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report smoke
all: run report

run:
	# Clean first so a re-run does not pool records from a different sweep size
	# (the rates are flips/N, so a mixed N would corrupt the denominator). N=201
	# is the paper resolution (step 0.05 over [-5,5]).
	rm -rf "$(OUTPUTS_ROOT)/e5b_verdict_flip"
	./batch.sh 201

report:
	uv run python aggregate.py --reference-values "5.0,-1.0,5.0" --result-dir "$(OUTPUTS_ROOT)/e5b_verdict_flip"

smoke:
	OUTPUTS_ROOT=/tmp/smoke \
	  uv run python run.py --backend=native --signal.s=1.0
