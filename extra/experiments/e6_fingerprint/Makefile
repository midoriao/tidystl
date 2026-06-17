export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report smoke synth quotient openworld tierb
all: run

run:
	for tool in native breach rtamt stlcgpp taliro pymtl; do \
	  uv run python run.py --tool $$tool; \
	done

report:
	uv run python decode.py

smoke:
	OUTPUTS_ROOT=/tmp/smoke \
	  uv run python run.py --tool breach

# Tier B experiments (synthesis, identifiability quotient, open-world residual).
synth:
	uv run python synth.py

quotient:
	uv run python quotient.py

openworld:
	uv run python openworld.py

tierb: synth quotient openworld
