export UV_PROJECT_ENVIRONMENT := $(abspath ../.venv-exp)
export UV_PYTHON := 3.12
export UV_NO_SYNC := true
export OUTPUTS_ROOT := ../../outputs

.PHONY: all run report profile smoke
all: run report

run:
	./batch.sh

report:
	uv run python aggregate.py --result-dir "$(OUTPUTS_ROOT)/e1_divergence" --emit-tex "$(OUTPUTS_ROOT)/e1_divergence/divergence.tex"

# Emit the paper's tab:divergence-profile. Reads the columns produced by `run`,
# so run that first (`make run profile`). Separate from the supplementary
# divergence-matrix that `report` emits.
profile:
	uv run python profile.py

smoke:
	uv sync --frozen --group experiments
	@echo "==== SMOKE TEST[E1/tidystl] ===="
	uv run --only-group experiments \
	  python ../tools/eval_tidystl.py \
	  --backend native \
	  --specs ../registry/specs_div.json \
	  --signals ../registry/signals_div.json \
	  --output /tmp/smoke/e1_divergence/native_div.json

	@echo "==== SMOKE TEST[E1/simd] ===="
	uv run --only-group experiments \
	  python ../tools/eval_tidystl.py \
	  --backend tidystl_simd \
	  --specs ../registry/specs_div.json \
	  --signals ../registry/signals_div.json \
	  --output /tmp/smoke/e1_divergence/simd_div.json

	@echo "==== SMOKE TEST[E1/stlcgpp] ===="
	uv run --only-group experiments \
	  python ../tools/eval_external_tools.py \
	  --tool stlcgpp \
	  --specs ../registry/specs_div.json \
	  --signals ../registry/signals_div.json \
	  --output /tmp/smoke/e1_divergence/stlcgpp_div.json

	@echo "==== SMOKE TEST[E1/rtamt] ===="
	uv run --only-group experiments \
	  python ../tools/eval_external_tools.py \
	  --tool rtamt \
	  --specs ../registry/specs_div.json \
	  --signals ../registry/signals_div.json \
	  --output /tmp/smoke/e1_divergence/rtamt_div.json
