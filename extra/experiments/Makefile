# Suite-level entry points: delegate to each experiment's Makefile.
#   make run     -- regenerate every experiment's records
#   make report  -- every aggregate + paper emitter
#   make smoke   -- one cheap harness check per experiment
#   make all     -- run + report
#   make test    -- run the experiment test suite (tests/, not gated in CI)

EXPERIMENTS := e1_divergence e2_localization e3_variation e4_scaling e5a_falsification e5b_verdict_flip

# Scratch root the smoke recipes write to. Must match the OUTPUTS_ROOT
# pinned in each experiment Makefile's own smoke target (/tmp/smoke).
SMOKE_ROOT := /tmp/smoke

RUN_TARGETS := $(addprefix run-,$(EXPERIMENTS))
REPORT_TARGETS := $(addprefix report-,$(EXPERIMENTS))
SMOKE_TARGETS := $(addprefix smoke-,$(EXPERIMENTS))

# Benchmarks must not run concurrently (timing contamination).
.NOTPARALLEL:

.PHONY: all run report smoke test clean-smoke \
        reproduce-smoke bootstrap-exp reproduce-full compare \
        $(RUN_TARGETS) $(REPORT_TARGETS) $(SMOKE_TARGETS)

all: test
# all: run report

# Experiment test suite.
# Deliberately NOT run in CI. The experiments are research artifacts validated on demand.
test:
	uv run --project ../.. python -m pytest tests

run: $(RUN_TARGETS)

report: $(REPORT_TARGETS)

smoke: clean-smoke $(SMOKE_TARGETS)

clean-smoke:
	rm -rf "$(SMOKE_ROOT)"

$(RUN_TARGETS): run-%:
	$(MAKE) -C $* run

$(REPORT_TARGETS): report-%:
	$(MAKE) -C $* report

$(SMOKE_TARGETS): smoke-%:
	$(MAKE) -C $* smoke

# ---------------------------------------------------------------------------
# Reproduction
# ---------------------------------------------------------------------------
# One env (.venv-exp, Python 3.12) is shared by the bootstrap, the per-experiment
# Makefiles, and the backend fidelity suite. UV_PROJECT_ENVIRONMENT is kept
# absolute: uv resolves a relative value against the project root (the repo root),
# not this directory.
REPRO_ENV := UV_PROJECT_ENVIRONMENT=$(abspath .venv-exp) UV_PYTHON=3.12

# Core API example: parse a formula and evaluate default-backend robustness on a
# two-trace signal.
define CORE_API_PROBE
import numpy as np
from tidystl import Signal, parse, robustness

phi = parse("G[0,5](F[0,2](velocity >= 10))")
times = np.linspace(0, 10, 200)
velocity_batch = np.array([np.sin(times), np.cos(times)])  # two traces
sig = Signal.from_dict(times=times, values={"velocity": velocity_batch})
rho = robustness(phi, sig)  # default (native) backend
assert rho.shape[0] == 2, rho.shape
print("Core API probe OK: robustness shape", rho.shape)
endef
export CORE_API_PROBE

check:
	@echo "==== light smoke: core test suite ===="
	cd ../.. && uv run --project . python -m pytest packages/tidystl/tests
	@echo "==== light smoke: Core API probe (default backend, numpy+lark only) ===="
	cd ../.. && uv run --project . python -c "$$CORE_API_PROBE"

# Provision .venv-exp at Python 3.12, matching the per-experiment Makefiles.
bootstrap-exp:
	cd ../.. && $(REPRO_ENV) uv sync --group experiments

reproduce-full: bootstrap-exp
	@echo "==== [1/6] e1: records + tab:divergence-profile ===="
	$(MAKE) -C e1_divergence run profile
	@echo "==== [2/6] e3: records + tab:variation ===="
	$(MAKE) -C e3_variation all
	@echo "==== [3/6] e2: records + tab:e2-localization ===="
	$(MAKE) -C e2_localization all
	@echo "==== [4/6] e5b: clean N=201 sweep + verdict-flip rates ===="
	$(MAKE) -C e5b_verdict_flip all
	@echo "==== [5/6] e4: records + tab:scaling ===="
	$(MAKE) -C e4_scaling all
	@echo "==== [6/6] backend fidelity suite ===="
	$(REPRO_ENV) UV_NO_SYNC=true uv run --only-group experiments \
	  python -m pytest ../../packages/tidystl-compat/tests

compare:
	@echo "==== compare against the committed references ===="
	@echo "  - tab:variation (e3):  LOC cells of extra/outputs/e3_variation/variation.tex"
	@echo "                         vs paper/figures/variation-table.tex"
	@echo "  - tab:e2-localization: diff extra/outputs/e2_localization/localization_summary_table.tex"
	@echo "                         vs extra/experiments/e2_localization/figs/localization_summary_table.tex"
	@echo "  - tab:scaling (e4):    columns/markers/ordering of extra/outputs/e4_scaling/scaling.tex"
	@echo "                         vs paper/figures/scaling-table.tex (absolute timings are host-dependent)"
	@echo "  - tab:divergence-profile (e1): values in extra/outputs/e1_divergence/profile.json"
	@echo "  - e5b rates: flips=/inversions= at n=201 from the step [4] output (101/201, 100/201, 80/201)"
