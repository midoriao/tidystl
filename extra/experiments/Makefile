# Suite-level entry points: delegate to each experiment's Makefile.
#   make run     -- regenerate every experiment's records
#   make report  -- every aggregate + paper emitter
#   make smoke   -- one cheap harness check per experiment
#   make all     -- run + report

EXPERIMENTS := e1_divergence e2_localization e3_variation e4_scaling e5a_falsification e5b_verdict_flip

# Scratch root the smoke recipes write to. Must match the OUTPUTS_ROOT
# pinned in each experiment Makefile's own smoke target (/tmp/smoke).
SMOKE_ROOT := /tmp/smoke

RUN_TARGETS := $(addprefix run-,$(EXPERIMENTS))
REPORT_TARGETS := $(addprefix report-,$(EXPERIMENTS))
SMOKE_TARGETS := $(addprefix smoke-,$(EXPERIMENTS))

# Benchmarks must not run concurrently (timing contamination).
.NOTPARALLEL:

.PHONY: all run report smoke clean-smoke $(RUN_TARGETS) $(REPORT_TARGETS) $(SMOKE_TARGETS)

all: run report

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
