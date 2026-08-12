SHELL := /bin/sh

PACKAGE_NAME := tidystl-rv2026-source
DIST_DIR := dist
STAGE_ROOT := .build/submission
STAGE_DIR := $(STAGE_ROOT)/$(PACKAGE_NAME)
SOURCE_ZIP := $(DIST_DIR)/$(PACKAGE_NAME).zip
DIST_PDF := $(DIST_DIR)/tidystl-rv2026.pdf

PAPER_SOURCES := \
	main.tex \
	latexmkrc \
	bib/references.bib \
	figures/architecture.pdf \
	$(wildcard figures/*.tex) \
	$(wildcard sections/*.tex)

SUBMISSION_FILES := Makefile README.md $(PAPER_SOURCES)

.PHONY: all pdf extended dist-pdf source-zip submission-check clean

all: pdf

pdf:
	latexmk -pdf main.tex

extended:
	latexmk -pdf extended.tex

dist-pdf: $(DIST_PDF)

$(DIST_PDF): $(PAPER_SOURCES)
	@mkdir -p "$(DIST_DIR)"
	latexmk -pdf main.tex
	cp main.pdf "$@"

source-zip: $(DIST_PDF) $(SOURCE_ZIP)

$(SOURCE_ZIP): $(SUBMISSION_FILES)
	@set -eu; \
	rm -rf "$(STAGE_DIR)"; \
	mkdir -p "$(STAGE_DIR)" "$(DIST_DIR)"; \
	for file in $(SUBMISSION_FILES); do \
		mkdir -p "$(STAGE_DIR)/$$(dirname "$$file")"; \
		cp "$$file" "$(STAGE_DIR)/$$file"; \
	done; \
	for dependency in llncs.cls splncs04.bst; do \
		path=$$(kpsewhich "$$dependency"); \
		test -n "$$path" || { echo "Could not find $$dependency with kpsewhich" >&2; exit 1; }; \
		cp "$$path" "$(STAGE_DIR)/$$dependency"; \
	done; \
	rm -f "$(SOURCE_ZIP)"; \
	cd "$(STAGE_DIR)"; \
	zip -X -q -r "$(abspath $(SOURCE_ZIP))" .; \
	echo "Created $(SOURCE_ZIP)"

# Compile from a clean extraction, so this check cannot use files left by a
# build in the working tree.
submission-check: $(DIST_PDF) $(SOURCE_ZIP)
	@set -eu; \
	tmp=$$(mktemp -d "$${TMPDIR:-/tmp}/tidystl-submission.XXXXXX"); \
	trap 'rm -rf "$$tmp"' EXIT HUP INT TERM; \
	unzip -q "$(abspath $(SOURCE_ZIP))" -d "$$tmp"; \
	test -f "$$tmp/main.tex"; \
	$(MAKE) -C "$$tmp" pdf; \
	test -s "$$tmp/main.pdf"; \
	test -s "$(abspath $(DIST_PDF))"; \
	echo "Verified: $(SOURCE_ZIP) builds main.pdf from a clean extraction"

clean:
	latexmk -c main.tex extended.tex
