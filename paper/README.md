# RV 2026 paper draft

- `main.tex`: LNCS main file
- `sections/`: section drafts
- `figures/`: paper figures
- `bib/references.bib`: bibliography

## Build

Run `make pdf` to build the proceedings paper, or `make extended` to build the
version with the supplementary appendix.

Run `make source-zip` to create both `dist/tidystl-rv2026.pdf` and
`dist/tidystl-rv2026-source.zip`. The archive contains the paper sources and the
LNCS class and bibliography style found in the local TeX installation. Run
`make submission-check` to create both submission files, extract the archive
into a fresh temporary directory, and verify that it builds `main.pdf` there.
