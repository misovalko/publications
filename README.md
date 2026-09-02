# Publications assets

Public publication-related assets for the Michal Valko academic website.

This repository is the canonical home for files served under:

`https://misovalko.github.io/publications/`

The main website source lives in `misovalko/misovalko.github.io`. Publication metadata, page rendering, and automation are maintained there; this repository owns the publication files themselves.

## Contents

- canonical paper PDFs
- BibTeX files
- talk and poster PDFs when available
- aggregate publication references such as `MASTER_PUBLICATIONS.md` and `all-publications.bib`

## File conventions

Each paper should have one canonical local PDF using its normal `.pdf` filename. External arXiv, DOI, PMLR, OpenReview, HAL, and similar URLs belong in website metadata rather than as duplicate local PDF aliases.

Related assets may use suffixes such as `.talk.pdf` and `.poster.pdf`.

A few unusually large archival talk decks are intentionally excluded from the GitHub Pages artifact by `_config.yml` and are linked directly from GitHub instead.

## Relationship to the website

The public URL prefix remains `/publications/` even though the files are maintained in this independent repository. The main website intentionally does not contain a `publications/` tree or a Git submodule for this repository.
