# Makefile — the single runnable reference for this repo's pipeline order.
#
# Two entry points cover almost all use:
#
#   make check       # the CI gate: lint + format + types + tests (+coverage)
#   make reproduce   # rebuild the paper's tables/figures/PDF from results/data/
#
# The full re-measurement pipeline (only needed to regenerate results/data/ from
# scratch — requires the raw corpora and trained tokenizers, which are deposited
# to Zenodo, not committed) runs in this order:
#
#   prep  ->  train  ->  audit  ->  measure  ->  generate  ->  paper
#
# Corpus prep is per-corpus and config-driven. For each headline corpus C in
# {pubchem, zinc22, coconut, real_space}:
#
#   make ingest CORPUS=C                                   # raw acquisition
#   # (zinc22 only) make tranche-union CONFIG=configs/preprocess/zinc22_tranche_union.yaml
#   make canon-dedup  CONFIG=configs/preprocess/C_canon_dedup.yaml      # -> canon_dedup_v1_full
#   make conformance  IN=data/processed/C/canon_dedup_v1_full OUT=data/processed/C/conformant_v1_full
#   # (pubchem, zinc22 only) make subsample CONFIG=configs/preprocess/C_hash_subsample.yaml  # conformant_v1_full -> conformant_v1_sub
#   make holdout      CONFIG=configs/preprocess/C_holdout_split.yaml    # -> canon_dedup_v1/{train,test}
#
# Conformance (drop non-OpenSMILES, closing the corpus under the 158-glyph base)
# sits after canon-dedup and before subsample, so train AND the in-distribution
# test split are both carved from the conformant set.
#
# OOD eval corpora are eval-only (never trained) and skip the conformance drop:
#   make ingest CORPUS=cycpeptmpdb && make canon-dedup CONFIG=configs/preprocess/cycpeptmpdb_canon_dedup.yaml
#   make ingest CORPUS=tmqm && make derive-tmqm   # dative -> OpenSMILES conversion, not a drop
#
# Run `make help` for the full target list.

PY     := uv run python
PYTEST := uv run pytest
SCRIPTS := scripts
# results/ is the standalone generation surface: results/data holds the
# measurement deposits, results/build renders results/{tables,figures}.
RESULTS_BUILD := results/build

.DEFAULT_GOAL := help

# --------------------------------------------------------------------------- #
# Help                                                                         #
# --------------------------------------------------------------------------- #

.PHONY: help
help: ## Show this help
	@echo "Targets (see the header of this Makefile for the full pipeline order):"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Setup & developer gate                                                       #
# --------------------------------------------------------------------------- #

.PHONY: setup
setup: ## Install deps (smirk builds from source — needs a Rust toolchain)
	uv sync --extra figures --extra crosstoolkit --dev

.PHONY: lint
lint: ## Lint (ruff check)
	uv run ruff check .

.PHONY: lint-fix
lint-fix: ## Lint and apply safe autofixes
	uv run ruff check --fix .

.PHONY: format
format: ## Format the tree (ruff format, writes)
	uv run ruff format .

.PHONY: format-check
format-check: ## Check formatting without writing
	uv run ruff format --check .

.PHONY: typecheck
typecheck: ## Type check (pyright over src/smiles_subword + scripts)
	uv run pyright

.PHONY: test
test: ## Run the test suite minus GPU tests
	$(PYTEST) -m "not gpu"

.PHONY: coverage
coverage: ## Run tests with the CI coverage gate (--cov-fail-under=85)
	$(PYTEST) -m "not gpu" --cov=smiles_subword --cov-report=term-missing --cov-fail-under=85

.PHONY: check
check: lint format-check typecheck coverage ## Full CI gate: lint + format + types + tests + coverage

# --------------------------------------------------------------------------- #
# Generation surface (results/) — render committed deposits into tables/figures #
# --------------------------------------------------------------------------- #
# `generate` rebuilds everything derivable from results/data alone (no tokenizer
# bundle); `generate-full` adds the piece-level artifacts that re-read the
# trained tokenizers. The manuscript (paper/) is a downstream consumer.

.PHONY: tables
tables: ## Deposit-derived tables -> results/tables (from results/data)
	$(PY) $(RESULTS_BUILD)/build_results_tables.py
	$(PY) $(RESULTS_BUILD)/table_ood_eval.py
	$(PY) $(RESULTS_BUILD)/table_transfer.py

.PHONY: figures
figures: ## Deposit-derived figures -> results/figures (needs the figures extra)
	$(PY) $(RESULTS_BUILD)/figure_cross_v_trends.py
	$(PY) $(RESULTS_BUILD)/figure_algo_boundary_interaction.py
	$(PY) $(RESULTS_BUILD)/figure_scale_anchor.py
	$(PY) $(RESULTS_BUILD)/figure_membership_upset.py
	$(PY) $(RESULTS_BUILD)/figure_fertility_curves.py
	$(PY) $(RESULTS_BUILD)/figure_distribution_intrinsics.py
	$(PY) $(RESULTS_BUILD)/figure_noncanon.py
	$(PY) $(RESULTS_BUILD)/figure_sensitivity_curves.py
	$(PY) $(RESULTS_BUILD)/figure_interaction_surfaces.py
	$(PY) $(RESULTS_BUILD)/figure_overlap_strip.py

.PHONY: tables-artifacts
tables-artifacts: ## Tokenizer-artifact-derived tables (needs the trained-tokenizer bundle)
	$(PY) $(RESULTS_BUILD)/table_base_glyphs.py
	$(PY) $(RESULTS_BUILD)/table_composition.py
	$(PY) $(RESULTS_BUILD)/table_multiglyph_split.py
	$(PY) $(RESULTS_BUILD)/table_shared_core_growth.py
	$(PY) $(RESULTS_BUILD)/table_arm_exclusive.py
	$(PY) $(RESULTS_BUILD)/table_coconut_contrast.py
	$(PY) $(RESULTS_BUILD)/table_narrow_contrast.py

.PHONY: figures-artifacts
figures-artifacts: ## Tokenizer-artifact-derived figures (needs the bundle)
	$(PY) $(RESULTS_BUILD)/figure_graphical_abstract.py
	$(PY) $(RESULTS_BUILD)/figure_glyph_cooccurrence.py
	$(PY) $(RESULTS_BUILD)/figure_segmentation_nesting.py
	$(PY) $(RESULTS_BUILD)/figure_piece_length.py

.PHONY: generate
generate: tables figures ## Render the deposit-derived surface (results/) from results/data

.PHONY: generate-full
generate-full: generate tables-artifacts figures-artifacts ## ...also the artifact-derived (needs bundle)

.PHONY: headline
headline: ## Re-measure the headline Jaccard deposit (needs tokenizers; not a render step)
	$(PY) $(SCRIPTS)/measure/compute_jaccard.py

# --------------------------------------------------------------------------- #
# Manuscript (paper/) — a consumer that copies results/ in and typesets        #
# --------------------------------------------------------------------------- #

.PHONY: paper
paper: ## Sync results/{tables,figures} into paper/ and compile paper.tex -> PDF
	rm -rf paper/tables paper/figures
	cp -R results/tables paper/tables
	cp -R results/figures paper/figures
	cd paper && latexmk -pdf paper.tex

.PHONY: reproduce
reproduce: generate paper ## Rebuild the deposit-derived paper from results/data (repo-only)

.PHONY: reproduce-full
reproduce-full: generate-full paper ## ...also the artifact-derived tables/figures (needs the bundle)

# --------------------------------------------------------------------------- #
# Full re-measurement pipeline (heavy; needs raw corpora + trained artifacts)  #
# --------------------------------------------------------------------------- #

CORPUS ?=
CONFIG ?=
IN     ?=
OUT    ?=

.PHONY: ingest
ingest: ## Acquire one raw corpus: make ingest CORPUS=pubchem
	@test -n "$(CORPUS)" || { echo "set CORPUS=<pubchem|zinc22|coconut|real_space|tmqm|cycpeptmpdb>"; exit 2; }
	$(PY) $(SCRIPTS)/ingest/ingest_$(CORPUS).py --config configs/corpus/$(CORPUS).yaml

.PHONY: tranche-union
tranche-union: ## (zinc22) Consolidate enumerated tranches: make tranche-union CONFIG=...
	@test -n "$(CONFIG)" || { echo "set CONFIG=configs/preprocess/zinc22_tranche_union.yaml"; exit 2; }
	$(PY) $(SCRIPTS)/preprocess/consolidate_tranches.py --config $(CONFIG)

.PHONY: canon-dedup
canon-dedup: ## RDKit-canonicalize + dedup a corpus: make canon-dedup CONFIG=...
	@test -n "$(CONFIG)" || { echo "set CONFIG=configs/preprocess/<corpus>_canon_dedup.yaml"; exit 2; }
	$(PY) $(SCRIPTS)/preprocess/canon_dedup.py --config $(CONFIG)

.PHONY: derive-tmqm
derive-tmqm: ## (tmQM OOD) Convert dative-bond SMILES to OpenSMILES -> opensmiles_v1
	$(PY) $(SCRIPTS)/preprocess/derive_tmqm_opensmiles.py

.PHONY: conformance
conformance: ## OpenSMILES-conformance filter: make conformance IN=<dir> OUT=<dir>
	@test -n "$(IN)" && test -n "$(OUT)" || { echo "set IN=<input-dir> OUT=<output-dir>"; exit 2; }
	$(PY) $(SCRIPTS)/preprocess/filter_conformance.py --input-dir $(IN) --output-dir $(OUT)

.PHONY: subsample
subsample: ## (pubchem, zinc22) Deterministic hash-subsample: make subsample CONFIG=...
	@test -n "$(CONFIG)" || { echo "set CONFIG=configs/preprocess/<corpus>_hash_subsample.yaml"; exit 2; }
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(CONFIG)

.PHONY: holdout
holdout: ## Carve the held-out test split: make holdout CONFIG=...
	@test -n "$(CONFIG)" || { echo "set CONFIG=configs/preprocess/<corpus>_holdout_split.yaml"; exit 2; }
	$(PY) $(SCRIPTS)/preprocess/holdout_split.py --config $(CONFIG)

# --------------------------------------------------------------------------- #
# Per-corpus prep convenience targets (the whole chain in one command)        #
# --------------------------------------------------------------------------- #

PROC := data/processed
PRE  := configs/preprocess
COR  := configs/corpus

.PHONY: prep-pubchem
prep-pubchem: ## PubChem grid prep: ingest -> canon -> conformance -> subsample -> holdout
	$(PY) $(SCRIPTS)/ingest/ingest_pubchem.py --config $(COR)/pubchem.yaml
	$(PY) $(SCRIPTS)/preprocess/canon_dedup.py --config $(PRE)/pubchem_canon_dedup.yaml
	$(PY) $(SCRIPTS)/preprocess/filter_conformance.py --input-dir $(PROC)/pubchem/canon_dedup_v1_full --output-dir $(PROC)/pubchem/conformant_v1_full
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_hash_subsample.yaml
	$(PY) $(SCRIPTS)/preprocess/holdout_split.py --config $(PRE)/pubchem_holdout_split.yaml

.PHONY: prep-zinc22
prep-zinc22: ## ZINC-22 grid prep: ingest -> tranche-union -> canon -> conformance -> subsample -> holdout
	$(PY) $(SCRIPTS)/ingest/ingest_zinc22.py --config $(COR)/zinc22.yaml
	$(PY) $(SCRIPTS)/preprocess/consolidate_tranches.py --config $(PRE)/zinc22_tranche_union.yaml
	$(PY) $(SCRIPTS)/preprocess/canon_dedup.py --config $(PRE)/zinc22_canon_dedup.yaml
	$(PY) $(SCRIPTS)/preprocess/filter_conformance.py --input-dir $(PROC)/zinc22/canon_dedup_v1_full --output-dir $(PROC)/zinc22/conformant_v1_full
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/zinc22_hash_subsample.yaml
	$(PY) $(SCRIPTS)/preprocess/holdout_split.py --config $(PRE)/zinc22_holdout_split.yaml

.PHONY: prep-coconut
prep-coconut: ## COCONUT grid prep: ingest -> canon -> conformance -> holdout (no subsample)
	$(PY) $(SCRIPTS)/ingest/ingest_coconut.py --config $(COR)/coconut.yaml
	$(PY) $(SCRIPTS)/preprocess/canon_dedup.py --config $(PRE)/coconut_canon_dedup.yaml
	$(PY) $(SCRIPTS)/preprocess/filter_conformance.py --input-dir $(PROC)/coconut/canon_dedup_v1_full --output-dir $(PROC)/coconut/conformant_v1_full
	$(PY) $(SCRIPTS)/preprocess/holdout_split.py --config $(PRE)/coconut_holdout_split.yaml

.PHONY: prep-real_space
prep-real_space: ## REAL-Space grid prep: ingest -> canon -> conformance -> holdout (no subsample)
	$(PY) $(SCRIPTS)/ingest/ingest_real_space.py --config $(COR)/real_space.yaml
	$(PY) $(SCRIPTS)/preprocess/canon_dedup.py --config $(PRE)/real_space_canon_dedup.yaml
	$(PY) $(SCRIPTS)/preprocess/filter_conformance.py --input-dir $(PROC)/real_space/canon_dedup_v1_full --output-dir $(PROC)/real_space/conformant_v1_full
	$(PY) $(SCRIPTS)/preprocess/holdout_split.py --config $(PRE)/real_space_holdout_split.yaml

.PHONY: prep-cycpeptmpdb
prep-cycpeptmpdb: ## CycPeptMPDB OOD prep: ingest -> canon (eval-only; no conformance/holdout)
	$(PY) $(SCRIPTS)/ingest/ingest_cycpeptmpdb.py --config $(COR)/cycpeptmpdb.yaml
	$(PY) $(SCRIPTS)/preprocess/canon_dedup.py --config $(PRE)/cycpeptmpdb_canon_dedup.yaml

.PHONY: prep-tmqm
prep-tmqm: ## tmQM OOD prep: ingest -> derive-tmqm (dative->OpenSMILES; eval-only)
	$(PY) $(SCRIPTS)/ingest/ingest_tmqm.py --config $(COR)/tmqm.yaml
	$(PY) $(SCRIPTS)/preprocess/derive_tmqm_opensmiles.py

.PHONY: prep-extras
prep-extras: ## Robustness-extras subsamples (PubChem+ZINC-22; run AFTER prep-pubchem/prep-zinc22)
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_extras_redraw_r1.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_extras_redraw_r2.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_extras_redraw_r3.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_extras_size_5m.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_extras_size_15m.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/pubchem_size700k.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/zinc22_extras_redraw_r1.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/zinc22_extras_redraw_r2.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/zinc22_extras_redraw_r3.yaml
	$(PY) $(SCRIPTS)/preprocess/hash_subsample.py --config $(PRE)/zinc22_size700k.yaml

.PHONY: prep-headline
prep-headline: prep-pubchem prep-zinc22 prep-coconut prep-real_space ## All 4 headline corpora

.PHONY: prep-ood
prep-ood: prep-cycpeptmpdb prep-tmqm ## Both OOD eval corpora

.PHONY: train
train: ## Train the 44-cell tokenizer grid (resumes; skips trained cells)
	$(PY) $(SCRIPTS)/tokenize/dispatch_grid_cell.py --all

.PHONY: train-extras
train-extras: ## Train the robustness-extras cells (resumes)
	$(PY) $(SCRIPTS)/tokenize/dispatch_extras_cell.py --all

.PHONY: audit
audit: ## Full audit: F95 + determinism + scaffold-retrain, over grid AND extras
	# Grid: F95 embedding-tail + train-twice determinism.
	$(PY) $(SCRIPTS)/audit/confirm_grid_cell_f95.py --all
	$(PY) $(SCRIPTS)/audit/verify_grid_cell_determinism.py --all
	# Scaffold-retrain materializes each BPE cell's scaffold.jsonl, required by
	# the scaffold measurement (grid AND extras BPE cells).
	$(PY) $(SCRIPTS)/tokenize/dispatch_grid_cell.py --retrain-scaffold --all
	$(PY) $(SCRIPTS)/tokenize/dispatch_extras_cell.py --retrain-scaffold --all
	# Extras F95: required by the deadzone measurement for extras-cell pairs
	# (confirm_grid_cell_f95.py covers the grid only).
	$(PY) $(SCRIPTS)/tokenize/dispatch_extras_cell.py --confirm-f95 --all
	# Robustness-extras audit summaries (seed-cap / prune-schedule /
	# merge-exhaustion) for Table A6, derived from the trained extras tokenizers.
	$(PY) $(SCRIPTS)/audit/build_extras_audits.py

.PHONY: measure
measure: ## Run all measurements (the seven + supplementary); idempotent
	$(PY) $(SCRIPTS)/measure/compute_absorption.py
	$(PY) $(SCRIPTS)/measure/compute_scaffold.py
	$(PY) $(SCRIPTS)/measure/compute_fertility.py
	$(PY) $(SCRIPTS)/measure/compute_nestedness.py
	$(PY) $(SCRIPTS)/measure/compute_jaccard.py
	$(PY) $(SCRIPTS)/measure/compute_closure.py
	$(PY) $(SCRIPTS)/measure/compute_fg_alignment.py
	$(PY) $(SCRIPTS)/measure/compute_noncanon.py
	$(PY) $(SCRIPTS)/measure/compute_distribution.py
	$(PY) $(SCRIPTS)/measure/compute_segmentation.py
	$(PY) $(SCRIPTS)/measure/compute_deadzone.py
	$(PY) $(SCRIPTS)/measure/compute_transfer.py
	$(PY) $(SCRIPTS)/measure/compute_ood_eval.py
	$(PY) $(SCRIPTS)/measure/compute_sensitivity.py
	$(PY) $(SCRIPTS)/measure/compute_marginal_jaccard.py

# --------------------------------------------------------------------------- #
# Cleanup                                                                      #
# --------------------------------------------------------------------------- #

# Regenerable per-cell / per-pair deposit directories under results/data/.
DEPOSIT_DIRS := absorption audits closure deadzone determinism distribution f95 \
                fertility fg_alignment jaccard nestedness noncanon ood_eval transfer \
                scaffold segmentation sensitivity \
                vocab_characterization

.PHONY: clean-deposits
clean-deposits: ## Clear regenerable results/data deposits for a clean re-measure (keeps .gitkeep)
	@echo "Clearing regenerable deposits under results/data/ (per-cell JSONs + aggregate tables)..."
	rm -rf $(addprefix results/data/,$(DEPOSIT_DIRS))
	rm -f results/data/*_table.json results/data/*_table.md
	@echo "Cleared. The measure -> generate pipeline regenerates these."

.PHONY: clean-local
clean-local: ## Remove local gitignored cruft (dispatch caches, smoke output, __pycache__)
	rm -rf configs/tokenizer/.dispatch_cache configs/tokenizer/.dispatch_cache_extras
	rm -rf results/data/_smoke
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
