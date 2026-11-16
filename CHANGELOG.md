# Changelog

## [0.2.0] - 2026-11-16

### Added

- Real-data adapters and benchmarks:
  - Matbench `expt_gap` (4,603 experimental band gaps, MIT) with element-system holdout;
  - EES Catalysis TiO2 HER table (552 complete records, 34 papers) with leave-papers-out CV.
- RBF kernel ridge model for small process tables; paper-held-out log R² 0.568.
- Robust formula parser (nested parentheses, fractional stoichiometry) and a 103-element table.
- Masked multi-target training: missing bandgap/HER/stability no longer produces fake labels.
- Process-condition fields (cocatalyst loading, light intensity, catalyst loading,
  sacrificial agent/concentration) in the record schema and numeric features.
- Machine-readable provenance manifest, dataset checksums, reference verification script.
- Real-data benchmark CLI (`photocat bench`) and reports.

### Changed

- Scenario results are now clearly separated from real-data benchmarks.
- Transformer element vocabulary expanded from 40 to 103 elements.
- Reports and README rewritten in a plain research style.

### Removed

- The 20 approximate “literature anchor” records (no traceable per-record provenance).
- The unused `ctx_numeric` tensor.
- The unused anchor plumbing in the pipeline, KG and CLI.

## [0.1.0] - 2026-11-16

### Added

- NumPy reverse-mode autograd, LayerNorm, attention, embeddings, Adam.
- Physics-informed synthetic photocatalyst library (12 families, site mixing, dopants,
  co-catalysts, conditions) with group-aware and family-held-out splits.
- SQLite semantic KG: ontology, triple store, numeric property index, path queries.
- Literature corpus generator and rule-based NER/RE with gold evaluation.
- Rule reasoning (candidate / earth-abundant / novel / type-II heterojunction) with provenance.
- DistMult link prediction with filtered ranking and full-vocabulary stress metric.
- KG-subgraph GCN and graph-augmented transformer (graph-biased attention, multi-task heads).
- Deep-ensemble uncertainty, split-conformal intervals, bootstrap confidence intervals.
- Screening: bandgap window, HER, stability, novelty (Tanimoto), Pareto front, diversity.
- Active-learning simulation with bootstrap ridge surrogate and experimental-budget savings.
- Attention rollout, exact element Shapley values and condition permutation importance.
- End-to-end CLI, Markdown report, figures, 20 unit/integration tests.
