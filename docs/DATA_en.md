# Data sources and processing

[English](DATA_en.md) · [中文](DATA_cn.md)

The repository uses data in two clearly separated ways:

1. **Real-data benchmarks**: Matbench experimental band gaps and the EES
   photocatalytic HER process table;
2. **Synthetic scenario**: a physics-inspired table used to test the knowledge
   graph, ranking and active-learning workflow.

The synthetic scenario is not experimental or DFT data and is not evidence of
materials discovery. The main results in the README and
`docs/RESULTS_en.md` come from the first category.

---

## 1. Matbench experimental band gap

| Item | Value |
| --- | --- |
| File | `data/external/matbench_expt_gap.json.gz` |
| Source | https://ndownloader.figshare.com/files/17494814 |
| Dataset DOI | 10.6084/m9.figshare.9765779 |
| Licence | MIT |
| SHA-256 | `1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a` |
| Records | 4,604; 4,603 after cleaning |
| Target | experimental band gap (eV) |
| Input | chemical formula |

Matbench compiled the table from the supplementary information of Zhuo et al.,
deduplicated by composition, and kept the experimental value closest to the
mean value for each composition. See the Figshare page for the original
description. Citations:

> A. Dunn, Q. Wang, A. Ganose, D. Dopp, A. Jain. Benchmarking materials
> property prediction methods: the Matbench test set and Automatminer
> reference algorithm. *npj Computational Materials* 6, 138 (2020).
> DOI: 10.1038/s41524-020-00406-3

> Y. Zhuo, A. Mansouri Tehrani, J. Brgoch. Predicting the Band Gaps of
> Inorganic Solids by Machine Learning. *J. Phys. Chem. Lett.* 9, 1668–1673
> (2018). DOI: 10.1021/acs.jpclett.8b00124

Cleaning removes one row with the historical typo `G1128Ga1As0.1P0.9`
(element `G` does not exist). Records with a gap of 0 (metals/semimetals) are
kept. Experimental gaps mix measurement methods and one value is kept per
composition, so they should not be compared directly with DFT gaps.

### Split

The split is grouped by **element system**: all compositions made from a given
set of elements appear in only one of train, validation or test. The current
split is 3,231 / 704 / 668. Matbench uses fixed five-fold splits and
structure-aware inputs; this repository is not a leaderboard submission and its
numbers are not directly comparable to the official leaderboard.

---

## 2. EES photocatalytic HER process table

| Item | Value |
| --- | --- |
| Upstream repository | https://github.com/yousof96/ligh_intensity_paper |
| Workbook | `paper_data/all_lit_data.xlsx` |
| Paper DOI | 10.1039/D3EY00246B |
| Licence | not declared upstream; not redistributed here |
| SHA-256 (workbook) | `9081eef866c4b14bc20a13ab0f6bccb5a4fe0d43474d21416175b7b9782beb03` |
| Original records | 920 rows, 52 papers |
| Complete records | 552 rows, 34 papers |
| Target | `Rate` (the authors' compiled photocatalytic hydrogen rate) |
| Inputs | cocatalyst and loading, surface area, band gap, sacrificial agent and concentration, light intensity, synthesis route |

Citations:

> Y. Haghshenas, W. P. Wong, D. Gunawan, A. Khataee, R. Keyikoglu,
> A. Razmjou, P. V. Kumar, C. Y. Toe, H. Masood, R. Amal, V. Sethu.
> Predicting the rates of photocatalytic hydrogen evolution over
> cocatalyst-deposited TiO2 using machine learning with active photon flux as a
> unifying feature. *EES Catalysis* 2, 612–623 (2024).
> DOI: 10.1039/D3EY00246B

### Processing

- Only rows with complete `Rate`, `SA`, `LI` and `BG` are kept (552 rows); no
  imputation is used, so missing experimental conditions are not invented.
- The target is `log10(1 + Rate)`. The `Rate` column follows the authors' table;
  original units differ between papers, so the benchmark evaluates relative
  ordering and log error, not absolute HER values.
- Evaluation uses leave-papers-out 5-fold cross-validation: papers in a test
  fold do not contribute to training.
- This is not a new-material screening dataset: all samples are TiO2-based and
  the data are suitable for relative process/cocatalyst comparisons.

Download:

```bash
python scripts/fetch_data.py
```

The script verifies SHA-256 and converts the workbook to
`data/external/ees_her.csv`. Because no upstream redistribution licence is
declared, neither the CSV nor the workbook is committed.

---

## 3. Synthetic scenario (not real)

`python -m photocat demo` generates a 12,000-record physics-inspired table to
exercise:

- KG construction, rule reasoning and link prediction;
- end-to-end training of the graph-augmented Transformer;
- candidate filtering, Pareto front and diversity selection;
- active-learning simulation.

The band gap, HER and stability are produced by the latent model in
`photocat/data/synth.py`; they are not experimental or DFT results. The
previous “20 literature anchors” have been removed because they lacked
record-level provenance. The repository no longer treats approximate
literature values as validation data.

---

## 4. Using your own data

See `data/README.md` for the CSV schema. After loading:

1. validate formulas with `photocat.data.formula.parse_formula`;
2. compute descriptors with `photocat.data.synth.compute_descriptors`;
3. write missing targets as `NaN` (the trainer masks them);
4. split with `assign_group_splits` or provide a `split` column;
5. call `photocat.training.train_ensemble`.

The element table covers 103 elements (`photocat/data/elements_extra.py` is
generated from pymatgen), including formulas with parentheses and fractional
stoichiometry.

---

## 5. Provenance checks

- Machine-readable manifest: `docs/references.json`
- Human-readable list: `docs/REFERENCES_en.md`
- Verification:

```bash
python scripts/verify_provenance.py
```

The script queries Crossref/arXiv and verifies the SHA-256 of downloaded data
files. The 2026-11-16 check found all eight proposal references and both data
sources reachable, with matching hashes.
