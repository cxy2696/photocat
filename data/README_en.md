# Data directory

[English](README_en.md) · [中文](README_cn.md)

```
data/
├── README.md
└── external/
    ├── SOURCES.json              # URLs, hashes, licences, citations
    ├── matbench_expt_gap.json.gz # real experimental gaps (MIT)
    └── ees_her.csv               # real HER process table (downloaded locally)
```

## Fetch

```bash
python scripts/fetch_data.py
```

The script verifies SHA-256. The Matbench file is only 37 KB and is committed.
The EES workbook (about 195 KB) is downloaded locally because the upstream
repository does not declare a redistribution licence.

## Real data

| File | Purpose | Source |
| --- | --- | --- |
| `matbench_expt_gap.json.gz` | composition → experimental gap | Matbench / Figshare 10.6084/m9.figshare.9765779 |
| `ees_her.csv` | TiO2 process conditions → HER | EES Catalysis 2024, DOI 10.1039/D3EY00246B |

Processing and limitations are described in `docs/DATA_en.md`.

## Synthetic scenario

`python -m photocat data --out data` still generates `materials.csv` and
`corpus.jsonl` from the latent model in `photocat/data/synth.py`. These are
**not** experimental or DFT data; they only exercise the KG, ranking and
active-learning workflow.

## `materials.csv` schema

| Column | Meaning |
| --- | --- |
| `id` | unique identifier |
| `formula` | chemical formula; parentheses and fractional counts are supported |
| `label` | display name, `formula:dopant` when doped |
| `family`, `crystal_system` | crystal family/system; `unknown` when absent |
| `elements` | `Ti:1:cation;O:2:anion` |
| `dopant`, `dopant_fraction` | dopant and fraction |
| `cocatalyst`, `cocatalyst_loading` | cocatalyst and loading |
| `method` | synthesis route |
| `ph`, `surface_area`, `strain`, `defect` | condition/structure values |
| `light_intensity`, `catalyst_loading` | light intensity and catalyst amount |
| `sacrificial_agent`, `sacrificial_concentration` | sacrificial agent and concentration |
| `heterojunction` | `none` / `type-II` / `Z-scheme` |
| `eg`, `her`, `stability`, `cs_eff`, `cb`, `vb` | targets and derived properties; missing values are `NaN` |
| `descriptors` | composition descriptors as JSON |
| `source`, `reference` | provenance |
| `split` | `train` / `val` / `test` / `ood` |

## Using your own CSV

1. Prepare a CSV with the columns above and validate formulas with
   `photocat.data.formula.parse_formula`.
2. Compute missing descriptors with
   `photocat.data.synth.compute_descriptors`.
3. Leave missing targets as `NaN`; the trainer masks them.
4. Split with `photocat.data.external.assign_group_splits` (e.g. by composition,
   paper or element system).
5. Train with `photocat.training.train_ensemble`.

The element table covers 103 elements. Unknown `family`, `crystal_system` or
other context values map to the unknown vocabulary slot and do not prevent
loading.
