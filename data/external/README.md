# External data

The real benchmarks use two external sources. See `docs/DATA_en.md` for processing
and `docs/REFERENCES_en.md` for citations.

| File | Source | Licence | Committed |
| --- | --- | --- | --- |
| `matbench_expt_gap.json.gz` | Matbench, Figshare 10.6084/m9.figshare.9765779 | MIT | yes |
| `ees_her.csv` | EES Catalysis 2024, github.com/yousof96/ligh_intensity_paper | not declared upstream | no |

Run `python scripts/fetch_data.py` to download/re-create both and verify
SHA-256. The generated `SOURCES.json` records the exact URLs, hashes and
citations used by `scripts/verify_provenance.py`.
