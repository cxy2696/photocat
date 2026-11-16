# Reference verification

[English](REFERENCES_en.md) · [中文](REFERENCES_cn.md)

The eight references in the proposal were checked against Crossref or arXiv on
2026-11-16. Titles, authors, venue/volume and DOI are listed below. A
machine-readable copy is in `docs/references.json`; rerun the check with
`python scripts/verify_provenance.py`.

| # | Reference | DOI / URL | Checked with |
| --- | --- | --- | --- |
| [1] | H. Masood, C. Y. Toe, W. Y. Teoh, V. Sethu, R. Amal. *Machine Learning for Accelerated Discovery of Solar Photocatalysts.* ACS Catal. 9(12), 11774–11787 (2019). | [10.1021/acscatal.9b02531](https://doi.org/10.1021/acscatal.9b02531) | Crossref |
| [2] | Y. Xu, H. Wang, W. Zhang, L. Xie, Y. Chen, F. Salim, Y. Zhang, J. Gooding. *AI-Empowered Catalyst Discovery: A Survey from Classical Machine Learning Approaches to Large Language Models.* arXiv:2502.13626 (2025). | [arXiv:2502.13626](https://arxiv.org/abs/2502.13626) | arXiv API |
| [3] | T. Xie, Y. Wan, H. Wang, I. Østrøm, et al. *Opinion Mining by Convolutional Neural Networks for Maximizing Discoverability of Nanomaterials.* J. Chem. Inf. Model. 64(7), 2746–2759 (2024). | [10.1021/acs.jcim.3c00746](https://doi.org/10.1021/acs.jcim.3c00746) | Crossref |
| [4] | F. Delgado-Licona, M. Abolhasani. *Research Acceleration in Self-Driving Labs: Technological Roadmap toward Accelerated Materials and Molecular Discovery.* Adv. Intell. Syst. 5(4), 2200331 (2023). | [10.1002/aisy.202200331](https://doi.org/10.1002/aisy.202200331) | Crossref |
| [5] | V. Venugopal, E. Olivetti. *MatKG: An autonomously generated knowledge graph in Material Science.* Sci. Data 11, 217 (2024). | [10.1038/s41597-024-03039-z](https://doi.org/10.1038/s41597-024-03039-z) | Crossref |
| [6] | V. Venugopal, S. Pai, E. Olivetti. *MatKG: The Largest Knowledge Graph in Materials Science — Entities, Relations, and Link Prediction through Graph Representation Learning.* NeurIPS 2022 AI4Materials Workshop; arXiv:2210.17340. | [arXiv:2210.17340](https://arxiv.org/abs/2210.17340) · [OpenReview](https://openreview.net/forum?id=xyJ_0-WCIZN) | arXiv / Semantic Scholar |
| [7] | P. Guo, W. Meng, Y. Bao. *Knowledge graph-guided data-driven design of ultra-high-performance concrete (UHPC) with interpretability and physicochemical reaction discovery capability.* Constr. Build. Mater. 430, 136502 (2024). | [10.1016/j.conbuildmat.2024.136502](https://doi.org/10.1016/j.conbuildmat.2024.136502) | Crossref |
| [8] | J. Liu, L. Liang, B. Su, D. Wu. *Transformative strategies in photocatalyst design: merging computational methods and deep learning.* J. Mater. Inform. 4(4) (2024). | [10.20517/jmi.2024.48](https://doi.org/10.20517/jmi.2024.48) | Crossref |

Notes:

- OpenReview uses a browser challenge for automated requests. Reference [6] was
  therefore checked against the arXiv record and Semantic Scholar; the
  OpenReview link is retained for human reading.
- Entries abbreviated as `et al.` in the proposal are expanded only to the
  authors confirmed by the metadata source; the meaning of the citation is not
  changed.

## Data citations

| Data | Citation | DOI | Licence |
| --- | --- | --- | --- |
| Matbench `expt_gap` | Dunn et al., *npj Comput. Mater.* 6, 138 (2020) | 10.1038/s41524-020-00406-3; data 10.6084/m9.figshare.9765779 | MIT |
| original experimental gaps | Zhuo et al., *J. Phys. Chem. Lett.* 9, 1668–1673 (2018) | 10.1021/acs.jpclett.8b00124 | see paper |
| EES HER data | Haghshenas et al., *EES Catalysis* 2, 612–623 (2024) | 10.1039/D3EY00246B | not declared upstream; not redistributed |

## Sources not used

The proposal also mentions Materials Project, Open Catalyst Project and
SciBERT. This repository does **not** download or redistribute raw data from
those sources:

- the band-gap benchmark uses the Matbench experimental subset;
- the HER benchmark uses the EES TiO2 process table;
- NER/RE is tested only on a synthetic corpus and makes no claim to reproduce
  SciBERT extraction quality.

This keeps every number in the repository tied to a specific file and DOI.
