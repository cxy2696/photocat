# 引用核对

[English](REFERENCES_en.md) · 中文

提案中的 8 条参考文献在 2026-11-16 通过 Crossref 或 arXiv 核对，标题、作者、
期刊/卷期与 DOI 如下。机器可读版本在 `docs/references.json`，可用
`python scripts/verify_provenance.py` 重新检查。

| # | 文献 | DOI / URL | 核对 |
| --- | --- | --- | --- |
| [1] | H. Masood, C. Y. Toe, W. Y. Teoh, V. Sethu, R. Amal. *Machine Learning for Accelerated Discovery of Solar Photocatalysts.* ACS Catal. 9(12), 11774–11787 (2019). | [10.1021/acscatal.9b02531](https://doi.org/10.1021/acscatal.9b02531) | Crossref |
| [2] | Y. Xu, H. Wang, W. Zhang, L. Xie, Y. Chen, F. Salim, Y. Zhang, J. Gooding. *AI-Empowered Catalyst Discovery: A Survey from Classical Machine Learning Approaches to Large Language Models.* arXiv:2502.13626 (2025). | [arXiv:2502.13626](https://arxiv.org/abs/2502.13626) | arXiv API |
| [3] | T. Xie, Y. Wan, H. Wang, I. Østrøm, et al. *Opinion Mining by Convolutional Neural Networks for Maximizing Discoverability of Nanomaterials.* J. Chem. Inf. Model. 64(7), 2746–2759 (2024). | [10.1021/acs.jcim.3c00746](https://doi.org/10.1021/acs.jcim.3c00746) | Crossref |
| [4] | F. Delgado-Licona, M. Abolhasani. *Research Acceleration in Self-Driving Labs: Technological Roadmap toward Accelerated Materials and Molecular Discovery.* Adv. Intell. Syst. 5(4), 2200331 (2023). | [10.1002/aisy.202200331](https://doi.org/10.1002/aisy.202200331) | Crossref |
| [5] | V. Venugopal, E. Olivetti. *MatKG: An autonomously generated knowledge graph in Material Science.* Sci. Data 11, 217 (2024). | [10.1038/s41597-024-03039-z](https://doi.org/10.1038/s41597-024-03039-z) | Crossref |
| [6] | V. Venugopal, S. Pai, E. Olivetti. *MatKG: The Largest Knowledge Graph in Materials Science — Entities, Relations, and Link Prediction through Graph Representation Learning.* NeurIPS 2022 AI4Materials Workshop; arXiv:2210.17340. | [arXiv:2210.17340](https://arxiv.org/abs/2210.17340) · [OpenReview](https://openreview.net/forum?id=xyJ_0-WCIZN) | arXiv / Semantic Scholar |
| [7] | P. Guo, W. Meng, Y. Bao. *Knowledge graph-guided data-driven design of ultra-high-performance concrete (UHPC) with interpretability and physicochemical reaction discovery capability.* Constr. Build. Mater. 430, 136502 (2024). | [10.1016/j.conbuildmat.2024.136502](https://doi.org/10.1016/j.conbuildmat.2024.136502) | Crossref |
| [8] | J. Liu, L. Liang, B. Su, D. Wu. *Transformative strategies in photocatalyst design: merging computational methods and deep learning.* J. Mater. Inform. 4(4) (2024). | [10.20517/jmi.2024.48](https://doi.org/10.20517/jmi.2024.48) | Crossref |

说明：

- OpenReview 对自动请求启用了浏览器校验，因此 [6] 的核对以 arXiv 与
  Semantic Scholar 记录为准；会议页面链接保留供人工查看。
- 提案中的作者列表使用 `et al.` 的条目，这里只补全核对到的前几位作者，
  不改变引用含义。

## 数据引用

| 数据 | 引用 | DOI | 许可证 |
| --- | --- | --- | --- |
| Matbench `expt_gap` | Dunn et al., *npj Comput. Mater.* 6, 138 (2020) | 10.1038/s41524-020-00406-3；数据 10.6084/m9.figshare.9765779 | MIT |
| 实验带隙原始整理 | Zhuo et al., *J. Phys. Chem. Lett.* 9, 1668–1673 (2018) | 10.1021/acs.jpclett.8b00124 | 见原论文 |
| EES HER 数据 | Haghshenas et al., *EES Catalysis* 2, 612–623 (2024) | 10.1039/D3EY00246B | 上游未声明，不再分发 |

## 未使用的来源

提案还提到 Materials Project、Open Catalyst Project 与 SciBERT。当前仓库
**没有**直接下载或再分发这些数据：

- 带隙基准使用 Matbench 的实验子集；
- HER 基准使用 EES 论文附带的 TiO2 过程表；
- NER/RE 只在合成语料上做单元测试，不声称复现 SciBERT 的文献抽取性能。

这样处理是为了让仓库里每个数字都能回到具体文件和 DOI。
