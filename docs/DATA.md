# 数据来源与处理

这个仓库包含两类数据用途完全不同的材料数据：

1. **真实数据基准**：Matbench 实验带隙、EES 光催化 HER 过程表；
2. **合成场景**：用于测试知识图谱、候选排序和主动学习流程的物理启发表。

合成场景不是实验或 DFT 数据，不能作为材料发现的证据。README 与
`docs/RESULTS.md` 中的主要结果来自第 1 类。

---

## 1. Matbench 实验带隙

| 项目 | 内容 |
| --- | --- |
| 数据文件 | `data/external/matbench_expt_gap.json.gz` |
| 原始地址 | https://ndownloader.figshare.com/files/17494814 |
| 数据集 DOI | 10.6084/m9.figshare.9765779 |
| 许可证 | MIT |
| SHA-256 | `1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a` |
| 记录数 | 4,604；清洗后 4,603 |
| 目标 | 实验带隙（eV） |
| 输入 | 化学式 |

数据由 Matbench 从 Zhuo 等人的补充材料整理，按组成去重，并保留同一组成
的最近实验值。原始说明见 Figshare 页面。引用：

> A. Dunn, Q. Wang, A. Ganose, D. Dopp, A. Jain. Benchmarking materials
> property prediction methods: the Matbench test set and Automatminer
> reference algorithm. *npj Computational Materials* 6, 138 (2020).
> DOI: 10.1038/s41524-020-00406-3

> Y. Zhuo, A. Mansouri Tehrani, J. Brgoch. Predicting the Band Gaps of
> Inorganic Solids by Machine Learning. *J. Phys. Chem. Lett.* 9, 1668–1673
> (2018). DOI: 10.1021/acs.jpclett.8b00124

清洗只删除一行历史拼写错误 `G1128Ga1As0.1P0.9`（元素 `G` 不存在）。
数值为 0 的金属/半金属记录保留。实验带隙混合了不同测量方法，同一组成只
保留一个值，因此不能与 DFT 带隙直接比较。

### 划分

按**元素体系**分组：同一组元素的全部组成只会出现在 train、val 或 test
之一。当前划分为 3,231 / 704 / 668。Matbench 官方使用固定 5 折与结构
输入；本仓库不是 leaderboard 提交，数字不能与官方榜单直接比较。

---

## 2. EES 光催化 HER 过程表

| 项目 | 内容 |
| --- | --- |
| 上游仓库 | https://github.com/yousof96/ligh_intensity_paper |
| 工作簿 | `paper_data/all_lit_data.xlsx` |
| 论文 DOI | 10.1039/D3EY00246B |
| 许可证 | 上游未声明；本仓库**不再分发**原始工作簿 |
| SHA-256（工作簿） | `9081eef866c4b14bc20a13ab0f6bccb5a4fe0d43474d21416175b7b9782beb03` |
| 原始记录 | 920 行，52 篇论文 |
| 完整记录 | 552 行，34 篇论文 |
| 目标 | `Rate`（作者整理的光催化产氢速率） |
| 输入 | 助催化剂及负载量、比表面积、带隙、牺牲剂及浓度、光强、合成路线 |

引用：

> Y. Haghshenas, W. P. Wong, D. Gunawan, A. Khataee, R. Keyikoglu,
> A. Razmjou, P. V. Kumar, C. Y. Toe, H. Masood, R. Amal, V. Sethu.
> Predicting the rates of photocatalytic hydrogen evolution over
> cocatalyst-deposited TiO2 using machine learning with active photon flux as
> a unifying feature. *EES Catalysis* 2, 612–623 (2024).
> DOI: 10.1039/D3EY00246B

### 处理

- 只保留 `Rate`、`SA`、`LI`、`BG` 均完整的行，得到 552 行；不做插值，
  避免把缺失的实验条件当成真实条件。
- 目标使用 `log10(1 + Rate)`。`Rate` 列沿用作者表中数值，不同论文的原始
  单位并不完全统一，因此本仓库只评估相对排序与对数误差，不把绝对值当作
  跨论文可比的 HER。
- 按论文分组做 5 折交叉验证（leave-papers-out），测试折的论文不参与训练。
- 这不是新材料筛选数据：全部样品为 TiO2 体系，适合做工艺条件/助催化剂
  的相对比较。

下载方式：

```bash
python scripts/fetch_data.py
```

脚本会校验 SHA-256，并把工作簿转换成 `data/external/ees_her.csv`。由于上游
没有声明再分发许可证，CSV 与工作簿都不计入版本库。

---

## 3. 合成场景数据（非真实）

`python -m photocat demo` 会生成一个 12,000 条材料的物理启发表，用于测试：

- KG 构建、规则推理、链接预测；
- 图增强 Transformer 的端到端训练；
- 候选筛选、Pareto、多样性；
- 主动学习模拟。

这些数据的带隙、HER、稳定性由 `photocat/data/synth.py` 的潜热模型产生，
不是实验或 DFT 结果。此前的“20 个文献材料近似锚点”已删除，因为它们没有
逐条可核对的出处。当前仓库不再把任何近似文献值当作验证数据。

---

## 4. 接入自己的数据

`Material` 的 CSV 字段见 `data/README.md`。数据加载后：

1. 用 `photocat.data.formula.parse_formula` 检查化学式；
2. 用 `photocat.data.synth.compute_descriptors` 计算组成描述符；
3. 缺失目标写 `NaN`（训练器会自动掩码）；
4. 用 `assign_group_splits` 或自己的 `split` 列划分；
5. 调用 `photocat.training.train_ensemble`。

支持全部 103 个元素（`photocat/data/elements_extra.py` 由 pymatgen 生成），
包括括号和分数计量的化学式。

---

## 5. 可追溯性检查

- 机器可读清单：`docs/references.json`
- 人工可读清单：`docs/REFERENCES.md`
- 校验脚本：

```bash
python scripts/verify_provenance.py
```

脚本会查询 Crossref/arXiv 并校验已下载数据文件的 SHA-256。2026-11-16
的校验结果是 8 条提案引用和 2 个数据源全部可达、哈希一致。
