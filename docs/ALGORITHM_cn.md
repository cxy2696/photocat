# 算法说明

[English](ALGORITHM_en.md) · 中文

仓库里有两条独立的建模路线，刻意没有强行合并：

1. **组成 → 带隙**：数据为任意化学式，使用 KG 子图 GNN 与图增强 Transformer；
2. **TiO2 工艺条件 → HER**：数据为小规模过程表，使用 RBF 核岭回归。

此外还有一条**合成场景**管线，用潜热模型生成材料，测试 KG、排序和主动学习。
它不参与真实数据的结论。以下分别说明。

---

## 1. 组成模型：GNN + 图增强 Transformer

### 1.1 化学式与元素角色

化学式解析（`photocat/data/formula.py`）支持嵌套括号和小数计量，例如
`Ag(AuS)2`、`Ba2Sb7HO14`、`Ca0.5Sr0.5TiO3`。解析后给每个元素分配
cation/anion 角色：O、N、S、Se、Te、F、Cl、Br、I 视为阴离子；无氧时 P 视为
阴离子；其余为阳离子。这只是组成启发式，不做氧化态平衡。

元素表包含 103 个元素。原先手写的 37 个元素保留文献离子半径；其余 63 个由
pymatgen 2026.10.24 的周期表数据生成，缺失的 d 电子数和丰度用保守占位值。
因此模型可以读任意 Matbench 化学式，但描述符精度低于专门的结构数据。

### 1.2 KG 子图

每个材料是一个小图：

- 节点：1 个材料节点 + 最多 6 个元素节点；
- 边：`material—hasElement—element`（权重为化学计量分数），元素共现边
  （权重 $2f_i f_j$），自环；
- 邻接按行归一化：$\hat A = D^{-1}(A+I)$。

节点特征：元素节点使用 9 个元素性质 + `is_material` + 角色 one-hot，共 14 维；
材料节点使用 9 个组成描述符。图的消息传递为

$$
H^{(\ell+1)} = \mathrm{ReLU}\!\left(\hat A H^{(\ell)} W^{(\ell)} + b^{(\ell)}\right),
$$

材料向量取材料节点。

### 1.3 Token 序列

```text
[CLS]  element×≤6  family  crystal  cocatalyst  method  dopant
       pH-bucket  heterojunction  sacrificial-agent
```

元素 token 由元素嵌入、角色嵌入、化学计量投影、GNN 节点嵌入投影和位置嵌入
相加。`[CLS]` 额外接收材料图向量与 10 维数值上下文（pH、比表面积、应变、缺陷、
异质结、掺杂分数、助催化剂负载、光强、催化剂用量、牺牲剂浓度）。真实数据缺失的
上下文字段映射到 unknown 槽，不会让模型崩溃。

### 1.4 图偏置注意力

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}} + \beta_h \tilde A_{ij},
\qquad
\alpha_{ij} = \mathrm{softmax}_{j\in\text{valid}}(s_{ij}),
$$

$\tilde A_{ij}$ 是 token 对应节点间的归一化邻接权重，$\beta_h$ 是每个头学习的
标量。padding key 被 mask。两层 pre-LN 残差块之后，CLS 向量与均值池化向量拼接，
经过一个共享 MLP，分别输出带隙、$\log(1+HER)$、稳定性 logit。

### 1.5 掩码多任务损失

真实数据往往只有一个目标。设 $m_{i,t}\in\{0,1\}$ 表示第 $i$ 个样本的第
$t$ 个目标是否存在：

$$
\mathcal L = \sum_t \lambda_t
\frac{\sum_i m_{i,t} w_i (\hat y_{i,t}-y_{i,t})^2}
     {\sum_i m_{i,t} w_i},
$$

稳定性项用带权 BCE。某个任务在当前 batch 没有标签时直接跳过。目标标准化使用
`nanmean`/`nanstd`；全缺失的目标列在推理时返回 NaN，而不是编造一个数。
这是当前代码能同时跑 Matbench（只有带隙）和 EES（带隙 + HER）的原因。

### 1.6 训练

- AdamW，cosine 衰减到初始学习率的 10%，梯度裁剪 5；
- 早停看验证损失；合成场景按晶族可选逆频率加权或重采样；
- 集成多个随机种子，均值作为预测，标准差作为不确定性；Matbench 使用 2 个普通模型 + 2 个 dropout=0.1 模型的混合集成。

Matbench 实测：4,603 条、元素体系留出，2 个普通模型 + 2 个 dropout=0.1 模型的混合集成，RMSE 0.799 eV、MAE 0.437 eV。
同数据上的 ridge/kNN 为 1.218/1.138 eV。详细协议见 `docs/RESULTS_cn.md`。

### 1.6 Dropout 与 PyTorch 实现

Dropout 施加在初始 token 嵌入之后，以及 FFN 输出加回残差之前：

$$
\tilde h = \frac{m\odot h}{1-p},\qquad m\sim\mathrm{Bernoulli}(1-p),
$$

dropout 集成成员取 $p=0.1$，普通成员取 $p=0$；推理时关闭。

`photocat/models/torch_transformer.py` 用 `torch.nn` 实现同一套图、注意力、
LayerNorm 和输出头。`load_from_numpy` 把 NumPy 参数复制到 Torch 模块；
`tests/test_torch.py` 在关闭 dropout 时检查两者的 logits 与 attention 是否
一致。这是等价的实现检查，不是一个无法对照的第二套模型。

---

## 2. 小表模型：RBF 核岭回归

EES 表只有 552 条完整记录、34 篇论文。组成 Transformer 在这类数据上会记住
工艺组合；因此 HER 基准使用核岭回归：

$$
f(x) = \sum_{i=1}^{n} \alpha_i
\exp\!\left(-\gamma \|x-x_i\|^2\right) + \bar y,
\qquad
(K+\lambda I)\alpha = y-\bar y.
$$

特征为助催化剂 one-hot、助催化剂负载、比表面积、带隙、牺牲剂 one-hot 与浓度、
光强、合成路线 one-hot。目标为 $\log_{10}(1+\text{Rate})$。所有特征先标准化。
$\gamma=0.03$、$\lambda=0.1$ 固定，不做测试折调参。

评估用 leave-papers-out 5 折：同一篇论文的所有记录只出现在一个折里。结果
R² 0.568、MAE 0.289、Spearman 0.698；线性 ridge 为 0.470、0.324、0.586。

这不是说核方法普遍优于 Transformer，而是说在 552 行、34 组的数据上，先验更强、
参数更少的模型更合适。README 中两个模型分开报告。

---

## 3. 知识图谱（合成场景）

`photocat/kg/` 提供 SQLite 三元组存储、本体、规则抽取、规则推理和 DistMult
链接预测。KG 的实体来自合成场景或用户自己的数据；当前没有把 Matbench 全量
4,603 条材料导入图谱作为真实知识库。

推理规则包括：

- `inferred:Candidate`：$1.23\le E_g\le 3.0$、$HER\ge 6$、稳定性 $\ge0.55$；
- `inferred:EarthAbundant`：组成元素地壳丰度均 $\ge10$ ppm；
- `inferred:Novel`：与训练库最大 Tanimoto $<0.8$；
- `inferred:HeterojunctionWith`：共享元素且能带交错
  $(E_{CB}^a<E_{CB}^b<E_{VB}^a<E_{VB}^b)$ 或对称情形。

规则结果带 `source` 字段，可回到具体三元组。抽取器只在合成语料上做单元测试，
不声称达到 SciBERT 的文献抽取水平。

---

## 4. 候选筛选（合成场景）

硬约束：带隙窗口、HER、稳定性、Tanimoto 新颖性。综合分数为

$$
0.42\,\tilde z(\log(1+HER))
+0.18\,e^{-\frac12\left(\frac{E_g-2.0}{0.55}\right)^2}
+0.14\,P_{stable}
+0.14\,\text{novelty}
+0.07\,\text{certainty}
+0.05\,\text{abundance}.
$$

shortlist 使用分数排序、Tanimoto 阈值 0.92、同 label 去重，并给出 Pareto 前沿。
这些权重是场景设置，不是从真实实验优化出来的。

---

## 5. 不确定性与主动学习

深度集成给出 $\mu(x)$、$\sigma(x)$；验证集上校准 split-conformal 区间：

$$
q = \mathrm{Quantile}_{\lceil(n+1)(1-\alpha)\rceil/n}(|y_i-\hat y_i|).
$$

Matbench 90% 区间半宽 2.221 eV、覆盖率 0.880。区间被高带隙离群点拉宽，说明
单点带隙预测不应被过度解读。

主动学习模拟只用于合成场景：岭回归代理 + bootstrap 标准差 + 带带隙窗口惩罚的
采集函数，与同预算随机采样比较。它验证的是流程可以闭环，不是真实实验节省。

---

## 6. 评测与复现

| 基准 | 划分 | 指标 |
| --- | --- | --- |
| Matbench 实验带隙 | 元素体系留出 | RMSE / MAE / R² / conformal coverage |
| EES HER | 按论文 5 折 | log R² / MAE / Spearman |
| 合成场景 | 组成+掺杂整组、晶族留出 | 流程完整性，不作为真实性能 |

数据哈希、DOI 与许可证见 `docs/DATA_cn.md`，引用核对见 `docs/REFERENCES_cn.md`。
`python scripts/verify_provenance.py` 会重新查询 Crossref/arXiv 并校验文件哈希。

## 7. 复杂度与限制

- Matbench：4,603 条、2 个模型 × 24 epoch，Apple Silicon CPU 约 1.5 分钟。
- 合成场景：12,000 条、3 个模型 × 30 epoch，约 5 分钟。
- 组成模型没有晶体结构、温度、形貌和缺陷信息；实验带隙本身的测量差异也限制了
  可达到的精度。
- EES 表 TiO2 比例极高，不能外推到其他材料家族；Rate 单位未统一。
- 仓库没有使用 Materials Project、Open Catalyst 或 SciBERT 的原始数据，相关
  来源只在提案讨论中出现，不能算作本仓库的验证数据。


## 8. 被测试的不变量

测试不只检查形状，也对应上面的公式：

| 不变量 | 测试 |
| --- | --- |
| 掩码 MSE 等于加权平均公式，梯度正确 | `tests/test_derivations.py` |
| RBF 核岭回归等于闭式解 | `tests/test_derivations.py` |
| 共形分位数采用 `ceil((n+1)(1-alpha))/n` 的有限样本修正 | `tests/test_derivations.py` |
| 缺失目标保持 NaN，并在损失中被掩码 | `tests/test_derivations.py`、`tests/test_external.py` |
| attention 每行和为 1，padding key 注意力为 0 | `tests/test_models.py`、`tests/test_torch.py` |
| 权重复制后 PyTorch 与 NumPy 输出一致 | `tests/test_torch.py` |
| 自动微分梯度与有限差分一致 | `tests/test_autograd.py` |
| 图邻接行归一化，padding token 没有图偏置 | `tests/test_data.py` |
