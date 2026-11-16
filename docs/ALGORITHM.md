# 算法说明

本文档给出 `photocat` 各模块的数学定义、实现位置与设计取舍。代码只依赖 NumPy；
GNN、Transformer、DistMult 与 Adam 都在 `photocat/autograd.py` 的反向模式自动微分上实现。

---

## 1. 总体流程

对应提案 Figure 1 的五个阶段：

```text
数据整合 → KG 构建（NER/RE） → GNN 节点嵌入 → 图增强 Transformer → 候选排序
             ↑                                        ↓
        规则推理 / 链接预测                     共形区间 + 主动学习
```

`python -m photocat demo` 顺序执行：

1. 生成物理启发的合成材料库与文献语料；
2. 构建 SQLite 知识图谱，做规则抽取、链接预测与语义推理；
3. 拟合特征器并训练深度集成；
4. 在同分布测试集、晶族留出集与文献锚点上评测；
5. 按带隙窗口、HER、稳定性与新颖性筛选并排序；
6. 模拟主动学习、计算注意力/Shapley/扰动重要性；
7. 生成图表与 Markdown 报告。

---

## 2. 合成数据潜热模型

> 数据是**基准**，不是 DFT。所有公式在 `photocat/data/synth.py` 中可审计。

### 2.1 组成描述符

对材料 \(m\)，令阳离子/骨架元素集合为 \(C\)，阴离子集合为 \(A\)，化学计量为 \(q_i\)：

\[
\bar\chi = \frac{\sum_{i\in C} q_i \chi_i}{\sum_{i\in C} q_i},\qquad
\bar r = \frac{\sum_{i\in C} q_i r_i}{\sum_{i\in C} q_i},\qquad
\sigma_r^2 = \frac{\sum_{i\in C} q_i (r_i-\bar r)^2}{\sum_{i\in C} q_i}.
\]

钙钛矿使用 Goldschmidt 容差因子：

\[
t = \frac{r_A + r_O}{\sqrt 2 (r_B + r_O)},\qquad d_{\text{dist}} = |t-1|,
\]

其他晶族用半径方差与结构基线组合：

\[
d_{\text{dist}} = d_{\text{family}} + \frac{\sigma_r}{\bar r + 10^{-6}}.
\]

### 2.2 带隙

\[
E_g = E_g^{\text{family}}
+ 0.45(1.75-\bar\chi)
+ \Delta_{\text{ion}}
+ 0.90\,d_{\text{dist}}
- 0.60\cdot 50\epsilon
- 0.55\eta
+ \Delta_{\text{dop}}
- \Delta_{\text{bow}}
+ \varepsilon_g,
\]

其中 \(\epsilon\) 是应变、\(\eta\) 是缺陷密度、\(\Delta_{\text{ion}}\) 是元素/阴离子经验偏移，
\(\Delta_{\text{bow}} = 0.55\sqrt{\mathrm{Var}(\chi)}\) 是双阳离子 bowing，\(\varepsilon_g\sim\mathcal N(0,0.14^2)\)。
最后裁剪到 \([0.55,5.2]\) eV。

### 2.3 能带边与 HER

\[
E_{CB} = -0.10 + 0.22(\bar\chi-1.75) - 0.35 d_{\text{dist}} + 0.015 \bar d + 0.10\Delta_{\text{dop}},
\qquad E_{VB}=E_{CB}+E_g.
\]

\[
\begin{aligned}
D_h &= \sigma(4(-0.05-E_{CB})),\\
D_o &= \sigma(4(E_{VB}-1.23)),\\
L &= \exp\!\left[-\tfrac12\left(\frac{E_g-1.95}{0.62}\right)^2\right],\\
\eta_{cs} &= \sigma\!\left(4\left(\eta_{cs}^{\text{family}} + 1.4(d_{\text{dist}}-d_{\text{family}}) + \Delta_{cs} + \Delta_{het} - 1.2\eta - 0.35\right)\right),\\
HER &= 22\,D_h D_o L\,\eta_{cs}\,
\left(\frac{S}{25}\right)^{0.30} f_{\text{cocat}} f_{pH}\,
e^{\varepsilon_{her}},
\end{aligned}
\]

\(S\) 为比表面积，\(f_{pH}=\exp[-\tfrac12((pH-7)/3.2)^2]\)，\(f_{\text{cocat}}\) 为助催化剂因子，
\(\varepsilon_{her}\sim\mathcal N(0,0.22^2)\)，结果裁剪到 \([0.01,80]\)。

### 2.4 稳定性

\[
P(\text{stable})=\sigma\!\left(b_{\text{family}}+0.8\left(1-\frac{|pH-7|}{6}\right)-1.5\eta+0.6\Delta_{stab}-0.5\cdot 50|\epsilon|+0.35(1-d_{\text{dist}})+\varepsilon_s\right).
\]

硫化物晶族 \(b_{\text{family}}=-1.2\)，因此光腐蚀风险高；这为“高 HER 但低稳定”的筛选权衡提供了真实感。

---

## 3. 图表示与特征化

每个材料被表示为 KG 子图：

* 节点：1 个材料节点 + 最多 6 个元素节点；
* 边：`material—hasElement—element`（权重为化学计量分数），元素共现边（权重 \(2f_i f_j\)），自环；
* 邻接矩阵按行归一化：\(\hat A = D^{-1}(A+I)\)。

材料节点特征为 9 维组成描述符；元素节点为 9 维元素性质 + `is_material` 标志 + 角色 one-hot，
共 14 维。实现在 `photocat/data/dataset.py`。

Token 序列：

```text
[CLS]  element×≤6  family  crystal  cocatalyst  method  dopant  pH-bucket  heterojunction
```

元素 token 由元素嵌入、角色嵌入、化学计量投影、**GNN 节点嵌入投影**和位置嵌入相加；
`[CLS]` 额外接收材料图池化向量与实验条件数值向量；7 个上下文 token 由类型/取值嵌入构成。

---

## 4. KG 子图 GNN

\[
H^{(0)} = X,\qquad
H^{(\ell+1)} = \mathrm{ReLU}\!\left(\hat A H^{(\ell)} W^{(\ell)} + b^{(\ell)}\right),
\]

材料向量 \(h_m = H^{(L)}_0\)，池化向量 \(h_{\text{pool}} = \frac{1}{|\mathcal V|}\sum_{v} H^{(L)}_v\)。
实现：`photocat/models/gnn.py`。GNN 与 Transformer 端到端联合训练。

---

## 5. 图增强 Transformer

### 5.1 图偏置注意力

对第 \(\ell\) 层：

\[
\tilde h = \mathrm{LN}(h),\quad
Q=\tilde h W_Q,\; K=\tilde h W_K,\; V=\tilde h W_V,
\]

\[
S_{ij} = \frac{q_i^\top k_j}{\sqrt{d_h}} + \beta_h \tilde A_{ij},
\qquad
\alpha_{ij} = \mathrm{softmax}_{j\in\text{valid}}(S_{ij}),
\]

其中 \(\tilde A_{ij}\) 是 token 所映射节点间的归一化邻接权重（由 `node_select @ A @ node_select^T` 得到），
\(\beta_h\) 是每个注意力头学习的标量。padding key 被 mask，padding query 的输出被池化 mask 忽略。

残差与 pre-LN：

\[
h \leftarrow h + \mathrm{Concat}(\alpha V)W_O,\qquad
h \leftarrow h + \mathrm{GELU}(\mathrm{LN}(h)W_1+b_1)W_2+b_2.
\]

### 5.2 输出头

\[
z = [\,h_{\text{CLS}}; \mathrm{mean}_{i\in\text{valid}} h_i\,],\qquad
\hat y = W_2\,\mathrm{GELU}(W_1 z + b_1)+b_2,
\]

三个独立头分别输出标准化带隙、标准化 \(\log(1+HER)\)、稳定性 logit。
实现：`photocat/models/transformer.py`，参数量约 9 万（d_model=64，2 层）。

### 5.3 损失

\[
\mathcal L = \lambda_g \mathrm{MSE}(\hat g, g)
+ \lambda_h \mathrm{MSE}(\hat h, \log(1+HER))
+ \lambda_s \mathrm{BCE}(\hat s, s),
\]

默认 \((\lambda_g,\lambda_h,\lambda_s)=(1,1,0.25)\)。支持按晶族逆频率加权（`balanced`）
或按晶族重采样（`oversample`）以缓解 TiO₂/钙钛矿过代表问题。

---

## 6. 语义知识图谱

### 6.1 本体与存储

`photocat/kg/ontology.py` 定义 11 类实体与 23 种关系。SQLite 中有三张核心表：

* `entities(id, name, type)`；
* `triples(subject, relation, object, value, unit, source)`；
* `properties(material_id, eg, her, stability, cb, vb, ...)` 用于快速数值筛选。

唯一索引 `(subject, relation, object, source)` 保证重复导入安全。

### 6.2 规则抽取

`photocat/kg/extraction.py` 用正则 + 化学式校验抽取：化学式、带隙、HER、pH、合成方法、
掺杂元素、助催化剂、晶系。语料由 `photocat/kg/corpus.py` 生成并带 gold 三元组，
`evaluate_extraction` 计算 micro P/R/F1。

### 6.3 语义推理

`photocat/kg/reasoning.py` 物化以下推理三元组（带 `source` 溯源）：

* `inferred:Candidate`：\(1.23\le E_g\le 3.0\)，\(HER\ge6\)，\(P_{stable}\ge0.55\)；
* `inferred:EarthAbundant`：所有元素地壳丰度 \(\ge 10\) ppm；
* `inferred:Novel`：与训练库最大 Tanimoto \(<0.8\)；
* `inferred:HeterojunctionWith`：共享元素且能带交错
  \((E_{CB}^a<E_{CB}^b<E_{VB}^a<E_{VB}^b)\) 或对称情形。

`explain(material_id)` 返回直接证据、推理三元组、异质结伙伴与到应用节点的路径。

### 6.4 链接预测

DistMult 打分：

\[
f(s,r,o) = \sum_k e_{s,k}\, w_{r,k}\, e_{o,k},
\]

以负采样 BCE 训练；评测采用 filtered ranking，并额外报告全实体采样的 MRR。
实现：`photocat/kg/link_prediction.py`。

---

## 7. 候选筛选与排序

硬约束：

\[
1.23 \le \hat E_g \le 3.0,\quad \widehat{HER}\ge 6,\quad \hat P_{stable}\ge0.55,\quad
\max_{m\in\mathcal D_{\text{train}}}\mathrm{Tanimoto}(m, c)<0.8.
\]

组成指纹（`photocat/data/fingerprint.py`）对元素、计量分箱、元素对、比例分箱做 BLAKE2b 哈希，
Tanimoto 为 \( |A\cap B|/|A\cup B| \)。

综合得分（权重可配置）：

\[
\text{score}=0.42\,\tilde z(\log(1+HER))
+0.18\,e^{-\frac12\left(\frac{\hat E_g-2.0}{0.55}\right)^2}
+0.14\,P_{stable}
+0.14\,\text{novelty}
+0.07\,\text{certainty}
+0.05\,\text{abundance}.
\]

* certainty：\(\exp(-(\sigma_{E_g}/0.30+\sigma_{HER}/2))\)；
* novelty：\(0.8-\text{sim}\) 归一化；
* abundance：\(\log_{10}(\min_i \text{abundance}_i+1)/3\) 截断。

shortlist 采用贪心“分数优先 + 相似度阈值 0.92 + 同 label 去重”；同时给出
Pareto 前沿（HER、稳定性、novelty、带隙接近度）。

---

## 8. 不确定性与共形预测

深度集成（不同种子）给出 \(\mu(x)\) 与 \(\sigma(x)\)。在验证集上做 split-conformal：

\[
q = \mathrm{Quantile}_{\lceil (n+1)(1-\alpha)\rceil/n}\left(|y_i-\hat y_i|\right),\quad \alpha=0.1,
\]

测试区间 \(\hat y\pm q\)。带隙与 log-HER 分别校准。报告同分布覆盖率、OOD 覆盖率与区间宽度。

---

## 9. 主动学习模拟

* 用岭回归代理模型（`photocat/models/baselines.py` 的 `_fit_ridge`）拟合已揭示标签；
* 5 个 bootstrap 模型估计 \(\sigma\)；
* 采集函数 \(\text{acq}=\hat z_{\log HER}+\beta\hat\sigma-\text{window penalty}\)；
* 每轮选 100 个，记录对“真实 top 1%”的 recall，与同预算随机采样对比；
* 报告中给出达到 50%/80% recall 所需新标签数与相对节省。

---

## 10. 评测协议

| Split | 规则 | 用途 |
| --- | --- | --- |
| train / val / test | 按 `(formula, dopant)` 整组分层划分 | 同分布泛化 |
| ood | 完全留出 `oxynitride`、`sulfide` | 分布外压力测试 |
| anchors | 20 个文献材料近似值 | 外部 sanity check |

指标：带隙 RMSE/MAE/R²/Pearson；HER RMSE/MAE/R²/Pearson/Spearman；稳定性 MAE；
bootstrap 90% CI；共形覆盖率；短名单新颖性中位数；主动学习 recall 曲线。

---

## 11. 复杂度与运行时间

* 模型参数量约 5–9 万（取决于 d_model）；12k 材料、30 epoch、3 模型在 Apple Silicon CPU 上约 8–12 分钟；
* 自实现的 NumPy 自动微分保证零深度学习框架依赖，但比 PyTorch 慢约一个数量级；
* 全部随机过程由种子控制，数据、划分、训练与筛选可复现。

---

## 12. 关键假设与限制

1. 合成数据的绝对带隙/HER 不代表真实材料，仅用于可复现的算法基准；
2. 组成指纹是结构指纹的代理，真实场景应使用 SOAP/CrystalNN/RDKit；
3. 规则抽取器在受控语料上接近满分，真实文献需 SciBERT/LLM + 人工校验；
4. 晶族留出性能下降，说明离线模型不能替代主动学习/实验闭环；
5. 模型只预测三个标量目标，未显式建模温度、助催化剂负载量、光照强度等条件。
