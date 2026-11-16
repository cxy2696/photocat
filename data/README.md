# 数据目录

```
data/
├── README.md
└── external/
    ├── SOURCES.json              # 哈希、URL、许可证、引用（可追溯清单）
    ├── matbench_expt_gap.json.gz # 真实实验带隙（MIT，随仓库提供）
    └── ees_her.csv               # 真实 HER 过程表（下载后生成，不再分发）
```

## 取数据

```bash
python scripts/fetch_data.py
```

脚本会校验 SHA-256。Matbench 文件只有 37 KB，已经放在 `data/external/`；
EES 工作簿约 195 KB，因上游未声明再分发许可证，只在本地下载并转换成 CSV。

## 真实数据

| 文件 | 用途 | 来源 |
| --- | --- | --- |
| `matbench_expt_gap.json.gz` | 组成 → 实验带隙 | Matbench / Figshare 10.6084/m9.figshare.9765779 |
| `ees_her.csv` | TiO2 工艺条件 → HER | EES Catalysis 2024, DOI 10.1039/D3EY00246B |

详细处理和限制见 `docs/DATA.md`。

## 合成场景

`python -m photocat data --out data` 仍会生成 `materials.csv` 与
`corpus.jsonl`。这些文件来自 `photocat/data/synth.py` 的潜热模型，**不是**
实验或 DFT 数据；只用于测试 KG、排序和主动学习流程。

## `materials.csv` 字段

| 字段 | 说明 |
| --- | --- |
| `id` | 唯一标识 |
| `formula` | 化学式；支持括号与分数计量，如 `Ca0.5Sr0.5TiO3` |
| `label` | 显示名，含掺杂时写作 `formula:dopant` |
| `family`, `crystal_system` | 晶族/晶系；真实数据没有时用 `unknown` |
| `elements` | `Ti:1:cation;O:2:anion` |
| `dopant`, `dopant_fraction` | 掺杂元素与分数 |
| `cocatalyst`, `cocatalyst_loading` | 助催化剂及负载量 |
| `method` | 合成路线 |
| `ph`, `surface_area`, `strain`, `defect` | 条件/结构量 |
| `light_intensity`, `catalyst_loading` | 光强与催化剂用量 |
| `sacrificial_agent`, `sacrificial_concentration` | 牺牲剂及浓度 |
| `heterojunction` | `none` / `type-II` / `Z-scheme` |
| `eg`, `her`, `stability`, `cs_eff`, `cb`, `vb` | 目标与派生性质；缺失写 `NaN` |
| `descriptors` | 组成描述符 JSON |
| `source`, `reference` | 数据来源与文献标识 |
| `split` | `train` / `val` / `test` / `ood` |

### 接入自己的数据

1. 按上表准备 CSV；化学式先用 `photocat.data.formula.parse_formula` 验证；
2. 缺 `descriptors` 时调用 `photocat.data.synth.compute_descriptors`；
3. 目标缺失保持空值或 `NaN`，训练器会按可用目标计算损失；
4. 用 `photocat.data.external.assign_group_splits` 按组成/论文/元素体系划分；
5. 用 `photocat.training.train_ensemble` 训练。

元素表覆盖 103 个元素。真实数据的 `crystal_system`、`family` 等未知字段
会映射到上下文词典的 unknown 槽，不会导致加载失败。
