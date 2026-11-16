# 数据目录

`photocat data` 会生成以下文件（默认在 `data/`，也可用 `--out` 指定）：

| 文件 | 内容 |
| --- | --- |
| `materials.csv` | 12,000 条物理启发的合成光催化材料 |
| `anchors.csv` | 20 个文献光催化剂的近似带隙/HER（外部 sanity check） |
| `corpus.jsonl` | 合成文献句子 + gold 关系三元组 |

## `materials.csv` 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | str | `pc-000001` |
| `formula` | str | 基础化学式，如 `BiVO4`、`Ca0.5Sr0.5TiO3` |
| `label` | str | 含掺杂的显示名，如 `C3N4:N` |
| `family` | str | `rutile` / `perovskite` / `scheelite` / `spinel` / `carbon_nitride` / `aurivillius` / `oxynitride` / `sulfide` / `oxyhalide` / `corundum` |
| `crystal_system` | str | `tetragonal` / `orthorhombic` / `cubic` / `rhombohedral` / `hexagonal` |
| `elements` | str | `Ti:1:cation;O:2:anion` |
| `dopant`, `dopant_fraction` | str/float | 掺杂元素与分数；无掺杂为 `none` / 0 |
| `cocatalyst` | str | `none` / `Pt` / `MoS2` / `NiS` / `Co-Pi` / `Ni2P` |
| `method` | str | `hydrothermal` / `sol-gel` / `solid-state` / `CVD` / `thermal-polymerization` |
| `ph`, `surface_area`, `strain`, `defect` | float | 测试/结构条件 |
| `heterojunction` | str | `none` / `type-II` / `Z-scheme` |
| `eg`, `her`, `stability`, `cs_eff`, `cb`, `vb` | float | 目标与派生性质 |
| `descriptors` | JSON | 组成描述符；缺失时可用 `compute_descriptors` 重算 |
| `source` | str | `synthetic` / `literature_seed` |
| `is_anchor` | 0/1 | 是否文献锚点 |
| `split` | str | `train` / `val` / `test` / `ood` |

## 接入真实数据

1. 把 Materials Project / Open Catalyst / 文献表映射到上述字段；
2. 元素角色只用 `cation`、`anion`、`framework`、`dopant`；
3. `label` 必须唯一标识“组成 + 掺杂”，用于 KG 实体与按组划分；
4. 若没有 `descriptors`：

```python
from photocat.data.synth import load_materials, save_materials, compute_descriptors
materials = load_materials("raw.csv")
for material in materials:
    material.descriptors = compute_descriptors(
        material.elements, material.family, material.ph,
        material.surface_area, material.strain, material.defect,
    )
save_materials("data/real.csv", materials)
```

5. 重新运行 `python -m photocat demo`（或只训练/评测部分）。

> 注意：本仓库只附带合成数据。真实 DFT/实验数据的许可、单位（HER 建议统一为
> `mmol g-1 h-1`）与缺失值处理需要使用者自行负责。
