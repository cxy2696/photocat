# 偏置缓解消融（3k 快速配置）

同一数据种子、同一划分、同一模型尺寸（d_model=48、2 层、2 个种子、12 epoch），只改变训练时的晶族处理方式：

| 配置 | 测试 Eg RMSE | 测试 HER Pearson | 未见晶族 Eg RMSE | 未见晶族 HER Pearson |
| --- | --- | --- | --- | --- |
| `none`（均匀采样） | 0.163 | 0.848 | 0.400 | 0.499 |
| `balanced`（逆频率损失加权） | 0.162 | 0.856 | 0.370 | 0.541 |
| `oversample`（逆频率重采样） | 0.166 | 0.863 | **0.277** | **0.659** |

命令：

```bash
python -m photocat demo --fast --mitigation balanced --out runs/fast_balanced --no-figures
python -m photocat demo --fast --mitigation oversample --out runs/fast_oversample --no-figures
```

观察：

* 过采样把未见晶族的带隙 RMSE 降低约 **31%**（0.400 → 0.277 eV），HER 相关系数提高 **0.16**；
* 同分布带隙略差 0.003 eV，HER 反而提高 0.015，说明不是简单的“牺牲 ID 换 OOD”；
* 原因是训练集中 rutile / perovskite 占比高，均匀采样会让模型把“宽禁带氧化物”的先验带到所有晶族；
  逆频率重采样让 oxynitride / sulfide 之外的少数晶族获得更多梯度，元素层面的规律更可迁移。

注意：这是 3,000 条材料、单种子的快速消融；更大规模的主结果（`runs/demo`）仍采用 `none`，
用于观察未缓解时的真实分布外缺口。若要追求 OOD 指标，建议把 `--mitigation oversample` 作为默认，
并使用多个种子报告均值与方差。
