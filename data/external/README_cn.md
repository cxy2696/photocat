# 外部数据

[English](README.md) · 中文

真实数据基准使用两个外部来源。处理方式和许可证见 `docs/DATA_cn.md`，引用见
`docs/REFERENCES_cn.md`。

| 文件 | 来源 | 许可证 | 是否随仓库提供 |
| --- | --- | --- | --- |
| `matbench_expt_gap.json.gz` | Matbench，Figshare 10.6084/m9.figshare.9765779 | MIT | 是 |
| `ees_her.csv` | EES Catalysis 2024，github.com/yousof96/ligh_intensity_paper | 上游未声明 | 否 |

运行 `python scripts/fetch_data.py` 下载/重建两个文件并校验 SHA-256。生成的
`SOURCES.json` 记录 URL、哈希与引用，`scripts/verify_provenance.py` 可重新核对。
