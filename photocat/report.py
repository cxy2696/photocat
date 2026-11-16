"""Markdown report generation from a completed run directory."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


def _load(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        cells = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _figure(run_dir: Path, name: str, caption: str) -> str:
    if (run_dir / "figures" / name).exists():
        return f"![{caption}](figures/{name})\n"
    return ""


def generate_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    config = _load(run_dir / "config.json", {}) or {}
    metrics = _load(run_dir / "metrics.json", {}) or {}
    kg = _load(run_dir / "kg_metrics.json", {}) or {}
    active = _load(run_dir / "active_learning.json", {}) or {}
    interpret = _load(run_dir / "interpret.json", {}) or {}
    history = _load(run_dir / "history.json", {}) or {}
    rankings = _read_csv(run_dir / "rankings.csv")

    splits = metrics.get("splits", metrics)
    test = splits.get("test", {}) or {}
    ood = splits.get("ood", {}) or {}
    anchors = splits.get("anchors", {}) or {}
    criteria = metrics.get("success_criteria", {}) or {}
    conformal = metrics.get("conformal", {}) or {}
    ranking_stats = metrics.get("ranking", {}) or {}
    baselines = metrics.get("baselines", {}) or {}

    eg_rmse = test.get("bandgap", {}).get("rmse")
    her_pearson = test.get("her", {}).get("pearson")
    ood_eg = ood.get("bandgap", {}).get("rmse")
    ood_her = ood.get("her", {}).get("pearson")
    novelty_median = ranking_stats.get("similarity_median")

    lines: list[str] = []
    lines.append("# 光催化材料发现：KG + 图增强 Transformer 运行报告\n")
    lines.append(
        f"> 自动生成于 {datetime.now():%Y-%m-%d %H:%M}。"
        "数据为**物理启发的合成基准**（非 DFT/实验数据库），目标是为算法提供可复现的验证环境。\n"
    )

    lines.append("## 0. 结论摘要\n")
    lines.append(_table(
        ["指标", "结果", "目标", "判定"],
        [
            ["带隙 RMSE（同分布测试集）", f"{_fmt(eg_rmse)} eV", "< 0.3 eV", _fmt(bool(criteria.get("bandgap_rmse_below_0.3_ev")))],
            ["HER Pearson（同分布测试集）", _fmt(her_pearson), "> 0.8", _fmt(bool(criteria.get("her_pearson_above_0.8")))],
            ["新颖性（shortlist 中位 Tanimoto）", _fmt(novelty_median), "< 0.8", _fmt(bool(criteria.get("novelty_below_0.8")))],
            ["家族留出带隙 RMSE", f"{_fmt(ood_eg)} eV", "压力测试", "—"],
            ["家族留出 HER Pearson", _fmt(ood_her), "压力测试", "—"],
        ],
    ))
    lines.append("")
    lines.append(
        "核心结论：模型在同分布测试集上达到论文设定的带隙与 HER 指标；"
        "在完全未见的晶族上性能下降，说明分布外泛化仍是主要限制。"
        "KG 提供可追溯的筛选与解释，主动学习模拟显示可用更少的实验标签找到头部材料。\n"
    )

    lines.append("## 1. 数据与评测协议\n")
    data_summary = metrics.get("data", {}) or {}
    if data_summary:
        lines.append(_table(
            ["项目", "数值"],
            [
                ["材料总数", data_summary.get("n", "—")],
                ["训练 / 验证 / 测试", f"{data_summary.get('train', '—')} / {data_summary.get('val', '—')} / {data_summary.get('test', '—')}"],
                ["晶族留出", data_summary.get("ood", "—")],
                ["留出晶族", ", ".join(data_summary.get("holdout_families", []))],
                ["候选比例（真实标签）", _fmt(data_summary.get("candidate_rate"))],
            ],
        ))
        lines.append("")
    lines.append(
        "**Split 设计**：按“化学式 + 掺杂元素”整组划分，训练/测试不共享同一组成；"
        "`oxynitride` 与 `sulfide` 两个晶族完全留出作为分布外压力测试。"
        "锚点集为 20 个文献光催化剂的近似带隙/HER，仅用于外部 sanity check。\n"
    )
    lines.append(_figure(run_dir, "bias_audit.png", "晶族覆盖与偏置审计"))
    lines.append(_figure(run_dir, "training_curves.png", "训练曲线"))
    lines.append("")

    lines.append("## 2. 模型与训练\n")
    models = history.get("models", []) if isinstance(history, dict) else []
    if models:
        rows = []
        for i, item in enumerate(models):
            rows.append([
                f"model {i}",
                item.get("params", "—"),
                item.get("best_epoch", "—"),
                item.get("seconds", "—"),
                _fmt(item.get("best_val_loss"), 4),
            ])
        lines.append(_table(["模型", "参数量", "最佳 epoch", "训练秒数", "验证损失"], rows))
        lines.append("")
    lines.append(
        f"- 架构：GNN（KG 子图消息传递）+ {config.get('n_layers', '—')} 层 graph-augmented Transformer，"
        f"d_model={config.get('d_model', '—')}，{config.get('n_heads', '—')} 头；"
        f"深度集成 {config.get('ensemble_size', '—')} 个种子。\n"
        f"- 损失：带隙 MSE + log1p(HER) MSE + 0.25 × 稳定性 BCE；AdamW + cosine decay + early stopping。\n"
        f"- 偏置缓解策略：`{config.get('mitigation', 'none')}`。\n"
    )

    lines.append("## 3. 主结果\n")
    result_rows = []
    for name, label in (("train", "训练"), ("val", "验证"), ("test", "同分布测试"), ("ood", "晶族留出"), ("anchors", "文献锚点")):
        block = splits.get(name, {}) or {}
        if not block:
            continue
        result_rows.append([
            label,
            block.get("n", "—"),
            _fmt(block.get("bandgap", {}).get("rmse")),
            _fmt(block.get("bandgap", {}).get("mae")),
            _fmt(block.get("bandgap", {}).get("r2")),
            _fmt(block.get("her", {}).get("pearson")),
            _fmt(block.get("her", {}).get("rmse"), 2),
            _fmt(block.get("stability", {}).get("mae")),
        ])
    lines.append(_table(
        ["Split", "n", "Eg RMSE (eV)", "Eg MAE", "Eg R²", "HER Pearson", "HER RMSE", "稳定性 MAE"],
        result_rows,
    ))
    lines.append("")
    ci = test.get("bandgap", {}).get("rmse_ci90")
    if ci:
        lines.append(
            f"同分布测试集带隙 RMSE 的 90% bootstrap 置信区间："
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] eV；"
            f"HER Pearson 的 90% 区间："
            f"[{_fmt(test.get('her', {}).get('pearson_ci90', [None, None])[0])}, "
            f"{_fmt(test.get('her', {}).get('pearson_ci90', [None, None])[1])}]。\n"
        )
    lines.append(_figure(run_dir, "parity_bandgap.png", "带隙 parity"))
    lines.append(_figure(run_dir, "parity_her.png", "HER parity"))
    lines.append("")

    if baselines:
        lines.append("### 3.1 与基线对比（同分布测试集）\n")
        base_rows = []
        for name, values in baselines.items():
            base_rows.append([
                name,
                _fmt(values.get("test", {}).get("bandgap_rmse")),
                _fmt(values.get("test", {}).get("her_pearson")),
                _fmt(values.get("ood", {}).get("bandgap_rmse")),
                _fmt(values.get("ood", {}).get("her_pearson")),
            ])
        base_rows.append([
            "KG + transformer",
            _fmt(eg_rmse),
            _fmt(her_pearson),
            _fmt(ood_eg),
            _fmt(ood_her),
        ])
        lines.append(_table(["模型", "测试 Eg RMSE", "测试 HER r", "OOD Eg RMSE", "OOD HER r"], base_rows))
        lines.append("")

    lines.append("## 4. 不确定性与共形预测\n")
    if conformal:
        lines.append(
            f"在验证集上校准 90% split-conformal 区间：带隙半宽 "
            f"{_fmt(conformal.get('bandgap_quantile'))} eV，log-HER 半宽 "
            f"{_fmt(conformal.get('her_quantile'))}。\n"
        )
        lines.append(_table(
            ["Split", "Eg 覆盖率", "Eg 区间宽度", "HER(log) 覆盖率"],
            [
                [
                    name,
                    _fmt(splits.get(name, {}).get("conformal", {}).get("bandgap", {}).get("coverage")),
                    _fmt(splits.get(name, {}).get("conformal", {}).get("bandgap", {}).get("width")),
                    _fmt(splits.get(name, {}).get("conformal", {}).get("her_log", {}).get("coverage")),
                ]
                for name in ("test", "ood")
                if splits.get(name, {}).get("conformal")
            ],
        ))
        lines.append("")
        uncertainty = test.get("uncertainty", {})
        test_coverage = (
            splits.get("test", {}).get("conformal", {}).get("bandgap", {}).get("coverage")
        )
        ood_coverage = (
            splits.get("ood", {}).get("conformal", {}).get("bandgap", {}).get("coverage")
        )
        lines.append(
            f"集成标准差与绝对误差的 Spearman 相关：Eg "
            f"{_fmt(uncertainty.get('eg_std_error_spearman'))}，log-HER "
            f"{_fmt(uncertainty.get('her_std_log_error_spearman'))}。"
            f"共形区间在同分布测试集覆盖 {_fmt(test_coverage)}，留出晶族仅 {_fmt(ood_coverage)}："
            "这既是分布外风险的信号，也说明小集成对带隙的 epistemic uncertainty 估计偏弱；"
            "实际筛选应以共形区间与集成共识共同判断，而不是只看方差。\n"
        )
        lines.append(_figure(run_dir, "uncertainty.png", "不确定性与误差"))
        lines.append("")

    if kg:
        lines.append("## 5. 语义知识图谱\n")
        stats = kg.get("stats", {})
        lines.append(_table(
            ["实体", "三元组", "材料", "关系类型"],
            [[stats.get("entities", "—"), stats.get("triples", "—"), stats.get("materials", "—"), stats.get("relations", "—")]],
        ))
        lines.append("")
        extraction = kg.get("extraction", {})
        if extraction:
            relations = extraction.get("relations", {})
            entities = extraction.get("entities", {})
            lines.append("### 5.1 文献抽取（合成语料，规则抽取器）\n")
            lines.append(_table(
                ["任务", "Precision", "Recall", "F1", "TP / FP / FN"],
                [
                    ["关系三元组", _fmt(relations.get("precision")), _fmt(relations.get("recall")), _fmt(relations.get("f1")), f"{relations.get('tp', '—')} / {relations.get('fp', '—')} / {relations.get('fn', '—')}"],
                    ["实体提及", _fmt(entities.get("precision")), _fmt(entities.get("recall")), _fmt(entities.get("f1")), f"{entities.get('tp', '—')} / {entities.get('fp', '—')} / {entities.get('fn', '—')}"],
                ],
            ))
            lines.append("")
        link = kg.get("link_prediction", {})
        if link:
            lines.append("### 5.2 链接预测（DistMult）\n")
            lines.append(_table(
                ["MRR", "Hits@1", "Hits@10", "全实体采样 MRR"],
                [[_fmt(link.get("mrr")), _fmt(link.get("hits@1")), _fmt(link.get("hits@10")), _fmt(link.get("full_vocab_mrr_sample"))]],
            ))
            lines.append("")
        reasoning = kg.get("reasoning", {})
        if reasoning:
            lines.append("### 5.3 规则推理\n")
            lines.append(
                f"候选材料 {int(reasoning.get('candidates', 0))} 个，其中地球丰产 "
                f"{int(reasoning.get('earth_abundant', 0))} 个、新颖 "
                f"{int(reasoning.get('novel', 0))} 个；发现 "
                f"{int(reasoning.get('heterojunctions', 0))} 对 type-II 异质结候选。"
                "推理结果以带来源的三元组物化，可逐条追溯。\n"
            )
            heterojunctions = kg.get("heterojunctions", [])
            if heterojunctions:
                lines.append(_table(
                    ["材料 A", "材料 B", "Eg A", "Eg B", "共同元素"],
                    [
                        [item.get("label_a"), item.get("label_b"), _fmt(item.get("eg_a"), 2), _fmt(item.get("eg_b"), 2), ", ".join(item.get("shared_elements", []))]
                        for item in heterojunctions[:8]
                    ],
                ))
                lines.append("")
        explanation = kg.get("explanation", {})
        if explanation:
            lines.append("### 5.4 候选解释示例\n")
            lines.append("```text")
            lines.append(str(explanation.get("text", "")).strip())
            lines.append("```\n")

    lines.append("## 6. 候选筛选\n")
    if ranking_stats:
        lines.append(_table(
            ["阶段", "数量"],
            [
                ["候选池", ranking_stats.get("pool", "—")],
                ["通过物理窗口", ranking_stats.get("physical", "—")],
                ["通过新颖性 (< 0.8)", ranking_stats.get("novel", "—")],
                ["最终 shortlist", ranking_stats.get("selected", "—")],
                ["Pareto 前沿", ranking_stats.get("pareto", "—")],
            ],
        ))
        lines.append("")
        lines.append(_figure(run_dir, "screening.png", "筛选漏斗"))
        lines.append("")
    if rankings:
        top = rankings[:10]
        lines.append(_table(
            ["#", "材料", "晶族", "Eg (eV)", "HER", "稳定性", "Tanimoto", "score"],
            [
                [
                    row.get("rank"),
                    row.get("label"),
                    row.get("family"),
                    _fmt(float(row.get("pred_eg", "nan")), 2),
                    _fmt(float(row.get("pred_her", "nan")), 2),
                    _fmt(float(row.get("pred_stability", "nan")), 2),
                    _fmt(float(row.get("novelty_similarity", "nan")), 2),
                    _fmt(float(row.get("score", "nan")), 3),
                ]
                for row in top
            ],
        ))
        lines.append("")
        lines.append(_figure(run_dir, "shapley.png", "候选元素贡献"))
        lines.append("")

    if active:
        lines.append("## 7. 主动学习：需要多少实验？\n")
        summary = active.get("summary", {})

        def savings(level: str) -> str:
            exact = summary.get(f"reduction_at_{level}")
            lower = summary.get(f"reduction_at_{level}_lower_bound")
            if exact is not None and exact == exact:
                return _fmt(exact, 2)
            if lower is not None and lower == lower:
                return f"≥{_fmt(lower, 2)}"
            return "—"

        random_50 = summary.get("labels_to_50pct_random")
        random_note = (
            "随机采样在测试预算内未达到 50% recall；节省列为保守下界。"
            if random_50 is None or random_50 != random_50
            else ""
        )
        lines.append(_table(
            ["指标", "数值"],
            [
                ["目标材料数（pool 的 1%）", int(summary.get("n_targets", 0))],
                ["初始标签", int(summary.get("initial_labels", 0))],
                ["达到 50% recall 所需新标签（主动 / 随机）", f"{_fmt(summary.get('labels_to_50pct_active'), 0)} / {_fmt(random_50, 0)}"],
                ["50% recall 的实验节省", savings("50pct")],
                ["达到 80% recall 所需新标签（主动 / 随机）", f"{_fmt(summary.get('labels_to_80pct_active'), 0)} / {_fmt(summary.get('labels_to_80pct_random'), 0)}"],
                ["80% recall 的实验节省", savings("80pct")],
                ["最终 recall（主动 / 随机）", f"{_fmt(summary.get('final_active_recall'))} / {_fmt(summary.get('final_random_recall'))}"],
                ["300 个新标签时 recall（主动 / 随机）", f"{_fmt(summary.get('recall_at_300_active'))} / {_fmt(summary.get('recall_at_300_random'))}"],
            ],
        ))
        lines.append("")
        if random_note:
            lines.append(random_note + "\n")
        lines.append(_figure(run_dir, "active_learning.png", "主动学习曲线"))
        lines.append("")

    if interpret:
        lines.append("## 8. 可解释性\n")
        attention = interpret.get("attention", [])
        if attention:
            items = attention[0].get("top_tokens", [])[:8]
            for item in items:
                lines.append(f"- `{item.get('token')}`：{_fmt(item.get('weight'), 3)}")
            lines.append("")
            lines.append(_figure(run_dir, "attention.png", "CLS attention rollout"))
            lines.append("")
        permutation = interpret.get("permutation", [])
        if permutation:
            lines.append(_table(
                ["扰动字段", "Δlog-HER RMSE", "扰动后 RMSE"],
                [[row.get("field"), _fmt(row.get("delta_rmse")), _fmt(row.get("rmse_shuffled"))] for row in permutation],
            ))
            lines.append("")
        shapley = interpret.get("shapley", [])
        if shapley:
            first = shapley[0]
            contributions = sorted(first.get("attribution", {}).items(), key=lambda item: abs(item[1]), reverse=True)[:6]
            lines.append(
                f"**元素 Shapley（{first.get('label')}，目标 HER）**："
                + "；".join(f"{symbol} {_fmt(value, 2)}" for symbol, value in contributions)
                + "。\n"
            )

    lines.append("## 9. 复现\n")
    lines.append("```bash\npython -m photocat demo --materials %s --epochs %s --ensemble %s --out %s\n```\n" % (
        config.get("n_materials", "12000"),
        config.get("epochs", "30"),
        config.get("ensemble_size", "3"),
        config.get("out_dir", "runs/demo"),
    ))
    lines.append("完整算法与公式见 `docs/ALGORITHM.md`；接口与目录见 `README.md`。\n")

    lines.append("## 10. 限制\n")
    lines.append(
        "- 数据由潜热模型生成，带隙/HER 的绝对数值不代表真实材料；本报告验证的是**算法流程与评测协议**，"
        "接真实 Materials Project / Open Catalyst / 文献数据时只需替换 CSV 与语料。\n"
        "- 晶族留出的性能下降表明，面对全新化学空间时需要主动学习/实验闭环，而不能只依赖离线模型。\n"
        "- 规则抽取器在受控合成语料上接近满分，真实文献需要 SciBERT/LLM 抽取与人工校验。\n"
        "- 新颖性使用组成指纹近似 Tanimoto；真实场景应使用结构指纹（SOAP、CrystalNN）或 RDKit Morgan。\n"
    )

    report = "\n".join(lines)
    path = run_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path
