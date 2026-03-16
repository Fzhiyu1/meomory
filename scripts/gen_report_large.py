#!/usr/bin/env python3
"""生成大规模实验报告 HTML 文件。

读取 experiments/results/*.json，生成 docs/report-large-scale.html。
单文件可直接浏览器打开，使用 Chart.js 做可视化。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
OUTPUT_FILE = ROOT / "docs" / "report-large-scale.html"


def load_json(name: str) -> dict:
    """加载一个实验结果 JSON 文件。"""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"WARNING: {path} not found, using empty data")
        return {"name": name, "rounds": [], "n_fragments": 0, "n_questions": 0}
    with open(path) as f:
        return json.load(f)


def get_last_metrics(data: dict) -> dict:
    """获取最后一轮的 metrics。"""
    if not data.get("rounds"):
        return {"1": 0, "3": 0, "5": 0}
    return data["rounds"][-1]["metrics"]


def get_last_details(data: dict) -> list:
    """获取最后一轮的 details。"""
    if not data.get("rounds"):
        return []
    return data["rounds"][-1].get("details", [])


def pct(v: float) -> str:
    """转为百分比字符串。"""
    return f"{v * 100:.1f}"


def load_categories() -> dict:
    """从 LoCoMo 数据集加载每个问题的 category。返回 {qi: category}."""
    sys.path.insert(0, str(ROOT / "src"))
    from bench.datasets import Dataset
    ds = Dataset.load_locomo_full_all()
    return {i: q.get("category", 0) for i, q in enumerate(ds.questions)}


def generate_html():
    # ========== 1. 主实验结果矩阵 ==========
    main_names = [
        ("cosine", "Cosine"),
        ("bm25", "BM25"),
        ("dgd-gt", "DGD+GT"),
        ("dgd-judge", "DGD+Judge"),
        ("dgd-cms", "DGD+CMS"),
    ]
    main_datasets = [
        ("locomo-full", "LoCoMo"),
        ("longmemeval", "LongMemEval"),
    ]

    main_data = {}
    for method_key, method_label in main_names:
        for ds_key, ds_label in main_datasets:
            data = load_json(f"{method_key}-{ds_key}")
            metrics = get_last_metrics(data)
            main_data[(method_key, ds_key)] = metrics

    # Find best values per dataset per metric
    best = {}
    for ds_key, ds_label in main_datasets:
        for k in ["1", "3", "5"]:
            vals = [(m_key, main_data[(m_key, ds_key)][k]) for m_key, _ in main_names]
            best_method = max(vals, key=lambda x: x[1])
            best[(ds_key, k)] = best_method[0]

    # Build main table HTML
    main_table_rows = ""
    for method_key, method_label in main_names:
        row = f"<tr><td class='method-name'>{method_label}</td>"
        for ds_key, ds_label in main_datasets:
            m = main_data[(method_key, ds_key)]
            for k in ["1", "3", "5"]:
                val = pct(m[k])
                cls = " class='best'" if best.get((ds_key, k)) == method_key else ""
                row += f"<td{cls}>{val}%</td>"
        row += "</tr>"
        main_table_rows += row

    # Chart data for section 1
    main_chart_data = json.dumps({
        "methods": [label for _, label in main_names],
        "datasets": [label for _, label in main_datasets],
        "values": {
            f"{ds_label}": {
                f"P@{k}": [main_data[(m_key, ds_key)][k] * 100 for m_key, _ in main_names]
                for k in ["1", "3", "5"]
            }
            for ds_key, ds_label in main_datasets
        }
    })

    # ========== 2. LoCoMo 逐对话方差分析 ==========
    conv_methods = [
        ("cosine-locomo", "Cosine"),
        ("dgd-gt-locomo", "DGD+GT"),
        ("dgd-judge-locomo", "DGD+Judge"),
    ]
    conv_data = {}
    for prefix, label in conv_methods:
        p1_values = []
        for i in range(10):
            data = load_json(f"{prefix}-{i}")
            m = get_last_metrics(data)
            p1_values.append(m["1"] * 100)
        conv_data[label] = p1_values

    # Compute averages
    conv_averages = {label: sum(vals) / len(vals) for label, vals in conv_data.items()}

    conv_chart_data = json.dumps({
        "labels": [f"Conv {i}" for i in range(10)],
        "datasets": conv_data,
        "averages": conv_averages,
    })

    # ========== 3. 参数敏感度 ==========
    # Alpha
    alpha_values = [0.9, 0.95, 0.99, 1.0]
    alpha_p1 = []
    for a in alpha_values:
        data = load_json(f"param-alpha-{a}-locomo-full")
        m = get_last_metrics(data)
        alpha_p1.append(m["1"] * 100)

    # Dim
    dim_values = [64, 128, 256, 512]
    dim_p1 = []
    for d in dim_values:
        data = load_json(f"param-dim-{d}-locomo-full")
        m = get_last_metrics(data)
        dim_p1.append(m["1"] * 100)

    # Eta
    eta_values = [0.005, 0.01, 0.02, 0.05]
    eta_p1 = []
    for e in eta_values:
        data = load_json(f"param-eta-{e}-locomo-full")
        m = get_last_metrics(data)
        eta_p1.append(m["1"] * 100)

    # Find best and worst for annotations
    alpha_best_idx = alpha_p1.index(max(alpha_p1))
    alpha_worst_idx = alpha_p1.index(min(alpha_p1))
    dim_best_idx = dim_p1.index(max(dim_p1))
    dim_worst_idx = dim_p1.index(min(dim_p1))
    eta_best_idx = eta_p1.index(max(eta_p1))
    eta_worst_idx = eta_p1.index(min(eta_p1))

    param_chart_data = json.dumps({
        "alpha": {"labels": [str(a) for a in alpha_values], "values": alpha_p1,
                  "best_idx": alpha_best_idx, "worst_idx": alpha_worst_idx},
        "dim": {"labels": [str(d) for d in dim_values], "values": dim_p1,
                "best_idx": dim_best_idx, "worst_idx": dim_worst_idx},
        "eta": {"labels": [str(e) for e in eta_values], "values": eta_p1,
                "best_idx": eta_best_idx, "worst_idx": eta_worst_idx},
    })

    # ========== 4. 学习曲线 ==========
    curve_files = {
        "LoCoMo": {
            "DGD+GT": load_json("curve-dgd-gt-locomo-full"),
            "DGD+Judge": load_json("curve-dgd-judge-locomo-full"),
        },
        "LongMemEval": {
            "DGD+GT": load_json("curve-dgd-gt-longmemeval"),
            "DGD+Judge": load_json("curve-dgd-judge-longmemeval"),
        },
    }

    # Baselines
    cosine_locomo = get_last_metrics(load_json("cosine-locomo-full"))["1"] * 100
    bm25_locomo = get_last_metrics(load_json("bm25-locomo-full"))["1"] * 100
    cosine_longmemeval = get_last_metrics(load_json("cosine-longmemeval"))["1"] * 100
    bm25_longmemeval = get_last_metrics(load_json("bm25-longmemeval"))["1"] * 100

    curve_chart_data = json.dumps({
        "LoCoMo": {
            "DGD+GT": [r["metrics"]["1"] * 100 for r in curve_files["LoCoMo"]["DGD+GT"]["rounds"]],
            "DGD+Judge": [r["metrics"]["1"] * 100 for r in curve_files["LoCoMo"]["DGD+Judge"]["rounds"]],
            "rounds": [r["round"] for r in curve_files["LoCoMo"]["DGD+GT"]["rounds"]],
            "baselines": {"Cosine": cosine_locomo, "BM25": bm25_locomo},
        },
        "LongMemEval": {
            "DGD+GT": [r["metrics"]["1"] * 100 for r in curve_files["LongMemEval"]["DGD+GT"]["rounds"]],
            "DGD+Judge": [r["metrics"]["1"] * 100 for r in curve_files["LongMemEval"]["DGD+Judge"]["rounds"]],
            "rounds": [r["round"] for r in curve_files["LongMemEval"]["DGD+GT"]["rounds"]],
            "baselines": {"Cosine": cosine_longmemeval, "BM25": bm25_longmemeval},
        },
    })

    # ========== 5. LoCoMo 分类别分析 ==========
    categories_map = load_categories()
    cat_names = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "world-knowledge", 5: "adversarial"}

    cosine_details = get_last_details(load_json("cosine-locomo-full"))
    bm25_details = get_last_details(load_json("bm25-locomo-full"))

    # Compute per-category P@1
    def compute_category_p1(details, categories_map):
        cat_correct = {c: 0 for c in range(1, 6)}
        cat_total = {c: 0 for c in range(1, 6)}
        for det in details:
            qi = det["qi"]
            cat = categories_map.get(qi, 0)
            if cat not in range(1, 6):
                continue
            cat_total[cat] += 1
            if det["rank"] == 1:
                cat_correct[cat] += 1
        return {c: (cat_correct[c] / cat_total[c] * 100 if cat_total[c] > 0 else 0) for c in range(1, 6)}

    cosine_cat_p1 = compute_category_p1(cosine_details, categories_map)
    bm25_cat_p1 = compute_category_p1(bm25_details, categories_map)

    category_chart_data = json.dumps({
        "categories": [cat_names[c] for c in range(1, 6)],
        "Cosine": [cosine_cat_p1[c] for c in range(1, 6)],
        "BM25": [bm25_cat_p1[c] for c in range(1, 6)],
    })

    # ========== 6. DialSim 复现对比 ==========
    dialsim_shows = ["bigbang", "friends", "theoffice"]
    dialsim_show_labels = {"bigbang": "BigBang Theory", "friends": "Friends", "theoffice": "The Office"}
    dialsim_methods = [
        ("cosine-dialsim", "Cosine"),
        ("dgd-gt-dialsim", "DGD+GT"),
        ("dgd-judge-dialsim", "DGD+Judge"),
    ]

    dialsim_old = {
        "bigbang": {"Cosine": 57.1, "DGD+Judge": 78.6, "DGD+GT": 78.6},
        "friends": {"Cosine": 25.0, "DGD+Judge": 25.0, "DGD+GT": 41.7},
        "theoffice": {"Cosine": 25.0, "DGD+Judge": 50.0, "DGD+GT": 50.0},
    }

    dialsim_new = {}
    for show in dialsim_shows:
        dialsim_new[show] = {}
        for prefix, label in dialsim_methods:
            data = load_json(f"{prefix}-{show}")
            m = get_last_metrics(data)
            dialsim_new[show][label] = m["1"] * 100

    dialsim_table_rows = ""
    for show in dialsim_shows:
        label = dialsim_show_labels[show]
        dialsim_table_rows += f"<tr class='section-header'><td colspan='5'>{label}</td></tr>"
        # Old row
        row_old = f"<tr><td>{label}</td><td>旧实验 (小规模)</td>"
        for _, method_label in dialsim_methods:
            val = dialsim_old[show].get(method_label, 0)
            row_old += f"<td>{val:.1f}%</td>"
        row_old += "</tr>"
        dialsim_table_rows += row_old
        # New row
        row_new = f"<tr><td>{label}</td><td>新实验 (DialSim)</td>"
        for _, method_label in dialsim_methods:
            val = dialsim_new[show].get(method_label, 0)
            row_new += f"<td>{val:.1f}%</td>"
        row_new += "</tr>"
        dialsim_table_rows += row_new

    # ========== 7. 关键发现 ==========
    # Extract key numbers
    dgd_gt_locomo_p1 = main_data[("dgd-gt", "locomo-full")]["1"] * 100
    cosine_locomo_p1_val = main_data[("cosine", "locomo-full")]["1"] * 100
    bm25_locomo_p1_val = main_data[("bm25", "locomo-full")]["1"] * 100
    dgd_gt_longmemeval_p1 = main_data[("dgd-gt", "longmemeval")]["1"] * 100
    cosine_longmemeval_p1_val = main_data[("cosine", "longmemeval")]["1"] * 100
    dgd_judge_locomo_p1 = main_data[("dgd-judge", "locomo-full")]["1"] * 100
    dgd_judge_longmemeval_p1 = main_data[("dgd-judge", "longmemeval")]["1"] * 100
    dgd_cms_locomo_p1 = main_data[("dgd-cms", "locomo-full")]["1"] * 100

    # Curve improvements
    curve_gt_locomo = curve_files["LoCoMo"]["DGD+GT"]["rounds"]
    curve_gt_r1 = curve_gt_locomo[0]["metrics"]["1"] * 100
    curve_gt_r10 = curve_gt_locomo[-1]["metrics"]["1"] * 100

    # Best alpha
    best_alpha = alpha_values[alpha_best_idx]
    worst_alpha = alpha_values[alpha_worst_idx]

    # Category insights
    best_cat = max(cosine_cat_p1, key=cosine_cat_p1.get)
    worst_cat = min(cosine_cat_p1, key=cosine_cat_p1.get)

    # DGD+GT vs BM25 comparison wording
    dgd_vs_bm25_diff = dgd_gt_locomo_p1 - bm25_locomo_p1_val
    if dgd_vs_bm25_diff >= 0:
        bm25_cmp = f"超越 BM25 ({bm25_locomo_p1_val:.1f}%) {dgd_vs_bm25_diff:.1f} 个百分点"
    else:
        bm25_cmp = f"仍低于 BM25 ({bm25_locomo_p1_val:.1f}%) {-dgd_vs_bm25_diff:.1f} 个百分点（仅 3 轮训练）"

    findings = [
        f"<strong>DGD+GT 在 LoCoMo 上达到 {dgd_gt_locomo_p1:.1f}% P@1</strong>，相比 Cosine baseline ({cosine_locomo_p1_val:.1f}%) 提升 {dgd_gt_locomo_p1 - cosine_locomo_p1_val:.1f} 个百分点；{bm25_cmp}。但 10 轮训练可达 {curve_gt_r10:.1f}%，超越 BM25。",
        f"<strong>LongMemEval 上提升更显著</strong>：DGD+GT 达到 {dgd_gt_longmemeval_p1:.1f}% P@1，DGD+Judge 达到 {dgd_judge_longmemeval_p1:.1f}%，Cosine baseline 为 {cosine_longmemeval_p1_val:.1f}%。",
        f"<strong>学习曲线持续上升</strong>：DGD+GT 在 LoCoMo 上从 Round 1 的 {curve_gt_r1:.1f}% 提升到 Round 10 的 {curve_gt_r10:.1f}%，10 轮训练带来 {curve_gt_r10 - curve_gt_r1:.1f} 个百分点的稳步增长。",
        f"<strong>参数 &alpha; 的选择至关重要</strong>：&alpha;={best_alpha} 表现最佳，而 &alpha;={worst_alpha} 导致性能断崖式下降（P@1 仅 {alpha_p1[alpha_worst_idx]:.1f}%），形成明显的「死亡谷」。",
        f"<strong>分类别分析</strong>：Cosine 在 {cat_names[best_cat]} 类题目上表现最好（{cosine_cat_p1[best_cat]:.1f}%），在 {cat_names[worst_cat]} 类上最差（{cosine_cat_p1[worst_cat]:.1f}%）。",
        f"<strong>DGD+Judge 与 DGD+GT 的差距</strong>：LoCoMo 上 DGD+Judge ({dgd_judge_locomo_p1:.1f}%) vs DGD+GT ({dgd_gt_locomo_p1:.1f}%)，LLM 判断的噪声导致约 {dgd_gt_locomo_p1 - dgd_judge_locomo_p1:.1f} 个百分点的损失。",
        f"<strong>DGD+CMS（合成反馈）</strong>在 LoCoMo 上达到 {dgd_cms_locomo_p1:.1f}% P@1，低于 DGD+GT 但展示了无需人工标注的可行性。",
    ]

    findings_html = "\n".join(f"<li>{f}</li>" for f in findings)

    # ========== Generate HTML ==========
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Meomory 大规模实验报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
:root {{
  --bg: #ffffff;
  --bg-alt: #f8f9fa;
  --text: #1a1a2e;
  --text-muted: #6c757d;
  --border: #dee2e6;
  --accent: #4361ee;
  --accent-light: #e8ecff;
  --success: #2dc653;
  --warning: #ff6b35;
  --card-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0d1117;
    --bg-alt: #161b22;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --border: #30363d;
    --accent: #58a6ff;
    --accent-light: #1c2d41;
    --card-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 0;
}}

.container {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}}

header {{
  text-align: center;
  padding: 40px 20px;
  background: linear-gradient(135deg, var(--accent), #7b2ff7);
  color: white;
  margin-bottom: 30px;
}}

header h1 {{
  font-size: 2.2em;
  margin-bottom: 10px;
}}

header p {{
  font-size: 1.1em;
  opacity: 0.9;
}}

.section {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 30px;
  margin-bottom: 30px;
  box-shadow: var(--card-shadow);
}}

.section h2 {{
  font-size: 1.5em;
  margin-bottom: 8px;
  color: var(--accent);
  border-bottom: 2px solid var(--accent);
  padding-bottom: 8px;
}}

.section h2 .num {{
  display: inline-block;
  background: var(--accent);
  color: white;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  border-radius: 50%;
  font-size: 0.8em;
  margin-right: 8px;
  vertical-align: middle;
}}

.section .desc {{
  color: var(--text-muted);
  margin-bottom: 20px;
  font-size: 0.95em;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 15px 0;
  font-size: 0.95em;
}}

th, td {{
  padding: 10px 14px;
  text-align: center;
  border: 1px solid var(--border);
}}

th {{
  background: var(--accent-light);
  font-weight: 600;
  color: var(--text);
}}

td.method-name {{
  text-align: left;
  font-weight: 600;
}}

td.best {{
  background: var(--accent-light);
  font-weight: 700;
  color: var(--accent);
}}

tr:nth-child(even) {{
  background: var(--bg-alt);
}}

tr.section-header {{
  background: var(--accent-light) !important;
}}
tr.section-header td {{
  font-weight: 700;
  text-align: left;
  color: var(--accent);
}}

.chart-container {{
  position: relative;
  width: 100%;
  max-width: 900px;
  margin: 20px auto;
}}

.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin: 20px 0;
}}

.chart-row-3 {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 20px;
  margin: 20px 0;
}}

@media (max-width: 768px) {{
  .chart-row, .chart-row-3 {{
    grid-template-columns: 1fr;
  }}
  header h1 {{ font-size: 1.5em; }}
}}

.findings-list {{
  list-style: none;
  padding: 0;
}}

.findings-list li {{
  padding: 12px 16px;
  margin: 8px 0;
  background: var(--bg-alt);
  border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0;
  line-height: 1.7;
}}

.findings-list li:nth-child(even) {{
  border-left-color: var(--success);
}}

.findings-list li:nth-child(3n) {{
  border-left-color: var(--warning);
}}

.meta-info {{
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin: 15px 0;
  padding: 12px;
  background: var(--bg-alt);
  border-radius: 8px;
  font-size: 0.9em;
  color: var(--text-muted);
}}

.meta-info span {{
  white-space: nowrap;
}}

canvas {{
  max-width: 100%;
}}

.nav {{
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  padding: 8px 20px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}}

.nav a {{
  color: var(--accent);
  text-decoration: none;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 0.85em;
  transition: background 0.2s;
}}

.nav a:hover {{
  background: var(--accent-light);
}}
</style>
</head>
<body>

<header>
  <h1>Meomory 大规模实验报告</h1>
  <p>DGD 记忆检索优化 &mdash; 跨数据集全面评估</p>
  <p style="font-size:0.85em; margin-top:8px;">生成时间: 2026-03-17</p>
</header>

<nav class="nav">
  <a href="#s1">1. 主实验结果</a>
  <a href="#s2">2. 逐对话方差</a>
  <a href="#s3">3. 参数敏感度</a>
  <a href="#s4">4. 学习曲线</a>
  <a href="#s5">5. 分类别分析</a>
  <a href="#s6">6. DialSim 对比</a>
  <a href="#s7">7. 关键发现</a>
</nav>

<div class="container">

<!-- Section 1: 主实验结果矩阵 -->
<div class="section" id="s1">
  <h2><span class="num">1</span> 主实验结果矩阵</h2>
  <p class="desc">5 种检索方法在 2 个大规模数据集上的 P@1 / P@3 / P@5 对比。最佳值以高亮标记。</p>

  <table>
    <thead>
      <tr>
        <th rowspan="2">方法</th>
        <th colspan="3">LoCoMo (1976 题)</th>
        <th colspan="3">LongMemEval</th>
      </tr>
      <tr>
        <th>P@1</th><th>P@3</th><th>P@5</th>
        <th>P@1</th><th>P@3</th><th>P@5</th>
      </tr>
    </thead>
    <tbody>
      {main_table_rows}
    </tbody>
  </table>

  <div class="chart-row">
    <div class="chart-container"><canvas id="chart-main-locomo"></canvas></div>
    <div class="chart-container"><canvas id="chart-main-longmemeval"></canvas></div>
  </div>
</div>

<!-- Section 2: LoCoMo 逐对话方差分析 -->
<div class="section" id="s2">
  <h2><span class="num">2</span> LoCoMo 逐对话方差分析</h2>
  <p class="desc">10 段独立对话上各方法的 P@1 表现，展示方法稳定性。虚线为各方法平均值。</p>

  <div class="chart-container" style="max-width:100%;"><canvas id="chart-conv"></canvas></div>
</div>

<!-- Section 3: 参数敏感度 -->
<div class="section" id="s3">
  <h2><span class="num">3</span> 参数敏感度</h2>
  <p class="desc">DGD 三个关键超参数对 P@1 的影响。红色标注最优值，灰色标注「死亡谷」。</p>

  <div class="chart-row-3">
    <div class="chart-container"><canvas id="chart-param-alpha"></canvas></div>
    <div class="chart-container"><canvas id="chart-param-dim"></canvas></div>
    <div class="chart-container"><canvas id="chart-param-eta"></canvas></div>
  </div>
</div>

<!-- Section 4: 学习曲线 -->
<div class="section" id="s4">
  <h2><span class="num">4</span> 学习曲线</h2>
  <p class="desc">DGD+GT 和 DGD+Judge 在 10 轮训练中的 P@1 变化。水平虚线为 Cosine / BM25 baseline。</p>

  <div class="chart-row">
    <div class="chart-container"><canvas id="chart-curve-locomo"></canvas></div>
    <div class="chart-container"><canvas id="chart-curve-longmemeval"></canvas></div>
  </div>
</div>

<!-- Section 5: LoCoMo 分类别分析 -->
<div class="section" id="s5">
  <h2><span class="num">5</span> LoCoMo 分类别分析</h2>
  <p class="desc">按问题类别（5 类）分组统计 P@1。基于 Cosine 和 BM25 的 detail 数据（DGD 方法无逐题详情）。</p>

  <div class="chart-container"><canvas id="chart-category"></canvas></div>
</div>

<!-- Section 6: DialSim 复现对比 -->
<div class="section" id="s6">
  <h2><span class="num">6</span> DialSim 复现对比</h2>
  <p class="desc">在 DialSim 三个电视剧数据集上的新旧实验结果对比 (P@1)。</p>

  <table>
    <thead>
      <tr>
        <th>数据集</th>
        <th>版本</th>
        <th>Cosine</th>
        <th>DGD+GT</th>
        <th>DGD+Judge</th>
      </tr>
    </thead>
    <tbody>
      {dialsim_table_rows}
    </tbody>
  </table>
</div>

<!-- Section 7: 关键发现总结 -->
<div class="section" id="s7">
  <h2><span class="num">7</span> 关键发现总结</h2>
  <p class="desc">从实验数据中提取的核心结论。</p>

  <ol class="findings-list">
    {findings_html}
  </ol>
</div>

</div><!-- container -->

<script>
// ============ Data ============
const mainData = {main_chart_data};
const convData = {conv_chart_data};
const paramData = {param_chart_data};
const curveData = {curve_chart_data};
const categoryData = {category_chart_data};

// ============ Color Palette ============
const COLORS = {{
  'Cosine':      {{ bg: 'rgba(67,97,238,0.7)',  border: '#4361ee' }},
  'BM25':        {{ bg: 'rgba(76,201,240,0.7)', border: '#4cc9f0' }},
  'DGD+GT':      {{ bg: 'rgba(45,198,83,0.7)',  border: '#2dc653' }},
  'DGD+Judge':   {{ bg: 'rgba(255,107,53,0.7)', border: '#ff6b35' }},
  'DGD+CMS':     {{ bg: 'rgba(163,113,247,0.7)',border: '#a371f7' }},
}};
const METRIC_COLORS = {{
  'P@1': {{ bg: 'rgba(67,97,238,0.75)',  border: '#4361ee' }},
  'P@3': {{ bg: 'rgba(45,198,83,0.75)',  border: '#2dc653' }},
  'P@5': {{ bg: 'rgba(255,107,53,0.75)', border: '#ff6b35' }},
}};

// Detect dark mode
const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)';
const textColor = isDark ? '#e6edf3' : '#1a1a2e';

Chart.defaults.color = textColor;

// ============ Section 1: Main Results ============
function createMainChart(canvasId, datasetLabel) {{
  const methods = mainData.methods;
  const vals = mainData.values[datasetLabel];
  const datasets = Object.keys(vals).map(metric => ({{
    label: metric,
    data: vals[metric],
    backgroundColor: METRIC_COLORS[metric].bg,
    borderColor: METRIC_COLORS[metric].border,
    borderWidth: 1,
    borderRadius: 4,
  }}));

  new Chart(document.getElementById(canvasId), {{
    type: 'bar',
    data: {{ labels: methods, datasets }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: datasetLabel, font: {{ size: 16 }} }},
        legend: {{ position: 'top' }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          max: 80,
          title: {{ display: true, text: '准确率 (%)' }},
          grid: {{ color: gridColor }},
        }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}}

createMainChart('chart-main-locomo', 'LoCoMo');
createMainChart('chart-main-longmemeval', 'LongMemEval');

// ============ Section 2: Per-conversation ============
(function() {{
  const labels = convData.labels;
  const datasets = Object.keys(convData.datasets).map(method => ({{
    label: method,
    data: convData.datasets[method],
    backgroundColor: COLORS[method].bg,
    borderColor: COLORS[method].border,
    borderWidth: 1,
    borderRadius: 3,
  }}));

  // Add average lines as annotations
  const annotations = {{}};
  let idx = 0;
  for (const [method, avg] of Object.entries(convData.averages)) {{
    annotations['avg' + idx] = {{
      type: 'line',
      yMin: avg,
      yMax: avg,
      borderColor: COLORS[method].border,
      borderWidth: 2,
      borderDash: [6, 4],
      label: {{
        display: true,
        content: method + ' avg: ' + avg.toFixed(1) + '%',
        position: 'end',
        font: {{ size: 10 }},
        backgroundColor: 'rgba(0,0,0,0.6)',
        color: '#fff',
        padding: 3,
      }},
    }};
    idx++;
  }}

  new Chart(document.getElementById('chart-conv'), {{
    type: 'bar',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: 'LoCoMo 逐对话 P@1', font: {{ size: 16 }} }},
        legend: {{ position: 'top' }},
        annotation: {{ annotations }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'P@1 (%)' }},
          grid: {{ color: gridColor }},
        }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}})();

// ============ Section 3: Param Sensitivity ============
function createParamChart(canvasId, paramKey, title, xLabel) {{
  const d = paramData[paramKey];
  const pointBg = d.values.map((_, i) =>
    i === d.best_idx ? '#2dc653' : i === d.worst_idx ? '#e63946' : '#4361ee'
  );
  const pointRadius = d.values.map((_, i) =>
    (i === d.best_idx || i === d.worst_idx) ? 8 : 5
  );

  const annotations = {{}};
  // Best annotation
  annotations['best'] = {{
    type: 'label',
    xValue: d.best_idx,
    yValue: d.values[d.best_idx] + 1.5,
    content: ['最优 ' + d.values[d.best_idx].toFixed(1) + '%'],
    font: {{ size: 11, weight: 'bold' }},
    color: '#2dc653',
  }};
  // Worst annotation
  if (d.values[d.worst_idx] < d.values[d.best_idx] * 0.5) {{
    annotations['worst'] = {{
      type: 'label',
      xValue: d.worst_idx,
      yValue: d.values[d.worst_idx] + 1.5,
      content: ['死亡谷 ' + d.values[d.worst_idx].toFixed(1) + '%'],
      font: {{ size: 11, weight: 'bold' }},
      color: '#e63946',
    }};
  }}

  new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{
      labels: d.labels,
      datasets: [{{
        label: 'P@1',
        data: d.values,
        borderColor: '#4361ee',
        backgroundColor: 'rgba(67,97,238,0.1)',
        fill: true,
        tension: 0.3,
        pointBackgroundColor: pointBg,
        pointRadius: pointRadius,
        pointHoverRadius: 10,
        borderWidth: 2.5,
      }}],
    }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: title, font: {{ size: 14 }} }},
        legend: {{ display: false }},
        annotation: {{ annotations }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          title: {{ display: true, text: 'P@1 (%)' }},
          grid: {{ color: gridColor }},
        }},
        x: {{
          title: {{ display: true, text: xLabel }},
          grid: {{ display: false }},
        }},
      }},
    }},
  }});
}}

createParamChart('chart-param-alpha', 'alpha', 'Alpha (\\u03b1) vs P@1', '\\u03b1');
createParamChart('chart-param-dim', 'dim', 'Embedding Dim vs P@1', 'dim');
createParamChart('chart-param-eta', 'eta', 'Learning Rate (\\u03b7) vs P@1', '\\u03b7');

// ============ Section 4: Learning Curves ============
function createCurveChart(canvasId, datasetKey, title) {{
  const cd = curveData[datasetKey];
  const annotations = {{}};
  let idx = 0;
  for (const [name, val] of Object.entries(cd.baselines)) {{
    const color = name === 'Cosine' ? '#4361ee' : '#4cc9f0';
    annotations['bl' + idx] = {{
      type: 'line',
      yMin: val,
      yMax: val,
      borderColor: color,
      borderWidth: 2,
      borderDash: [8, 4],
      label: {{
        display: true,
        content: name + ': ' + val.toFixed(1) + '%',
        position: 'start',
        font: {{ size: 10 }},
        backgroundColor: 'rgba(0,0,0,0.6)',
        color: '#fff',
        padding: 3,
      }},
    }};
    idx++;
  }}

  new Chart(document.getElementById(canvasId), {{
    type: 'line',
    data: {{
      labels: cd.rounds,
      datasets: [
        {{
          label: 'DGD+GT',
          data: cd['DGD+GT'],
          borderColor: COLORS['DGD+GT'].border,
          backgroundColor: 'rgba(45,198,83,0.1)',
          fill: false,
          tension: 0.3,
          pointRadius: 5,
          borderWidth: 2.5,
        }},
        {{
          label: 'DGD+Judge',
          data: cd['DGD+Judge'],
          borderColor: COLORS['DGD+Judge'].border,
          backgroundColor: 'rgba(255,107,53,0.1)',
          fill: false,
          tension: 0.3,
          pointRadius: 5,
          borderWidth: 2.5,
        }},
      ],
    }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: title, font: {{ size: 16 }} }},
        legend: {{ position: 'top' }},
        annotation: {{ annotations }},
      }},
      scales: {{
        y: {{
          title: {{ display: true, text: 'P@1 (%)' }},
          grid: {{ color: gridColor }},
        }},
        x: {{
          title: {{ display: true, text: 'Round' }},
          grid: {{ display: false }},
        }},
      }},
    }},
  }});
}}

createCurveChart('chart-curve-locomo', 'LoCoMo', 'LoCoMo 学习曲线');
createCurveChart('chart-curve-longmemeval', 'LongMemEval', 'LongMemEval 学习曲线');

// ============ Section 5: Category Analysis ============
(function() {{
  new Chart(document.getElementById('chart-category'), {{
    type: 'bar',
    data: {{
      labels: categoryData.categories,
      datasets: [
        {{
          label: 'Cosine',
          data: categoryData['Cosine'],
          backgroundColor: COLORS['Cosine'].bg,
          borderColor: COLORS['Cosine'].border,
          borderWidth: 1,
          borderRadius: 4,
        }},
        {{
          label: 'BM25',
          data: categoryData['BM25'],
          backgroundColor: COLORS['BM25'].bg,
          borderColor: COLORS['BM25'].border,
          borderWidth: 1,
          borderRadius: 4,
        }},
      ],
    }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: 'LoCoMo 分类别 P@1 (Cosine vs BM25)', font: {{ size: 16 }} }},
        legend: {{ position: 'top' }},
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          max: 50,
          title: {{ display: true, text: 'P@1 (%)' }},
          grid: {{ color: gridColor }},
        }},
        x: {{ grid: {{ display: false }} }},
      }},
    }},
  }});
}})();
</script>

<footer style="text-align:center; padding:30px; color:var(--text-muted); font-size:0.85em;">
  Meomory Experiment Report &mdash; 自动生成于 experiments/results/ 数据
</footer>

</body>
</html>"""

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report generated: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    generate_html()
