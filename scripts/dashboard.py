#!/usr/bin/env python3
"""FunSearch 进化仪表盘 — 实时监控进化进度。

用法:
    .venv/bin/python scripts/dashboard.py
    然后打开 http://localhost:8765
"""
import json
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = 8765
SSH_CMD = "ssh -p 2222 -o ConnectTimeout=5 ubuntu@100.94.126.19"


def _build_response(pop, hist, best):
    """从原始 population/history/best 数据构建 API 响应。"""
    progs = pop.get("all_programs", [])

    # --- 岛屿状态 ---
    island_map = {}  # island_id -> {programs, best_score, best_id, clusters}
    for p in progs:
        iid = p.get("island_id", 0)
        if iid not in island_map:
            island_map[iid] = {"programs": [], "best_score": -1, "best_id": "", "signatures": set()}
        island_map[iid]["programs"].append(p)
        if p["score"] > island_map[iid]["best_score"]:
            island_map[iid]["best_score"] = p["score"]
            island_map[iid]["best_id"] = p["id"]
        # 用 scores_per_test 的值元组作为 signature
        spt = p.get("scores_per_test", {})
        if spt:
            sig = tuple(spt[k] for k in sorted(spt.keys()))
            island_map[iid]["signatures"].add(sig)

    islands = []
    for iid in sorted(island_map.keys()):
        info = island_map[iid]
        islands.append({
            "id": iid,
            "best_score": info["best_score"],
            "best_id": info["best_id"],
            "n_programs": len(info["programs"]),
            "n_clusters": len(info["signatures"]),
        })

    # --- 模型统计（从全量程序聚合）---
    model_stats = {}
    for p in progs:
        m = p.get("eval_details", {}).get("model", "seed")
        if m not in model_stats:
            model_stats[m] = {"count": 0, "best": 0}
        model_stats[m]["count"] += 1
        model_stats[m]["best"] = max(model_stats[m]["best"], p["score"])

    # --- Top 10（含 code）---
    sorted_progs = sorted(progs, key=lambda x: x["score"], reverse=True)[:10]
    top10 = [{
        "id": p["id"],
        "score": p["score"],
        "model": p.get("eval_details", {}).get("model", "seed"),
        "code": p.get("code", ""),
    } for p in sorted_progs]

    return {
        "total": len(progs),
        "best_score": best[0]["score"] if best else 0,
        "best_id": best[0]["id"] if best else "",
        "best_model": best[0].get("eval_details", {}).get("model", "?") if best else "",
        "best_code": best[0]["code"] if best else "",
        "history": hist[-20:],
        "top10": top10,
        "islands": islands,
        "models": model_stats,
    }


def fetch_5070_data():
    """从 5070 获取最新进化数据。"""
    # 先尝试 SSH 拉原始 JSON，然后本地构建响应
    try:
        result = subprocess.run(
            f'{SSH_CMD} "cat ~/meomory/experiments/funsearch-v4/population.json"',
            shell=True, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            pop = json.loads(result.stdout.strip())
            # 也拉 history 和 best
            hist_result = subprocess.run(
                f'{SSH_CMD} "cat ~/meomory/experiments/funsearch-v4/history.json"',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            hist = json.loads(hist_result.stdout.strip()) if hist_result.returncode == 0 else []
            best_result = subprocess.run(
                f'{SSH_CMD} "cat ~/meomory/experiments/funsearch-v4/best.json"',
                shell=True, capture_output=True, text=True, timeout=10,
            )
            best = json.loads(best_result.stdout.strip()) if best_result.returncode == 0 else []
            return json.dumps(_build_response(pop, hist, best))
    except Exception:
        pass

    # 后备：后台 rsync + 本地文件
    import threading
    def _rsync_bg():
        try:
            subprocess.run(
                'rsync -az -e "ssh -p 2222 -o ConnectTimeout=10" '
                'ubuntu@100.94.126.19:~/meomory/experiments/funsearch-v4/{population.json,history.json,best.json} '
                'experiments/funsearch-v4/ 2>/dev/null',
                shell=True, timeout=20,
            )
        except Exception:
            pass
    threading.Thread(target=_rsync_bg, daemon=True).start()

    try:
        for data_dir in ["experiments/funsearch-v4", "experiments/funsearch-v2", "experiments/funsearch"]:
            pop_file = Path(data_dir) / "population.json"
            if pop_file.exists():
                pop = json.loads(pop_file.read_text())
                hist_file = Path(data_dir) / "history.json"
                hist = json.loads(hist_file.read_text()) if hist_file.exists() else []
                best_file = Path(data_dir) / "best.json"
                best = json.loads(best_file.read_text()) if best_file.exists() else []
                resp = _build_response(pop, hist, best)
                resp["source"] = "local"
                return json.dumps(resp)
    except Exception:
        pass
    return json.dumps({"error": "No data available"})


DASHBOARD_HTML = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>meomory 自进化仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 5px; }
.subtitle { color: #8b949e; margin-bottom: 20px; font-size: 14px; }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
.card h2 { color: #58a6ff; font-size: 14px; margin-bottom: 10px; }
.big-number { font-size: 48px; font-weight: bold; color: #39d353; }
.big-number.blue { color: #58a6ff; }
.label { color: #8b949e; font-size: 12px; }
.top-list { list-style: none; }
.top-list li {
  padding: 8px 10px; border-bottom: 1px solid #21262d;
  display: flex; justify-content: space-between; align-items: center;
  cursor: pointer; border-radius: 4px; transition: background 0.15s;
}
.top-list li:hover { background: #1c2333; }
.top-list li.selected { background: #1f3a5f; border-left: 3px solid #58a6ff; }
.top-list .score { color: #39d353; font-weight: bold; font-size: 14px; }
.top-list .rank { color: #8b949e; font-size: 12px; margin-right: 8px; min-width: 40px; }
.top-list .prog-id { font-size: 13px; color: #c9d1d9; }
.top-list .model { font-size: 11px; margin-left: 6px; padding: 1px 6px; border-radius: 3px; }
.model-9b { color: #f0883e; background: rgba(240,136,62,0.1); }
.model-gpt54 { color: #bc8cff; background: rgba(188,140,255,0.1); }
.model-deepseek { color: #58a6ff; background: rgba(88,166,255,0.1); }
.model-seed { color: #8b949e; background: rgba(139,148,158,0.1); }
.code-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.code-header .prog-label { color: #39d353; font-size: 15px; font-weight: bold; }
.code-header .prog-score { color: #58a6ff; font-size: 14px; }
pre {
  background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
  padding: 15px; overflow: auto; font-size: 13px; color: #c9d1d9;
  max-height: 500px; line-height: 1.5;
}
.chart-container { height: 250px; }
.donut-container { height: 200px; display: flex; justify-content: center; align-items: center; }
.status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.status.live { background: #39d353; animation: pulse 2s infinite; }
.status.stale { background: #d29922; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.refresh-info { color: #8b949e; font-size: 12px; float: right; }

/* 岛屿状态 */
.islands-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
.island-block {
  border-radius: 8px; padding: 12px; text-align: center;
  border: 1px solid rgba(255,255,255,0.1);
}
.island-block .island-id { font-size: 12px; font-weight: bold; margin-bottom: 4px; opacity: 0.9; }
.island-block .island-score { font-size: 20px; font-weight: bold; }
.island-block .island-meta { font-size: 11px; margin-top: 4px; opacity: 0.8; }
</style>
</head>
<body>
<h1>meomory 自进化仪表盘</h1>
<p class="subtitle"><span class="status live" id="statusDot"></span>
<span id="statusText">连接中...</span>
<span class="refresh-info">每 30 秒自动刷新</span></p>

<!-- 大数字卡片 -->
<div class="grid">
  <div class="card">
    <h2>最优 P@1</h2>
    <div class="big-number" id="bestScore">--</div>
    <div class="label" id="bestInfo">--</div>
  </div>
  <div class="card">
    <h2>已评测程序数</h2>
    <div class="big-number blue" id="totalPrograms">--</div>
    <div class="label">进化产出的候选算法</div>
  </div>
  <div class="card">
    <h2>模型贡献</h2>
    <div class="donut-container"><canvas id="modelChart"></canvas></div>
  </div>
</div>

<!-- 岛屿状态 -->
<div class="card" style="margin-bottom: 20px;">
  <h2>岛屿状态</h2>
  <div class="islands-grid" id="islandsGrid"></div>
</div>

<!-- 进化历史 + 排行榜 -->
<div class="grid-2">
  <div class="card">
    <h2>进化历史</h2>
    <div class="chart-container"><canvas id="historyChart"></canvas></div>
  </div>
  <div class="card">
    <h2>排行榜 Top 10</h2>
    <ul class="top-list" id="topList"></ul>
  </div>
</div>

<!-- 代码面板 -->
<div class="card" style="margin-top: 15px;">
  <h2>最优算法代码</h2>
  <div class="code-header">
    <span class="prog-label" id="codeLabel">--</span>
    <span class="prog-score" id="codeScore">--</span>
  </div>
  <pre id="bestCode">加载中...</pre>
</div>

<script>
let historyChart = null;
let modelChart = null;
let currentData = null;
let selectedIndex = 0;  // 默认选中第 1 名

const ISLAND_COLORS = [
  { bg: 'rgba(57, 211, 83, 0.15)', border: '#39d353', text: '#39d353' },
  { bg: 'rgba(88, 166, 255, 0.15)', border: '#58a6ff', text: '#58a6ff' },
  { bg: 'rgba(188, 140, 255, 0.15)', border: '#bc8cff', text: '#bc8cff' },
  { bg: 'rgba(240, 136, 62, 0.15)', border: '#f0883e', text: '#f0883e' },
  { bg: 'rgba(210, 153, 34, 0.15)', border: '#d29922', text: '#d29922' },
];

const MODEL_COLORS = {
  'seed':     '#8b949e',
  '9b':       '#f0883e',
  'deepseek': '#58a6ff',
  'gpt54':    '#bc8cff',
};

function getModelColor(model) {
  return MODEL_COLORS[model] || '#c9d1d9';
}

async function fetchData() {
  try {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    if (data.error) {
      document.getElementById('statusText').textContent = '错误: ' + data.error;
      document.getElementById('statusDot').className = 'status stale';
      return;
    }
    currentData = data;
    updateDashboard(data);
    document.getElementById('statusText').textContent = '运行中 — ' + new Date().toLocaleTimeString();
    document.getElementById('statusDot').className = 'status live';
  } catch(e) {
    document.getElementById('statusText').textContent = '连接失败';
    document.getElementById('statusDot').className = 'status stale';
  }
}

function selectProgram(index) {
  selectedIndex = index;
  if (!currentData || !currentData.top10 || !currentData.top10[index]) return;
  const p = currentData.top10[index];
  document.getElementById('codeLabel').textContent = '#' + (index + 1) + ' ' + p.id;
  document.getElementById('codeScore').textContent = 'P@1: ' + (p.score * 100).toFixed(1) + '%';
  document.getElementById('bestCode').textContent = p.code || 'N/A';
  // 更新选中高亮
  document.querySelectorAll('.top-list li').forEach((li, i) => {
    li.classList.toggle('selected', i === index);
  });
}

function updateDashboard(data) {
  // -- 大数字 --
  document.getElementById('bestScore').textContent = (data.best_score * 100).toFixed(1) + '%';
  document.getElementById('bestInfo').textContent = data.best_id + ' (model: ' + data.best_model + ')';
  document.getElementById('totalPrograms').textContent = data.total;

  // -- Top 10 --
  const list = document.getElementById('topList');
  list.innerHTML = '';
  (data.top10 || []).forEach((p, i) => {
    const li = document.createElement('li');
    const modelClass = 'model model-' + (p.model || 'seed');
    li.innerHTML =
      '<span><span class="rank">#' + (i+1) + '</span>' +
      '<span class="prog-id">' + p.id + '</span>' +
      '<span class="' + modelClass + '">' + (p.model || 'seed') + '</span></span>' +
      '<span class="score">' + (p.score * 100).toFixed(1) + '%</span>';
    li.addEventListener('click', () => selectProgram(i));
    if (i === selectedIndex) li.classList.add('selected');
    list.appendChild(li);
  });

  // -- 默认显示选中程序的代码 --
  if (data.top10 && data.top10.length > 0) {
    const idx = Math.min(selectedIndex, data.top10.length - 1);
    selectProgram(idx);
  }

  // -- 岛屿状态 --
  const islandsGrid = document.getElementById('islandsGrid');
  islandsGrid.innerHTML = '';
  (data.islands || []).forEach((isl, i) => {
    const color = ISLAND_COLORS[i % ISLAND_COLORS.length];
    const block = document.createElement('div');
    block.className = 'island-block';
    block.style.background = color.bg;
    block.style.borderColor = color.border;
    block.innerHTML =
      '<div class="island-id" style="color:' + color.text + '">岛屿 ' + isl.id + '</div>' +
      '<div class="island-score" style="color:' + color.text + '">' + (isl.best_score * 100).toFixed(1) + '%</div>' +
      '<div class="island-meta">' + isl.n_programs + ' 个程序 / ' + isl.n_clusters + ' 个簇</div>';
    islandsGrid.appendChild(block);
  });

  // -- 模型贡献饼图 --
  const models = data.models || {};
  const modelLabels = Object.keys(models);
  const modelCounts = modelLabels.map(m => models[m].count);
  const modelColors = modelLabels.map(m => getModelColor(m));

  if (modelChart) modelChart.destroy();
  modelChart = new Chart(document.getElementById('modelChart'), {
    type: 'doughnut',
    data: {
      labels: modelLabels,
      datasets: [{
        data: modelCounts,
        backgroundColor: modelColors,
        borderColor: '#161b22',
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '55%',
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#c9d1d9', font: { size: 11 }, padding: 8, usePointStyle: true, pointStyleWidth: 8 }
        },
        tooltip: {
          callbacks: {
            label: function(ctx) {
              const m = ctx.label;
              const info = models[m];
              return m + ': ' + info.count + ' 个, 最优 ' + (info.best * 100).toFixed(1) + '%';
            }
          }
        }
      }
    }
  });

  // -- 进化历史折线图 --
  const hist = data.history || [];
  if (hist.length > 0) {
    const labels = hist.map(h => h.total_programs || h.register_count || '');
    const bestData = hist.map(h => (h.global_best || 0) * 100);
    const avgData = hist.map(h => (h.global_avg || 0) * 100);

    if (historyChart) historyChart.destroy();
    historyChart = new Chart(document.getElementById('historyChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: '最优 P@1', data: bestData, borderColor: '#39d353', backgroundColor: 'rgba(57,211,83,0.1)', tension: 0.3, fill: true, pointRadius: 2 },
          { label: '平均 P@1', data: avgData, borderColor: '#8b949e', tension: 0.3, fill: false, pointRadius: 2 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 100, grid: { color: '#21262d' }, ticks: { color: '#8b949e' } },
          x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' } }
        },
        plugins: { legend: { labels: { color: '#c9d1d9' } } }
      }
    });
  }
}

fetchData();
setInterval(fetchData, 30000);
</script>
</body>
</html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/data':
            data = fetch_5070_data()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())

    def log_message(self, format, *args):
        pass  # Suppress logs


if __name__ == '__main__':
    print(f"Dashboard running at http://localhost:{PORT}")
    print(f"Fetching data from 5070 via SSH...")
    HTTPServer(('', PORT), Handler).serve_forever()
