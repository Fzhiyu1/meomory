# FunSearch v6: 在线流式评测 + 多架构 seed Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 改进化评测为在线流式（先测再学），加入 CMS/原始DGD/gen059/gen057 多架构 seed，进化出能泛化的在线学习算法。

**Architecture:** 改 evaluator.py 评测循环为"每题先评分再学习"模式，改 specification.py 加入 5 种 seed（原始 DGD、CMS 三层、gen059 双通道、gen057 单通道、空白初始），改 runner.py 移除多轮参数。5 个岛分别种不同 seed。

**Tech Stack:** Python, numpy, FunSearch 群岛进化框架

**额外改动：** evaluator 只传 dim 给算法，不传 alpha/eta——让进化自己决定所有超参。

---

### Task 1: 改 evaluator 为在线流式评测

**核心改动：从"先学再测"改为"先测再学"。**

**Files:**
- Modify: `src/funsearch/evaluator.py:204-233`

**Step 1: 修改评测循环**

当前代码（第 204-233 行）：
```python
for r in range(1, n_rounds + 1):
    # GT feedback — 先全部 update
    for qi, q in enumerate(questions):
        _update(qi, q)
    # 再全部评测
    for qi, q in enumerate(questions):
        # ... evaluate ...
```

改为：
```python
for r in range(1, n_rounds + 1):
    hits = {1: 0, 3: 0, 5: 0}
    per_question_hit = [False] * len(questions)
    for qi, q in enumerate(questions):
        # 1. 先评测（DGD 还没见过这题的答案）
        answer_set = set(q["answer_indices"])
        top_ids = _query_top5(qi)
        for k in [1, 3, 5]:
            if any(idx in answer_set for idx in top_ids[:k]):
                hits[k] += 1
        if top_ids and top_ids[0] in answer_set:
            per_question_hit[qi] = True

        # 2. 再学习（GT/noisy 反馈）
        try:
            _update(qi, q)
        except Exception:
            return {
                "p_at_1": 0.0, "error": True, "elapsed": time.time() - t0,
                "scores_per_test": {"overall": 0.0},
            }

    total = len(questions)
    metrics = {k: hits[k] / total for k in hits}
    rounds.append({
        "round": r, "p_at_1": metrics[1],
        "p_at_3": metrics[3], "p_at_5": metrics[5],
    })
```

**Step 2: 验证语法**

Run: `PYTHONPATH=. .venv/bin/python -c "import ast; ast.parse(open('src/funsearch/evaluator.py').read()); print('OK')"`
Expected: OK

**Step 3: 快速冒烟测试**

Run:
```bash
PYTHONPATH=. .venv/bin/python -c "
from src.funsearch.sandbox import compile_class
from src.funsearch.evaluator import evaluate_update_fn
import json
b = json.load(open('experiments/funsearch-v4/best.json'))
cls = compile_class(b[0]['code'])
r = evaluate_update_fn(cls, dim=256, n_rounds=1, max_questions=50)
print(f'Online P@1 (50q): {r[\"p_at_1\"]:.1%}')
"
```
Expected: 一个比之前低的分数（因为先测再学）

**Step 4: Commit**

```bash
git add src/funsearch/evaluator.py
git commit -m "feat(eval): online streaming evaluation — test before learn"
```

---

### Task 2: 准备 5 种 seed

创建 seed 文件，每种架构一个独立实现，分配到不同岛。

**Files:**
- Create: `src/funsearch/seeds.py`

**Step 1: 写 seed 文件**

```python
"""v6 seed 集合：5 种不同架构。

岛 0: 原始 DGD（最简单）
岛 1: CMS 三层架构（抗干扰）
岛 2: gen059 双通道（当前最强精度）
岛 3: gen057 单通道（抗噪王）
岛 4: 空白初始实现（自由探索）
"""

# 岛 0: 原始 DGD — 最简单的关联记忆
SEED_ORIGINAL_DGD = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01):
        self.dim = dim
        self.alpha = alpha
        self.eta = eta
        self.M = [[0.0]*dim for _ in range(dim)]

    def query(self, key):
        dim = self.dim
        out = [0.0]*dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += self.M[i][j] * key[j]
            out[i] = s
        return out

    def update(self, key, target):
        dim = self.dim
        act = self.query(key)
        for i in range(dim):
            err = target[i] - act[i]
            for j in range(dim):
                self.M[i][j] = self.alpha * self.M[i][j] + self.eta * err * key[j]
'''

# 岛 1: CMS 三层架构 — 多时间尺度，天然抗干扰
SEED_CMS = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01):
        self.dim = dim
        self.eta = eta
        # 三层：高频（每次更新）、中频（每5次）、低频（每20次）
        self.M_high = [[0.0]*dim for _ in range(dim)]
        self.M_mid = [[0.0]*dim for _ in range(dim)]
        self.M_low = [[0.0]*dim for _ in range(dim)]
        self.step = 0

    def _matvec(self, M, v):
        dim = self.dim
        out = [0.0]*dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += M[i][j] * v[j]
            out[i] = s
        return out

    def query(self, key):
        dim = self.dim
        # 串联：高频 → 中频 → 低频
        act = key[:]
        for M, w in [(self.M_high, 0.5), (self.M_mid, 0.3), (self.M_low, 0.2)]:
            layer_act = self._matvec(M, act)
            act = [w * la + (1-w) * a for la, a in zip(layer_act, act)]
        return act

    def update(self, key, target):
        dim = self.dim
        self.step += 1
        act = self.query(key)

        def _update_layer(M, key, target, act):
            for i in range(dim):
                err = target[i] - act[i]
                for j in range(dim):
                    M[i][j] += self.eta * err * key[j]

        # 高频：每次都更新
        _update_layer(self.M_high, key, target, act)
        # 中频：每5次更新
        if self.step % 5 == 0:
            _update_layer(self.M_mid, key, target, act)
        # 低频：每20次更新
        if self.step % 20 == 0:
            _update_layer(self.M_low, key, target, act)
'''

# 岛 2: gen059 双通道（从 old_population 提取，代码太长，运行时从文件加载）
SEED_GEN059 = "LOAD_FROM_FILE:gen059"

# 岛 3: gen057 单通道（从 old_population 提取，运行时从文件加载）
SEED_GEN057 = "LOAD_FROM_FILE:gen057"

# 岛 4: 空白初始实现（最小可运行）
SEED_BLANK = '''
class AssociativeMemory:
    def __init__(self, dim, alpha=1.0, eta=0.01):
        self.dim = dim
        self.eta = eta
        self.M = [[1.0 if i==j else 0.0 for j in range(dim)] for i in range(dim)]

    def query(self, key):
        dim = self.dim
        out = [0.0]*dim
        for i in range(dim):
            s = 0.0
            for j in range(dim):
                s += self.M[i][j] * key[j]
            out[i] = s
        return out

    def update(self, key, target):
        dim = self.dim
        act = self.query(key)
        for i in range(dim):
            err = target[i] - act[i]
            for j in range(dim):
                self.M[i][j] += self.eta * err * key[j]
'''

ALL_SEEDS = [
    ("original_dgd", SEED_ORIGINAL_DGD),
    ("cms_3layer", SEED_CMS),
    ("gen059", SEED_GEN059),
    ("gen057", SEED_GEN057),
    ("blank_identity", SEED_BLANK),
]
```

**Step 2: 验证 seed 可编译**

Run:
```bash
PYTHONPATH=. .venv/bin/python -c "
from src.funsearch.sandbox import compile_class
from src.funsearch.seeds import ALL_SEEDS
for name, code in ALL_SEEDS:
    if code.startswith('LOAD_FROM_FILE'):
        print(f'{name}: skip (file load)')
        continue
    cls = compile_class(code.strip())
    if cls:
        obj = cls(dim=8, alpha=1.0, eta=0.01)
        q = obj.query([0.1]*8)
        obj.update([0.1]*8, [0.2]*8)
        print(f'{name}: OK')
    else:
        print(f'{name}: COMPILE FAILED')
"
```
Expected: original_dgd: OK, cms_3layer: OK, blank_identity: OK

**Step 3: Commit**

```bash
git add src/funsearch/seeds.py
git commit -m "feat: 5 architecture seeds for v6 — DGD/CMS/gen059/gen057/blank"
```

---

### Task 3: 改 runner 支持多架构 seed + 在线模式

**Files:**
- Modify: `src/funsearch/runner.py`
- Modify: `scripts/run_funsearch.py`

**Step 1: 修改 runner seed 加载逻辑**

在 `run_funsearch()` 函数中，当 `seed_from` 指向 seeds.py 时，每个 seed 分配到对应的岛：

替换现有 seed 加载块，支持两种模式：
- `--seed-from seeds` → 从 seeds.py 加载 5 种架构
- `--seed-from <file.json>` → 兼容旧模式（从 JSON 加载）

seed 加载完后，每个 seed 用在线流式评测打分（已有的 `_eval_mixed_in_subprocess`，现在自动是先测再学）。

**Step 2: 设置 n_rounds 默认值为 1**

在线模式下 1 轮就够。在 `run_funsearch()` 参数中：
```python
eval_rounds: int = 1,  # v6: 在线流式只需 1 轮
```

**Step 3: 在 run_funsearch.py CLI 中添加 `--seed-mode` 参数**

```python
parser.add_argument("--seed-mode", default="file",
    choices=["file", "multi-arch"],
    help="file: load from JSON; multi-arch: use 5 architecture seeds")
```

**Step 4: 验证**

Run: `PYTHONPATH=. .venv/bin/python -c "import ast; ast.parse(open('src/funsearch/runner.py').read()); print('OK')"`

**Step 5: Commit**

```bash
git add src/funsearch/runner.py scripts/run_funsearch.py
git commit -m "feat(runner): v6 multi-arch seed mode + online eval defaults"
```

---

### Task 4: 本机快速验证

在本机用少量数据验证整个链路（在线评测 + 多架构 seed + 混合 fitness）。

**Files:**
- Create: `scripts/verify_v6.py`

**Step 1: 写验证脚本**

```python
"""v6 快速验证：50 题在线评测，3 个 seed。"""
import json
from src.funsearch.sandbox import compile_class
from src.funsearch.evaluator import evaluate_update_fn
from src.funsearch.seeds import ALL_SEEDS

for name, code in ALL_SEEDS:
    if code.startswith("LOAD_FROM_FILE"):
        # Load gen059/gen057 from best.json
        best = json.load(open("experiments/funsearch-v4/best.json"))
        target = "gen059" if "gen059" in code else "gen057"
        match = [p for p in best if target in p["id"]]
        if not match:
            # Try old_population
            pop = json.load(open("experiments/funsearch-v4/old_population.json"))
            match = [p for p in pop["all_programs"] if target in p["id"]]
        code = match[0]["code"] if match else ALL_SEEDS[-1][1]  # fallback to blank

    cls = compile_class(code.strip())
    if cls is None:
        print(f"{name}: COMPILE FAILED")
        continue

    # 在线评测 50 题
    r = evaluate_update_fn(cls, dim=256, n_rounds=1, max_questions=50)
    print(f"{name}: online P@1={r['p_at_1']:.1%} (50q)")
```

**Step 2: Run**

Run: `PYTHONPATH=. .venv/bin/python scripts/verify_v6.py`
Expected: 5 个 seed 都出分数，都低于之前的 train=test 分数

**Step 3: Commit**

```bash
git add scripts/verify_v6.py
git commit -m "test: v6 quick verification script"
```

---

### Task 5: 部署 v6 到 5070

停掉 v5，推代码，启动 v6。

**Files:**
- No new files, deployment only

**Step 1: 停掉 v5**

```bash
ssh -p 2222 ubuntu@100.94.126.19 'tmux kill-session -t v5'
```

**Step 2: 推代码到 5070**

```bash
rsync -az --exclude '.venv' --exclude '__pycache__' --exclude 'experiments' --exclude '.git' --exclude 'data' \
  -e "ssh -p 2222" \
  /Users/fangzhiyu/run/meomory/src/ ubuntu@100.94.126.19:~/meomory/src/

rsync -az -e "ssh -p 2222" \
  /Users/fangzhiyu/run/meomory/scripts/run_funsearch.py \
  ubuntu@100.94.126.19:~/meomory/scripts/run_funsearch.py
```

别忘了在 5070 上加 `from __future__ import annotations`。

**Step 3: 准备 seed 文件**

在 5070 上提取 gen059/gen057 代码生成 seeds.json：
```bash
ssh -p 2222 ubuntu@100.94.126.19 'cd ~/meomory && .venv/bin/python -c "
import json
from src.funsearch.seeds import ALL_SEEDS

pop = json.load(open(\"experiments/funsearch-v4/old_population.json\"))
progs = {p[\"id\"]: p for p in pop[\"all_programs\"]}

seeds = []
for name, code in ALL_SEEDS:
    if code.startswith(\"LOAD_FROM_FILE\"):
        target = code.split(\":\")[1]
        match = [p for pid, p in progs.items() if target in pid]
        if match:
            seeds.append(match[0])
            continue
    seeds.append({\"id\": name, \"code\": code.strip(), \"score\": 0, \"scores_per_test\": {}})

with open(\"experiments/funsearch-v6/seeds.json\", \"w\") as f:
    json.dump(seeds, f, indent=2)
print(f\"Saved {len(seeds)} seeds\")
"'
```

**Step 4: 启动 v6**

```bash
ssh -p 2222 ubuntu@100.94.126.19 'cd ~/meomory && \
  mkdir -p experiments/funsearch-v6 && \
  tmux new-session -d -s v6 "cd ~/meomory && \
    MEOMORY_LLM_KEY=${MEOMORY_LLM_KEY} \
    .venv/bin/python -u scripts/run_funsearch.py \
      --ollama-host http://localhost:11434 \
      --model-weights 50,50,0 \
      --iterations 0 \
      --rounds 1 \
      --samples 6 \
      --max-questions 9999 \
      --data-dir experiments/funsearch-v6 \
      --seed-from experiments/funsearch-v6/seeds.json \
    2>&1 | tee experiments/funsearch-v6.log"'
```

**Step 5: 验证启动**

```bash
sleep 15 && ssh -p 2222 ubuntu@100.94.126.19 'tail -10 ~/meomory/experiments/funsearch-v6.log'
```

Expected: 看到 5 个 seed 加载 + "Starting FunSearch v6" + 在线评测模式

**Step 6: 更新仪表盘**

本地 dashboard.py 里把 `funsearch-v5` 改为 `funsearch-v6`。

**Step 7: Commit + push**

```bash
git add -A
git commit -m "feat: FunSearch v6 — online streaming eval + multi-arch seeds"
git push
```

---

## 总结

| Task | 内容 | 风险 |
|------|------|------|
| 1 | evaluator 先测再学 | 核心改动，需验证 |
| 2 | 5 种架构 seed | CMS 需要验证能编译 |
| 3 | runner 多架构 seed 模式 | 兼容旧模式 |
| 4 | 本机验证 | 确认链路通 |
| 5 | 部署 5070 | 停 v5 + 启 v6 |

**预期：**
- 在线评测下 gen059 可能只有 25-35%（不再是 56.6%）
- CMS 可能表现更好（多层结构天然抗干扰）
- 进化会发现新的算法结构来解决泛化问题
- 这是一个全新的进化方向——从"记忆能力"转向"在线学习能力"
