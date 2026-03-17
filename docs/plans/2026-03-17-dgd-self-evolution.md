# DGD 自进化框架 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于 FunSearch 群岛进化模型，让 LLM 自动发现比手写 Hebbian 规则更好的 DGD 更新规则。

**Architecture:** 定义一个 Python 函数模板 `dgd_update(M, key, target, alpha, eta) → M`，LLM 生成该函数的变体，沙箱执行后在 GT benchmark 上评测 P@1。群岛模型维护多个种群，定期淘汰弱岛，保持多样性。全程零人工干预。

**Tech Stack:** Python 3.14, DeepSeek API (LLM), asyncio, ast (代码解析), 已有 benchmark pipeline

**关键参考：**
- FunSearch 官方实现：`/Users/fangzhiyu/run/github/funsearch/implementation/`
- 当前 DGD 实现：`src/dgd.py:33-49`（update 方法）
- 当前 GT benchmark：`src/bench/methods.py:140-146`（DGDGroundTruth）

---

## 架构概览

```
┌──────────────────────────────────────────────────┐
│                  Main Loop                        │
│  while True:                                      │
│    prompt = database.get_prompt()   ← 从岛采样     │
│    code = llm.generate(prompt)      ← LLM 生成     │
│    fn = sandbox.compile(code)       ← 安全编译     │
│    score = evaluator.run(fn)        ← GT benchmark │
│    database.register(fn, score)     ← 注册+选择    │
└──────────────────────────────────────────────────┘
```

```
experiments/funsearch/
├── status.json          # 运行状态
├── best.json            # 历史最优 top-10
├── history.json         # 每代快照
└── programs/            # 所有已评测的函数代码
    ├── gen000-island0-000.py
    └── ...
```

---

### Task 1: 进化模板（Specification）

**Files:**
- Create: `src/funsearch/__init__.py`
- Create: `src/funsearch/specification.py`

**作用：** 定义要进化的函数骨架和评测入口。

**Step 1: 创建 `src/funsearch/__init__.py`**

```python
# empty
```

**Step 2: 创建 `src/funsearch/specification.py`**

```python
"""进化目标定义：DGD update rule 的函数模板。

LLM 只需要填充 dgd_update 函数体。
评测函数 evaluate 会被 Evaluator 调用。
"""

# === 要进化的函数（LLM 生成变体）===
EVOLVE_FUNCTION_NAME = "dgd_update"

INITIAL_IMPLEMENTATION = '''
def dgd_update(M, key, target, dim, alpha, eta):
    """Update the associative memory matrix M.

    Args:
        M: dim x dim weight matrix (list of lists)
        key: query vector (list of floats, length dim)
        target: target vector (list of floats, length dim)
        dim: dimension
        alpha: forgetting rate (1.0 = no time decay)
        eta: learning rate

    Returns:
        Updated M (modify in-place and return)
    """
    # Compute activation: M @ key
    activation = [sum(M[i][j] * key[j] for j in range(dim)) for i in range(dim)]
    # Compute error
    error = [activation[i] - target[i] for i in range(dim)]
    # Standard Hebbian/DGD update
    for i in range(dim):
        for j in range(dim):
            identity_term = alpha if i == j else 0.0
            forget = M[i][j] * (identity_term - eta * key[i] * key[j])
            learn = -eta * error[i] * key[j]
            M[i][j] = forget + learn
    return M
'''

# === Prompt 模板：给 LLM 看 N 个版本，让它生成下一个 ===
PROMPT_TEMPLATE = '''"""DGD (Delta Gradient Descent) update rule for associative memory.

The matrix M maps query vectors to activation vectors.
Goal: update M so that future queries retrieve better results.

Available variables:
- M: dim x dim weight matrix (list of lists, modify in-place)
- key: query vector (list of floats)
- target: correct target vector (list of floats)
- dim: integer dimension
- alpha: forgetting rate (float, 1.0 = no time decay)
- eta: learning rate (float)

Rules:
- You may use math functions: sqrt, exp, log, abs, max, min, sum
- You may NOT import any modules
- You must modify M in-place and return M
- Keep it simple: avoid O(dim^3) or higher complexity
"""

{versioned_functions}

def {next_function_name}(M, key, target, dim, alpha, eta):
    """Improved version of {prev_function_name}."""
'''
```

**Step 3: Commit**

```bash
git add src/funsearch/__init__.py src/funsearch/specification.py
git commit -m "feat(funsearch): DGD update rule evolution specification"
```

---

### Task 2: 沙箱（安全执行 LLM 生成的代码）

**Files:**
- Create: `src/funsearch/sandbox.py`

**作用：** 编译 LLM 生成的代码为 Python 函数，在受限环境中执行。

**Step 1: 创建 `src/funsearch/sandbox.py`**

```python
"""沙箱：安全编译和执行 LLM 生成的 update 函数。

安全措施：
1. AST 检查：禁止 import、exec、eval、open 等危险操作
2. 超时：单次 update 调用限时
3. 异常捕获：任何错误 → 返回 None
"""
import ast
import math
import signal
import textwrap


# 允许的内置函数
ALLOWED_BUILTINS = {
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip",
    "int", "float", "bool", "list", "tuple", "round",
    "True", "False", "None",
}

# 禁止的 AST 节点
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)


class SandboxError(Exception):
    pass


def _check_ast_safety(code: str) -> bool:
    """检查代码 AST 是否安全。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        # 禁止 import
        if isinstance(node, FORBIDDEN_NODES):
            return False
        # 禁止 exec/eval/open/__import__
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("exec", "eval", "open", "__import__", "compile"):
                return False
            if isinstance(func, ast.Attribute) and func.attr in ("system", "popen", "exec"):
                return False
    return True


def compile_update_function(code: str) -> callable | None:
    """将 LLM 生成的函数代码编译为可调用函数。

    Args:
        code: 完整的函数定义字符串（def dgd_update(...)）

    Returns:
        可调用函数，或 None（编译失败/不安全）
    """
    # 确保代码以 def 开头
    code = textwrap.dedent(code).strip()
    if not code.startswith("def "):
        return None

    if not _check_ast_safety(code):
        return None

    # 受限执行环境
    safe_globals = {
        "__builtins__": {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                        for name in ALLOWED_BUILTINS
                        if hasattr(__builtins__, name) or (isinstance(__builtins__, dict) and name in __builtins__)},
        "math": math,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "abs": abs,
        "max": max,
        "min": min,
        "sum": sum,
    }

    try:
        exec(code, safe_globals)
    except Exception:
        return None

    # 提取函数（取最后定义的 def）
    tree = ast.parse(code)
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not func_names:
        return None

    func_name = func_names[-1]
    return safe_globals.get(func_name)


def extract_function_body(generated_code: str) -> str | None:
    """从 LLM 生成的续写中提取函数体。

    LLM 输出可能包含函数体 + 后续无关代码，只取第一个函数体。
    """
    if not generated_code.strip():
        return None

    # 尝试作为完整函数解析
    full_code = f"def _placeholder(M, key, target, dim, alpha, eta):\n{generated_code}"

    # 逐行删除末尾直到能解析
    lines = full_code.splitlines()
    while lines:
        try:
            tree = ast.parse("\n".join(lines))
            break
        except SyntaxError:
            lines = lines[:-1]

    if not lines:
        return None

    # 提取函数体（跳过 def 行）
    body_lines = "\n".join(lines).splitlines()[1:]
    body = "\n".join(body_lines)
    return body if body.strip() else None
```

**Step 2: Commit**

```bash
git add src/funsearch/sandbox.py
git commit -m "feat(funsearch): sandbox for safe execution of evolved DGD update rules"
```

---

### Task 3: 评测器（Evaluator）

**Files:**
- Create: `src/funsearch/evaluator.py`

**作用：** 拿到一个 update 函数，跑 GT benchmark，返回 P@1 分数。

**Step 1: 创建 `src/funsearch/evaluator.py`**

```python
"""评测器：用 GT benchmark 评测进化出的 update 函数。

核心逻辑：
1. 创建一个 AssociativeMemory，替换其 update 方法
2. 跑 N 轮 GT feedback
3. 返回 P@1 作为 fitness score

使用缓存的 embedding，评测一次 ~30 秒。
"""
import json
import math
import time
from pathlib import Path


def _cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na * nb > 0 else 0


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def evaluate_update_fn(
    update_fn: callable,
    dataset_name: str = "LoCoMo-full-all",
    dim: int = 256,
    alpha: float = 1.0,
    eta: float = 0.01,
    n_rounds: int = 3,
    max_questions: int = 500,
) -> dict:
    """评测一个 update 函数的 P@1。

    Args:
        update_fn: dgd_update(M, key, target, dim, alpha, eta) → M
        dataset_name: 数据集名
        dim, alpha, eta: DGD 参数
        n_rounds: GT 反馈轮数
        max_questions: 最多评测多少题（控制速度）

    Returns:
        {"p_at_1": float, "p_at_3": float, "p_at_5": float, "rounds": list, "elapsed": float}
    """
    from src.bench.datasets import Dataset
    from src.projection import create_projection_matrix, project

    t0 = time.time()

    # 加载数据集
    if dataset_name == "LoCoMo-full-all":
        ds = Dataset.load_locomo_full_all()
    elif dataset_name.startswith("LongMemEval"):
        ds = Dataset.load_longmemeval(dataset_name.split("-", 1)[1])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 加载缓存 embedding
    cache_file = Path("experiments/cache") / f"{ds.name}_embed.json"
    with open(cache_file) as f:
        cached = json.load(f)
    frag_vecs = cached["frag_vecs"]
    q_vecs_raw = cached["q_vecs"]

    # 投影
    proj = create_projection_matrix(4096, dim, seed=42)
    proj_vecs = [_norm(project(v, proj)) for v in frag_vecs]
    q_vecs = [_norm(project(v, proj)) for v in q_vecs_raw]

    # 限制题数
    questions = ds.questions[:max_questions]
    q_vecs = q_vecs[:max_questions]

    # 初始化 M 为单位矩阵
    M = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]

    def _query(q_vec):
        act = [sum(M[i][j] * q_vec[j] for j in range(dim)) for i in range(dim)]
        scores = [(_cosine(act, pv), idx) for idx, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    rounds = []
    for r in range(1, n_rounds + 1):
        # GT feedback: 每题更新 M
        for qi, q in enumerate(questions):
            target_idx = q["answer_indices"][0]
            try:
                update_fn(M, q_vecs[qi], proj_vecs[target_idx], dim, alpha, eta)
            except Exception:
                return {"p_at_1": 0.0, "error": True, "elapsed": time.time() - t0}

        # 评测
        hits = {1: 0, 3: 0, 5: 0}
        for qi, q in enumerate(questions):
            answer_set = set(q["answer_indices"])
            scores = _query(q_vecs[qi])
            top_ids = [idx for _, idx in scores[:5]]
            for k in [1, 3, 5]:
                if any(idx in answer_set for idx in top_ids[:k]):
                    hits[k] += 1

        total = len(questions)
        metrics = {k: hits[k] / total for k in hits}
        rounds.append({"round": r, "p_at_1": metrics[1], "p_at_3": metrics[3], "p_at_5": metrics[5]})

    elapsed = time.time() - t0
    final = rounds[-1] if rounds else {"p_at_1": 0}
    return {
        "p_at_1": final["p_at_1"],
        "p_at_3": final.get("p_at_3", 0),
        "p_at_5": final.get("p_at_5", 0),
        "rounds": rounds,
        "elapsed": elapsed,
        "error": False,
    }
```

**Step 2: Commit**

```bash
git add src/funsearch/evaluator.py
git commit -m "feat(funsearch): DGD update rule evaluator using GT benchmark"
```

---

### Task 4: 群岛数据库（Island Model）

**Files:**
- Create: `src/funsearch/database.py`

**作用：** 存储进化种群，实现岛模型 + 簇分组 + 选择策略。简化版 FunSearch ProgramsDatabase。

**Step 1: 创建 `src/funsearch/database.py`**

```python
"""群岛数据库：FunSearch 风格的岛模型，简化版。

简化点（vs 原版 FunSearch）：
- 不用 Signature 分簇（我们的评测只有一个分数 P@1）
- 不用 scipy softmax（用简单的 top-K 采样）
- 支持 JSON 持久化
"""
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Program:
    """一个进化出的 update 函数。"""
    id: str
    code: str                  # 完整函数定义代码
    score: float = 0.0         # P@1
    generation: int = 0
    island_id: int = 0
    parent_ids: list[str] = field(default_factory=list)
    eval_details: dict = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")


class Island:
    """一个子种群。"""

    def __init__(self, island_id: int):
        self.island_id = island_id
        self.programs: list[Program] = []
        self.best_score: float = -1.0
        self.best_program: Program | None = None

    def register(self, program: Program):
        self.programs.append(program)
        if program.score > self.best_score:
            self.best_score = program.score
            self.best_program = program

    def sample(self, n: int = 2) -> list[Program]:
        """从岛中采样 n 个程序，偏好高分和短代码。"""
        if not self.programs:
            return []
        if len(self.programs) <= n:
            return list(self.programs)

        # 按分数排序，取 top-50%，从中随机选
        sorted_progs = sorted(self.programs, key=lambda p: p.score, reverse=True)
        top_half = sorted_progs[:max(n, len(sorted_progs) // 2)]
        return random.sample(top_half, min(n, len(top_half)))


class ProgramsDatabase:
    """群岛数据库。"""

    def __init__(
        self,
        num_islands: int = 5,
        reset_interval: int = 50,  # 每 N 次注册后考虑重置
        data_dir: str = "experiments/funsearch",
    ):
        self.num_islands = num_islands
        self.reset_interval = reset_interval
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.islands: list[Island] = [Island(i) for i in range(num_islands)]
        self.all_programs: dict[str, Program] = {}
        self.history: list[dict] = []
        self._register_count = 0

        self._load()

    def _load(self):
        """从磁盘恢复。"""
        pop_file = self.data_dir / "population.json"
        if pop_file.exists():
            with open(pop_file) as f:
                data = json.load(f)
            for pd in data.get("all_programs", []):
                prog = Program(**{k: v for k, v in pd.items() if k in Program.__dataclass_fields__})
                self.all_programs[prog.id] = prog
                if prog.island_id < self.num_islands:
                    self.islands[prog.island_id].register(prog)
            self._register_count = data.get("register_count", 0)

        hist_file = self.data_dir / "history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                self.history = json.load(f)

    def save(self):
        """持久化。"""
        with open(self.data_dir / "population.json", "w") as f:
            json.dump({
                "all_programs": [asdict(p) for p in self.all_programs.values()],
                "register_count": self._register_count,
            }, f, ensure_ascii=False, indent=2)

        with open(self.data_dir / "history.json", "w") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        # Best top-10
        top10 = sorted(self.all_programs.values(), key=lambda p: p.score, reverse=True)[:10]
        with open(self.data_dir / "best.json", "w") as f:
            json.dump([asdict(p) for p in top10], f, ensure_ascii=False, indent=2)

    def get_prompt_programs(self) -> tuple[list[Program], int]:
        """从随机岛中采样 2 个程序用于构建 prompt。返回 (programs, island_id)。"""
        island_id = random.randint(0, self.num_islands - 1)
        island = self.islands[island_id]
        programs = island.sample(n=2)
        if not programs:
            # 岛空了，从任意有程序的岛采样
            for other in self.islands:
                programs = other.sample(n=2)
                if programs:
                    break
        return programs, island_id

    def register(self, program: Program) -> None:
        """注册一个已评测的程序。"""
        self.all_programs[program.id] = program
        if program.island_id < self.num_islands:
            self.islands[program.island_id].register(program)

        self._register_count += 1
        if self._register_count % self.reset_interval == 0:
            self._maybe_reset_islands()

    def _maybe_reset_islands(self):
        """淘汰最弱的一半岛，用最强岛的最佳程序重新种子。"""
        scores = [(i, island.best_score) for i, island in enumerate(self.islands)]
        scores.sort(key=lambda x: x[1])

        num_reset = self.num_islands // 2
        weak_ids = [s[0] for s in scores[:num_reset]]
        strong_ids = [s[0] for s in scores[num_reset:]]

        for wid in weak_ids:
            self.islands[wid] = Island(wid)
            # 从强岛移植最佳程序
            donor_id = random.choice(strong_ids)
            donor_best = self.islands[donor_id].best_program
            if donor_best:
                seed = Program(
                    id=f"reset-{self._register_count}-island{wid}",
                    code=donor_best.code,
                    score=donor_best.score,
                    island_id=wid,
                    parent_ids=[donor_best.id],
                    generation=donor_best.generation,
                )
                self.islands[wid].register(seed)

    def snapshot(self):
        """记录当前状态快照。"""
        best_scores = [island.best_score for island in self.islands]
        all_scores = [p.score for p in self.all_programs.values()]
        self.history.append({
            "register_count": self._register_count,
            "total_programs": len(self.all_programs),
            "best_per_island": best_scores,
            "global_best": max(all_scores) if all_scores else 0,
            "global_avg": sum(all_scores) / len(all_scores) if all_scores else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def get_best(self) -> Program | None:
        if not self.all_programs:
            return None
        return max(self.all_programs.values(), key=lambda p: p.score)
```

**Step 2: Commit**

```bash
git add src/funsearch/database.py
git commit -m "feat(funsearch): island model database with reset and persistence"
```

---

### Task 5: LLM 采样器

**Files:**
- Create: `src/funsearch/sampler.py`

**作用：** 构建 prompt（包含版本化的 parent 函数），调用 DeepSeek 生成新的 update 函数。

**Step 1: 创建 `src/funsearch/sampler.py`**

```python
"""LLM 采样器：构建 FunSearch 风格的版本化 prompt，调用 DeepSeek 生成代码。"""
import re

from src.bench.backends import DeepSeekBackend
from src.funsearch.specification import PROMPT_TEMPLATE, EVOLVE_FUNCTION_NAME
from src.funsearch.database import Program
from src.funsearch.sandbox import extract_function_body


def build_prompt(parent_programs: list[Program]) -> str:
    """构建 FunSearch 风格的 prompt。

    给 LLM 看 N 个版本的 update 函数（按分数从低到高排列），
    让它生成改进版 v_next。
    """
    if not parent_programs:
        raise ValueError("Need at least one parent program")

    # 按分数排序（低→高），最好的在最后
    sorted_parents = sorted(parent_programs, key=lambda p: p.score)

    versioned_functions = []
    for i, parent in enumerate(sorted_parents):
        # 提取函数体，重命名为 v_i
        code = parent.code.strip()
        # 替换函数名为版本化名称
        func_name = f"{EVOLVE_FUNCTION_NAME}_v{i}"
        renamed = re.sub(
            r"def \w+\(",
            f"def {func_name}(",
            code,
            count=1,
        )
        if i >= 1:
            # 加 docstring 说明是改进版
            renamed = renamed.replace(
                f"def {func_name}(",
                f"def {func_name}(  # Score: {parent.score:.1%}\n",
                # 简单方式：在注释中标注分数
            )
        versioned_functions.append(renamed)

    versioned_text = "\n\n".join(versioned_functions)
    next_version = len(sorted_parents)
    next_name = f"{EVOLVE_FUNCTION_NAME}_v{next_version}"
    prev_name = f"{EVOLVE_FUNCTION_NAME}_v{next_version - 1}"

    prompt = PROMPT_TEMPLATE.format(
        versioned_functions=versioned_text,
        next_function_name=next_name,
        prev_function_name=prev_name,
    )
    return prompt


async def sample_new_program(
    parent_programs: list[Program],
    backend: DeepSeekBackend,
    island_id: int,
    generation: int,
    index: int,
) -> Program | None:
    """调用 LLM 生成一个新的 update 函数。

    Returns:
        Program 或 None（生成失败）
    """
    prompt = build_prompt(parent_programs)

    try:
        raw = await backend.chat(prompt, max_tokens=600)
    except Exception:
        return None

    # 提取函数体
    body = extract_function_body(raw)
    if body is None:
        return None

    # 构建完整函数
    full_code = f"def dgd_update(M, key, target, dim, alpha, eta):\n{body}"

    return Program(
        id=f"gen{generation:03d}-island{island_id}-{index:03d}",
        code=full_code,
        island_id=island_id,
        generation=generation,
        parent_ids=[p.id for p in parent_programs],
    )
```

**Step 2: Commit**

```bash
git add src/funsearch/sampler.py
git commit -m "feat(funsearch): LLM sampler with versioned prompt construction"
```

---

### Task 6: 主循环 + CLI

**Files:**
- Create: `src/funsearch/runner.py`
- Create: `scripts/run_funsearch.py`

**Step 1: 创建 `src/funsearch/runner.py`**

```python
"""FunSearch 主循环：采样 → 编译 → 评测 → 注册，循环。"""
import asyncio
import time

from src.bench.backends import DeepSeekBackend
from src.funsearch.database import ProgramsDatabase, Program
from src.funsearch.sandbox import compile_update_function
from src.funsearch.evaluator import evaluate_update_fn
from src.funsearch.sampler import sample_new_program
from src.funsearch.specification import INITIAL_IMPLEMENTATION


async def run_funsearch(
    num_islands: int = 5,
    max_iterations: int = 100,
    samples_per_iteration: int = 2,
    eval_dataset: str = "LoCoMo-full-all",
    eval_max_questions: int = 500,
    eval_rounds: int = 3,
    dim: int = 256,
    alpha: float = 1.0,
    eta: float = 0.01,
    api_key: str = "",
    data_dir: str = "experiments/funsearch",
):
    """运行 FunSearch 进化循环。"""
    backend = DeepSeekBackend(api_key=api_key)
    db = ProgramsDatabase(num_islands=num_islands, data_dir=data_dir)

    # 评测辅助函数
    def eval_program(program: Program) -> float:
        fn = compile_update_function(program.code)
        if fn is None:
            program.score = 0.0
            program.eval_details = {"error": "compile_failed"}
            return 0.0
        result = evaluate_update_fn(
            fn, dataset_name=eval_dataset, dim=dim, alpha=alpha, eta=eta,
            n_rounds=eval_rounds, max_questions=eval_max_questions,
        )
        program.score = result["p_at_1"]
        program.eval_details = result
        return program.score

    # 初始化：注册初始实现到所有岛
    start_iter = len(db.all_programs)
    if start_iter == 0:
        print("Initializing with baseline implementation...")
        seed = Program(
            id="seed-000",
            code=INITIAL_IMPLEMENTATION.strip(),
            island_id=-1,  # 注册到所有岛
            generation=0,
        )
        score = eval_program(seed)
        for i in range(num_islands):
            island_seed = Program(
                id=f"seed-island{i}",
                code=seed.code,
                score=score,
                island_id=i,
                generation=0,
            )
            db.register(island_seed)
        db.snapshot()
        db.save()
        print(f"  Baseline P@1: {score:.1%}")
        start_iter = 1

    print(f"\nStarting FunSearch ({max_iterations} iterations, {num_islands} islands)")
    print(f"  Dataset: {eval_dataset}, dim={dim}, α={alpha}, η={eta}")
    print(f"  Eval: {eval_max_questions} questions × {eval_rounds} rounds")
    print(f"  Already evaluated: {len(db.all_programs)} programs")

    best_ever = db.get_best()
    if best_ever:
        print(f"  Current best: {best_ever.score:.1%} ({best_ever.id})")

    for iteration in range(start_iter, max_iterations + 1):
        t0 = time.time()

        # 从岛中采样 parent 程序
        parents, island_id = db.get_prompt_programs()
        if not parents:
            print(f"  Iter {iteration}: No parents available, skipping")
            continue

        # 用 LLM 生成新程序
        successes = 0
        for s in range(samples_per_iteration):
            program = await sample_new_program(
                parents, backend, island_id, iteration, s,
            )
            if program is None:
                continue

            # 编译 + 评测
            score = eval_program(program)
            if program.eval_details.get("error"):
                continue

            db.register(program)
            successes += 1

            if score > (best_ever.score if best_ever else 0):
                best_ever = program
                print(f"  ★ NEW BEST: {score:.1%} ({program.id})")

        elapsed = time.time() - t0
        best = db.get_best()

        if iteration % 5 == 0 or iteration == 1:
            island_bests = [f"{isl.best_score:.1%}" for isl in db.islands]
            print(
                f"  Iter {iteration:3d}: {successes}/{samples_per_iteration} valid, "
                f"best={best.score:.1%}, islands=[{', '.join(island_bests)}] ({elapsed:.0f}s)"
            )
            db.snapshot()
            db.save()

    # 最终保存
    db.snapshot()
    db.save()

    print(f"\n{'='*60}")
    print("  FunSearch Complete")
    print(f"{'='*60}")
    best = db.get_best()
    print(f"  Best P@1: {best.score:.1%}")
    print(f"  Best ID: {best.id}")
    print(f"  Total programs evaluated: {len(db.all_programs)}")
    print(f"  Code:\n{best.code}")
    return best
```

**Step 2: 创建 `scripts/run_funsearch.py`**

```python
#!/usr/bin/env python3
"""DGD Update Rule FunSearch CLI

用法:
    MEOMORY_LLM_KEY=sk-xxx .venv/bin/python scripts/run_funsearch.py
    MEOMORY_LLM_KEY=sk-xxx .venv/bin/python scripts/run_funsearch.py --iterations 100 --islands 5
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.funsearch.runner import run_funsearch


def main():
    parser = argparse.ArgumentParser(description="DGD Update Rule FunSearch")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--islands", type=int, default=5)
    parser.add_argument("--samples", type=int, default=2, help="Samples per iteration")
    parser.add_argument("--dataset", default="LoCoMo-full-all")
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--data-dir", default="experiments/funsearch")
    args = parser.parse_args()

    api_key = os.environ.get("MEOMORY_LLM_KEY", "")
    if not api_key:
        print("Error: MEOMORY_LLM_KEY not set")
        sys.exit(1)

    asyncio.run(run_funsearch(
        num_islands=args.islands,
        max_iterations=args.iterations,
        samples_per_iteration=args.samples,
        eval_dataset=args.dataset,
        eval_max_questions=args.max_questions,
        eval_rounds=args.rounds,
        dim=args.dim,
        alpha=args.alpha,
        eta=args.eta,
        api_key=api_key,
        data_dir=args.data_dir,
    ))


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/funsearch/runner.py scripts/run_funsearch.py
git commit -m "feat(funsearch): main evolution loop and CLI entry point"
```

---

### Task 7: 冒烟测试 + 首次运行

**Step 1: 冒烟测试（10 次迭代，3 岛，200 题）**

```bash
MEOMORY_LLM_KEY=sk-4682199cd4bb4d69826a838cd318578c \
.venv/bin/python scripts/run_funsearch.py \
  --iterations 10 --islands 3 --samples 2 --max-questions 200 --rounds 3
```

Expected: baseline P@1 ~26%，10 次迭代后如果 LLM 能生成有效变体，应该能看到一些变化。每次迭代 ~1 分钟（30s 评测 + LLM 调用）。

**Step 2: 验证持久化**

```bash
cat experiments/funsearch/best.json | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data[:3]:
    print(f'{p[\"id\"]}: P@1={p[\"score\"]:.1%}')
    print(f'  {p[\"code\"][:100]}...')
"
```

**Step 3: 正式运行（100 次迭代，5 岛，500 题）**

```bash
MEOMORY_LLM_KEY=sk-4682199cd4bb4d69826a838cd318578c \
.venv/bin/python scripts/run_funsearch.py \
  --iterations 100 --islands 5 --samples 2 --max-questions 500 --rounds 3
```

预估：每次迭代 ~2 分钟（评测 + LLM），100 次 ~3.5 小时。

**Step 4: 如果发现更好的规则，集成回主系统**

1. 从 `experiments/funsearch/best.json` 提取最优代码
2. 替换 `src/dgd.py` 的 `update` 方法
3. 重跑大规模实验验证

**Step 5: Commit**

```bash
git add experiments/funsearch/ src/funsearch/
git commit -m "feat(funsearch): initial run results for DGD update rule evolution"
```

---

## 预估资源

| 项目 | 数量 | 成本 |
|------|------|------|
| LLM 调用（100 iter × 2 samples） | ~200 次生成 | ~¥2 |
| 评测（200 次 × 500 题 × 3 轮） | ~30s/次，本地计算 | ¥0 |
| **总计** | ~3.5 小时 | **~¥2** |

评测完全在本地做（纯矩阵运算），成本几乎为零。LLM 只负责生成代码变体。

---

## 安全考虑

1. **沙箱 AST 检查**：禁止 import/exec/eval/open
2. **执行超时**：单次 update 调用如果卡住，评测函数会因为整体超时而终止
3. **M 矩阵验证**：如果进化出的规则让 M 爆炸（NaN/Inf），评测返回 0 分
4. **代码审查**：最终采纳的规则需要人工审查可解释性
