# FunSearch v2 升级计划

> 日期：2026-03-17
> 目标：将我们的 FunSearch 实现对齐 Google DeepMind 原版设计，补齐关键差距

---

## 逐组件对照分析

### 1. Database (`programs_database.py` vs `database.py`)

#### Google 实现关键特性

- **Signature 精确分簇**（L54-56）：用多测试用例的分数向量 `tuple[float, ...]` 作为 Cluster key，相同输出行为的程序归为一簇
  ```python
  def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
      return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))
  ```
- **_reduce_score**（L49-51）：取最后一个测试用例的分数作为单一 score（用于排序），signature 用于分簇
  ```python
  def _reduce_score(scores_per_test: ScoresPerTest) -> float:
      return scores_per_test[list(scores_per_test.keys())[-1]]
  ```
- **Prompt 数据结构**（L59-72）：`Prompt(code, version_generated, island_id)` —— 包含完整的 prompt 文本、版本号、岛 ID
- **Island 内部的 prompt 构建**（L205-271）：
  - 两层采样：softmax 选簇 → 簇内偏好短代码
  - **温度衰减周期**（L212-214）：`temperature = init * (1 - (num_programs % period) / period)`，周期性从探索到利用
  - **版本化函数名**（L244-255）：`_v0`, `_v1`, ...，并用 `code_manipulation.rename_function_calls` 精确替换递归调用
  - **docstring 注入**（L249-250）：`Improved version of _v{i-1}`
  - **生成完整 Program**（L270-271）：用 `dataclasses.replace(self._template, functions=versioned_functions)` 替换模板中的函数
- **Island 重置**（L147-167）：
  - 按分数排序 + 微量噪声打破平局（L150-152）
  - 重置后用强岛的 best program 作为种子，**包括其 scores_per_test**（L166-167）
  - **基于时间**触发（L143：`time.time() - self._last_reset_time > self._config.reset_period`）
- **register_program 区分初始 vs 进化**（L125-140）：`island_id=None` 时注册到所有岛

#### 我们的实现

- **score_bucket 分簇**（L92-95）：`round(score * 200) / 200`，即 0.5% 精度的浮点桶
- 没有 Signature 概念，没有多测试用例分数向量
- **Prompt 构建在 sampler.py 中**，database 只返回 `(list[Program], island_id)`
- **温度衰减**（L122-124）：`temperature_init * (1 - progress) + 0.01`，有 +0.01 下限，Google 没有
- **Island 重置基于计数**（L224：`_register_count % reset_interval == 0`），不是基于时间
- **没有模板系统**：没有 `Program`/`Function` AST 数据结构，prompt 生成用字符串拼接
- **有持久化**（Google 没有）：JSON 存盘/恢复、历史快照、best.json —— 这是我们的优势
- **有全局 all_programs 字典**（Google 没有）：便于分析和恢复

#### 差距分析

| 特性 | Google | 我们 | 差距等级 |
|------|--------|------|---------|
| Signature 分簇 | 精确多测试分数向量 | 单一 score 四舍五入 | **严重** |
| 模板 + AST 操作 | `Program`/`Function` dataclass | 无 | **严重** |
| Prompt 构建 | Island 内构建完整 prompt | sampler 中字符串拼接 | **严重** |
| 递归调用重命名 | tokenizer 级精确替换 | regex 粗糙替换 | **中等** |
| 温度衰减 | 纯线性到 0 | 有 +0.01 下限 | 低 |
| 重置触发 | 基于时间 | 基于计数 | 低（两种都合理） |
| 持久化 | 无 | 有 | **我们领先** |
| 快照/监控 | 无 | 有 | **我们领先** |

#### 修改方案

##### 1.1 引入 Signature 和多测试用例评分

在 `database.py` 中添加 Signature 类型和相关工具函数：

```python
# database.py 顶部新增
from collections.abc import Mapping
import numpy as np
import scipy.special

Signature = tuple[float, ...]
ScoresPerTest = Mapping[str, float]  # test_input -> score

def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """取最后一个测试用例的分数。"""
    return scores_per_test[list(scores_per_test.keys())[-1]]

def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """多测试用例分数向量，作为 Cluster key。"""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))
```

##### 1.2 修改 Cluster 使用 Signature 作为 key

当前 `Cluster.__init__`（L48）接受 `score_bucket: float`，需要改为接受 `Signature`：

```python
class Cluster:
    """分数签名相同的程序簇。"""

    def __init__(self, score: float, program: Program):
        self.score = score  # reduced score，用于排序
        self.programs: list[Program] = [program]
        self._lengths: list[int] = [len(program.code)]

    def register(self, program: Program):
        self.programs.append(program)
        self._lengths.append(len(program.code))

    def sample_program(self) -> Program:
        """采样，偏好短代码。与 Google 实现对齐。"""
        if len(self.programs) == 1:
            return self.programs[0]
        lengths = np.array(self._lengths, dtype=np.float32)
        normalized = (lengths - lengths.min()) / (lengths.max() + 1e-6)
        probs = scipy.special.softmax(-normalized / 1.0)
        # 确保概率和为 1
        idx = np.argmax(probs)
        probs[idx] = 1 - np.sum(probs[:idx]) - np.sum(probs[idx+1:])
        return np.random.choice(self.programs, p=probs)
```

##### 1.3 修改 Island 使用 Signature 分簇

当前 `Island.clusters` 是 `dict[float, Cluster]`（L85），需要改为 `dict[Signature, Cluster]`：

```python
class Island:
    def __init__(self, island_id: int, temperature_init: float = 0.1,
                 temperature_period: int = 30_000,
                 functions_per_prompt: int = 2):
        self.island_id = island_id
        self.clusters: dict[Signature, Cluster] = {}
        self.best_score: float = -float('inf')
        self.best_program: Program | None = None
        self.best_scores_per_test: ScoresPerTest | None = None
        self._num_programs = 0
        self._temperature_init = temperature_init
        self._temperature_period = temperature_period
        self._functions_per_prompt = functions_per_prompt

    def register(self, program: Program, scores_per_test: ScoresPerTest):
        """注册程序，用 Signature 分簇。"""
        signature = _get_signature(scores_per_test)
        score = _reduce_score(scores_per_test)
        if signature not in self.clusters:
            self.clusters[signature] = Cluster(score, program)
        else:
            self.clusters[signature].register(program)
        self._num_programs += 1
        if score > self.best_score:
            self.best_score = score
            self.best_program = program
            self.best_scores_per_test = scores_per_test

    def sample(self, n: int = None) -> list[Program]:
        """两层采样（与 Google 对齐）。"""
        if not self.clusters:
            return []
        n = n or self._functions_per_prompt

        signatures = list(self.clusters.keys())
        cluster_scores = np.array(
            [self.clusters[sig].score for sig in signatures])

        # 温度周期性衰减（与 Google 完全对齐，不加 +0.01 下限）
        period = self._temperature_period
        temperature = self._temperature_init * (
            1 - (self._num_programs % period) / period)
        temperature = max(temperature, 1e-8)  # 仅防除零

        probs = scipy.special.softmax(cluster_scores / temperature)
        idx_max = np.argmax(probs)
        probs[idx_max] = 1 - np.sum(probs[:idx_max]) - np.sum(probs[idx_max+1:])

        functions_per_prompt = min(len(self.clusters), n)
        idx = np.random.choice(len(signatures), size=functions_per_prompt, p=probs)
        chosen = [signatures[i] for i in idx]

        implementations = []
        scores = []
        for sig in chosen:
            cluster = self.clusters[sig]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        # 按分数排序（低→高），最好的在最后
        order = np.argsort(scores)
        return [implementations[i] for i in order]
```

##### 1.4 修改 ProgramsDatabase.register 接受 ScoresPerTest

```python
class ProgramsDatabase:
    def register(self, program: Program, island_id: int | None = None,
                 scores_per_test: ScoresPerTest | None = None) -> None:
        """注册一个已评测的程序。

        island_id=None 时注册到所有岛（初始种子程序）。
        """
        if scores_per_test is None:
            # 兼容旧接口：单一分数转为 ScoresPerTest
            scores_per_test = {"default": program.score}

        self.all_programs[program.id] = program
        if island_id is None:
            for i in range(self.num_islands):
                self._register_in_island(program, i, scores_per_test)
        else:
            self._register_in_island(program, island_id, scores_per_test)

        self._register_count += 1
        if self._register_count % self.reset_interval == 0:
            self._maybe_reset_islands()

    def _register_in_island(self, program: Program, island_id: int,
                             scores_per_test: ScoresPerTest) -> None:
        self.islands[island_id].register(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_score_per_island[island_id] = score
            self._best_program_per_island[island_id] = program
            self._best_spt_per_island[island_id] = scores_per_test
```

##### 1.5 重置时携带 scores_per_test

当前 `_maybe_reset_islands`（L227-249）重置后创建新 seed 只有 score，需要传递 `scores_per_test`：

```python
def _maybe_reset_islands(self):
    """淘汰最弱的一半岛，用强岛 best 重新播种。"""
    scores = np.array([isl.best_score for isl in self.islands])
    scores += np.random.randn(len(scores)) * 1e-6  # 打破平局
    indices = np.argsort(scores)

    num_reset = self.num_islands // 2
    weak_ids = indices[:num_reset]
    strong_ids = indices[num_reset:]

    for wid in weak_ids:
        self.islands[wid] = Island(
            wid,
            temperature_init=self._temperature_init,
            temperature_period=self._temperature_period,
            functions_per_prompt=self._functions_per_prompt,
        )
        donor_id = np.random.choice(strong_ids)
        donor = self.islands[donor_id]
        if donor.best_program and donor.best_scores_per_test:
            self.islands[wid].register(
                donor.best_program, donor.best_scores_per_test)
```

---

### 2. Evaluator (`evaluator.py`)

#### Google 实现关键特性

- **_FunctionLineVisitor**（L26-43）：AST visitor 找函数末行，用于 `_trim_function_body`
- **_trim_function_body**（L46-65）：从 LLM 续写中截取函数体，逐行删末尾直到能解析
  ```python
  code = f'def fake_function_header():\n{generated_code}'
  while tree is None:
      try:
          tree = ast.parse(code)
      except SyntaxError as e:
          code = '\n'.join(code.splitlines()[:e.lineno - 1])
  ```
- **_sample_to_program**（L68-85）：
  - 截取函数体 → 重命名版本号调用 → 替换模板中的函数 → 返回 `(Function, str)`
  - 用 `code_manipulation.rename_function_calls` 把 `_v{N}` 调用改回原名（递归修复）
- **_calls_ancestor**（L103-112）：检测生成的函数是否调用了祖先版本（防止无限递归）
- **Sandbox 抽象类**（L88-101）：`run(program, function_to_run, test_input, timeout)` → `(output, runs_ok)`
- **Evaluator.analyse**（L135-155）：
  - 对**每个 test_input** 运行程序，收集 `scores_per_test`
  - 检查：`runs_ok` + 不调用祖先 + 输出非 None + 输出是 int/float
  - 有结果才注册

#### 我们的实现

- **evaluate_update_fn**（L28-120）：直接跑 benchmark，返回 `p_at_1` 等指标
- 不是对每个 test_input 独立评分，而是**整体跑一遍** benchmark 拿最终 P@1
- 没有 `_trim_function_body`（在 sandbox.py 中有类似的 `extract_function_body`）
- 没有 `_sample_to_program` 流程
- 没有 `_calls_ancestor` 检查
- 没有 Sandbox 抽象 —— 直接 `exec()` 编译

#### 差距分析

| 特性 | Google | 我们 | 差距等级 |
|------|--------|------|---------|
| 多测试用例评分 | 每 input 独立评分 | 整体 P@1 一个数字 | **严重**（与 Signature 分簇联动） |
| _calls_ancestor 检查 | 有 | 无 | **中等** |
| _trim_function_body | 精确截取 | extract_function_body 逻辑相似 | 低 |
| _sample_to_program | 模板替换 + 递归修复 | 字符串拼接 | **中等**（与模板系统联动） |
| Sandbox 抽象 | 有（抽象类，可换后端） | 直接 exec | 低（当前够用） |

#### 修改方案

##### 2.1 添加多测试用例评分

我们的场景比较特殊 —— 评测一个 update function 是在整个 benchmark 上跑。要引入多测试用例，有两种方案：

**方案 A（推荐）：按对话分组作为多测试用例**

```python
# evaluator.py 新增

def evaluate_update_fn_per_test(
    update_fn: callable,
    dataset_name: str = "LoCoMo-full-all",
    dim: int = 256,
    alpha: float = 1.0,
    eta: float = 0.01,
    n_rounds: int = 3,
    max_questions: int = 500,
) -> dict[str, float]:
    """返回 ScoresPerTest —— 每个对话 session 的 P@1。

    Key: session_id, Value: 该 session 的 P@1。
    这样不同 update rule 如果在某些 session 表现一致，
    它们的 Signature 就会相同，被归入同一个 Cluster。
    """
    from src.bench.datasets import Dataset
    from src.projection import create_projection_matrix, project

    # ... 加载数据集和 embedding（与原 evaluate_update_fn 相同）...

    # 按对话 session 分组
    session_scores = {}
    sessions = {}
    for qi, q in enumerate(questions):
        sid = q.get("session_id", f"q{qi}")
        if sid not in sessions:
            sessions[sid] = []
        sessions[sid].append((qi, q))

    # 对每个 session 独立评测
    M = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
    for r in range(1, n_rounds + 1):
        for qi, q in enumerate(questions):
            target_idx = q["answer_indices"][0]
            try:
                update_fn(M, q_vecs[qi], proj_vecs[target_idx], dim, alpha, eta)
            except Exception:
                return {"error": 0.0}

    for sid, qs in sessions.items():
        hits = 0
        for qi, q in qs:
            answer_set = set(q["answer_indices"])
            scores = _query(q_vecs[qi], M, proj_vecs, dim)
            top_ids = [idx for _, idx in scores[:1]]
            if any(idx in answer_set for idx in top_ids):
                hits += 1
        session_scores[sid] = hits / len(qs) if qs else 0.0

    # 最后一个 key 放全局 P@1（_reduce_score 取最后一个）
    all_hits = sum(1 for sid, s in session_scores.items() if s > 0)
    session_scores["_global_p1"] = all_hits / len(sessions) if sessions else 0.0
    return session_scores
```

**方案 B（简化）：如果数据集没有 session 分组，用分桶方式**

将问题按 index 分成 K 个桶（如 K=10），每桶的 P@1 作为一个测试用例分数。这样 Signature 是一个 10 维向量。

##### 2.2 添加 _calls_ancestor 检查

```python
# evaluator.py 或 sandbox.py 中新增

def _calls_ancestor(code: str, function_to_evolve: str) -> bool:
    """检测生成的函数是否调用了祖先版本（_v0, _v1, ...）。"""
    import re
    pattern = rf'{function_to_evolve}_v\d+'
    return bool(re.search(pattern, code))
```

在 `runner.py` 的 `eval_program` 中使用：

```python
# runner.py L31-43 修改
def eval_program(program: Program) -> float:
    # 检查是否调用祖先
    if _calls_ancestor(program.code, EVOLVE_FUNCTION_NAME):
        program.score = 0.0
        program.eval_details = {"error": "calls_ancestor"}
        return 0.0

    fn = compile_update_function(program.code)
    if fn is None:
        program.score = 0.0
        program.eval_details = {"error": "compile_failed"}
        return 0.0
    # ... 继续评测 ...
```

---

### 3. Sampler (`sampler.py`)

#### Google 实现关键特性

- **LLM 抽象类**（L25-37）：`_draw_sample(prompt) -> str`，`draw_samples(prompt) -> Collection[str]`
  - `samples_per_prompt` 控制每次生成多少个样本
- **Sampler 无限循环**（L53-62）：
  ```python
  def sample(self):
      while True:
          prompt = self._database.get_prompt()
          samples = self._llm.draw_samples(prompt.code)
          for sample in samples:
              chosen_evaluator = np.random.choice(self._evaluators)
              chosen_evaluator.analyse(sample, prompt.island_id, prompt.version_generated)
  ```
  - 从 database 获取 `Prompt` 对象（含 `code`, `version_generated`, `island_id`）
  - 多个 evaluator 并行评测
  - **Sampler 不做任何后处理** —— 所有截取、编译、评测都在 Evaluator 中

#### 我们的实现

- `build_prompt`（L10-53）：手动构建 prompt，用 regex 替换函数名
  - 在函数签名里加注释 `# Score: xx%`（L38-39），Google 是用 docstring
  - 用 `PROMPT_TEMPLATE.format(...)` 字符串模板
- `sample_new_program`（L56-89）：调用 LLM，截取函数体，构建完整函数，**返回 Program**
  - 把采样、截取、构建全放在一个函数里
  - 硬编码函数签名 `def dgd_update(M, key, target, dim, alpha, eta)`

#### 差距分析

| 特性 | Google | 我们 | 差距等级 |
|------|--------|------|---------|
| Prompt 从 Database 获取 | `database.get_prompt()` 返回完整 prompt | sampler 自己构建 | **严重** |
| 职责分离 | Sampler 只采样，Evaluator 做截取+编译+评测 | 全混在一起 | **中等** |
| 版本化函数名 | AST tokenizer 级精确替换 | regex `re.sub(r"def \w+\(", ...)` | **中等** |
| 分数标注 | docstring `Improved version of _v{i-1}` | 注释 `# Score: xx%` | 低 |
| LLM 抽象 | `LLM` 基类 | 直接用 `DeepSeekBackend` | 低 |
| 多样本 | `samples_per_prompt` | 在 runner 循环中控制 | 低 |

#### 修改方案

##### 3.1 引入 code_manipulation 模块

创建 `src/funsearch/code_manipulation.py`，从 Google 实现移植核心功能：

```python
"""代码操作工具：AST 解析、函数重命名、递归调用修复。

移植自 Google FunSearch 的 code_manipulation.py。
"""
import ast
import dataclasses
import io
import tokenize
from collections.abc import Iterator, MutableSet, Sequence


@dataclasses.dataclass
class Function:
    """一个解析后的 Python 函数。"""
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None

    def __str__(self) -> str:
        return_type = f' -> {self.return_type}' if self.return_type else ''
        function = f'def {self.name}({self.args}){return_type}:\n'
        if self.docstring:
            new_line = '\n' if self.body else ''
            function += f'  """{self.docstring}"""{new_line}'
        function += self.body + '\n\n'
        return function

    def __setattr__(self, name: str, value: str) -> None:
        if name == 'body':
            value = value.strip('\n')
        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip().replace('"""', '')
        super().__setattr__(name, value)


@dataclasses.dataclass(frozen=True)
class Program:
    """一个解析后的 Python 程序（前缀 + 函数列表）。"""
    preface: str
    functions: list[Function]

    def __str__(self) -> str:
        program = f'{self.preface}\n' if self.preface else ''
        program += '\n'.join([str(f) for f in self.functions])
        return program

    def get_function(self, function_name: str) -> Function:
        for f in self.functions:
            if f.name == function_name:
                return f
        raise ValueError(f'Function {function_name} not found')


def text_to_program(text: str) -> Program:
    """AST 解析文本为 Program 对象。"""
    tree = ast.parse(text)
    visitor = _ProgramVisitor(text)
    visitor.visit(tree)
    return visitor.return_program()


def text_to_function(text: str) -> Function:
    """AST 解析文本为单个 Function 对象。"""
    program = text_to_program(text)
    if len(program.functions) != 1:
        raise ValueError(f'Expected 1 function, got {len(program.functions)}')
    return program.functions[0]


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    """用 tokenizer 精确重命名函数调用。"""
    if source_name not in code:
        return code
    modified_tokens = []
    for token, is_call in _yield_token_and_is_call(code):
        if is_call and token.string == source_name:
            modified_token = tokenize.TokenInfo(
                type=token.type, string=target_name,
                start=token.start, end=token.end, line=token.line)
            modified_tokens.append(modified_token)
        else:
            modified_tokens.append(token)
    return tokenize.untokenize(modified_tokens).decode() if isinstance(
        tokenize.untokenize(modified_tokens), bytes) else tokenize.untokenize(modified_tokens)


def get_functions_called(code: str) -> MutableSet[str]:
    """返回代码中所有被调用的函数名集合。"""
    return set(
        token.string for token, is_call
        in _yield_token_and_is_call(code) if is_call)


# === 内部辅助 ===

class _ProgramVisitor(ast.NodeVisitor):
    def __init__(self, sourcecode: str):
        self._codelines = sourcecode.splitlines()
        self._preface = ''
        self._functions: list[Function] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.col_offset == 0:
            if not self._functions:
                self._preface = '\n'.join(self._codelines[:node.lineno - 1])
            body_start = node.body[0].lineno - 1
            docstring = None
            if isinstance(node.body[0], ast.Expr) and isinstance(
                    node.body[0].value, ast.Str):
                docstring = f'  """{ast.literal_eval(ast.unparse(node.body[0]))}"""'
                body_start = node.body[1].lineno - 1 if len(node.body) > 1 else node.end_lineno
            self._functions.append(Function(
                name=node.name,
                args=ast.unparse(node.args),
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=docstring,
                body='\n'.join(self._codelines[body_start:node.end_lineno]),
            ))
        self.generic_visit(node)

    def return_program(self) -> Program:
        return Program(preface=self._preface, functions=self._functions)


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    """逐 token 遍历，标记是否为函数调用。"""
    try:
        code_bytes = code.encode()
        tokens = tokenize.tokenize(io.BytesIO(code_bytes).readline)
        prev_token = None
        is_attr = False
        for token in tokens:
            if (prev_token and prev_token.type == tokenize.NAME and
                    token.type == tokenize.OP and token.string == '('):
                yield prev_token, not is_attr
                is_attr = False
            else:
                if prev_token:
                    is_attr = (prev_token.type == tokenize.OP and
                               prev_token.string == '.')
                    yield prev_token, False
            prev_token = token
        if prev_token:
            yield prev_token, False
    except tokenize.TokenizeError:
        pass
```

##### 3.2 将 Prompt 构建移入 Database

在 `Island` 中增加 `_generate_prompt` 方法（对齐 Google L236-271）：

```python
# database.py Island 类中新增

def get_prompt(self) -> tuple[str, int]:
    """构建 FunSearch 风格的版本化 prompt。

    返回 (prompt_text, version_generated)。
    """
    programs = self.sample()
    if not programs:
        return "", 0

    from src.funsearch import code_manipulation as cm
    from src.funsearch.specification import EVOLVE_FUNCTION_NAME

    versioned_functions = []
    for i, prog in enumerate(programs):
        # 解析为 Function 对象
        try:
            func = cm.text_to_function(prog.code)
        except Exception:
            continue
        func.name = f'{EVOLVE_FUNCTION_NAME}_v{i}'
        if i >= 1:
            func.docstring = f'Improved version of `{EVOLVE_FUNCTION_NAME}_v{i-1}`.'
        # 修复递归调用
        func_str = cm.rename_function_calls(
            str(func), EVOLVE_FUNCTION_NAME, func.name)
        versioned_functions.append(cm.text_to_function(func_str))

    # 生成待完成的函数头
    next_v = len(versioned_functions)
    header = Function(
        name=f'{EVOLVE_FUNCTION_NAME}_v{next_v}',
        args=versioned_functions[-1].args if versioned_functions else 'M, key, target, dim, alpha, eta',
        body='',
        docstring=f'Improved version of `{EVOLVE_FUNCTION_NAME}_v{next_v-1}`.',
    )
    versioned_functions.append(header)

    # 组装 prompt
    prompt = '\n'.join(str(f) for f in versioned_functions)
    return prompt, next_v
```

在 `ProgramsDatabase` 中新增（对齐 Google L104-108）：

```python
@dataclasses.dataclass(frozen=True)
class Prompt:
    code: str
    version_generated: int
    island_id: int

class ProgramsDatabase:
    def get_prompt(self) -> Prompt:
        island_id = random.randint(0, self.num_islands - 1)
        code, version_generated = self.islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)
```

##### 3.3 简化 sampler.py

Sampler 只负责调用 LLM，不做 prompt 构建：

```python
"""LLM 采样器：从 Database 获取 prompt，调用 LLM 生成续写。"""
from src.funsearch.database import ProgramsDatabase, Prompt


class LLM:
    """LLM 抽象基类。"""
    def __init__(self, samples_per_prompt: int = 4):
        self._samples_per_prompt = samples_per_prompt

    async def draw_sample(self, prompt: str) -> str:
        raise NotImplementedError

    async def draw_samples(self, prompt: str) -> list[str]:
        return [await self.draw_sample(prompt)
                for _ in range(self._samples_per_prompt)]


class DeepSeekLLM(LLM):
    """DeepSeek 后端。"""
    def __init__(self, backend, samples_per_prompt: int = 4):
        super().__init__(samples_per_prompt)
        self._backend = backend

    async def draw_sample(self, prompt: str) -> str:
        return await self._backend.chat(prompt, max_tokens=600)


class Sampler:
    """采样节点：获取 prompt → 调用 LLM → 返回原始续写。"""
    def __init__(self, database: ProgramsDatabase, llm: LLM):
        self._database = database
        self._llm = llm

    async def sample(self) -> list[tuple[str, Prompt]]:
        """一轮采样，返回 [(raw_sample, prompt), ...]。"""
        prompt = self._database.get_prompt()
        samples = await self._llm.draw_samples(prompt.code)
        return [(s, prompt) for s in samples]
```

---

### 4. Code Manipulation / Sandbox (`code_manipulation.py` / `sandbox.py`)

#### Google 实现关键特性

- **`code_manipulation.py` 是核心模块**（254 行）：
  - `Function` dataclass（L33-65）：解析和序列化函数，`__setattr__` 自动清理 body/docstring
  - `Program` dataclass（L68-100）：preface + functions，支持 `find_function_index`、`get_function`
  - `ProgramVisitor`（L103-145）：AST 解析整个程序
  - `rename_function_calls`（L212-230）：**tokenizer 级别**精确重命名函数调用（不误改变量名）
  - `get_functions_called`（L233-236）：获取所有被调用函数名
  - `yield_decorated`（L239-253）：查找装饰器标记的函数
- **`Sandbox` 抽象类**在 evaluator.py L88-101：只有接口，具体实现需要用户提供

#### 我们的实现

- `sandbox.py`：
  - `_check_ast_safety`（L29-47）：AST 安全检查（禁 import/exec/eval/open）
  - `compile_update_function`（L50-94）：`exec()` 编译 + 受限环境
  - `extract_function_body`（L97-156）：从 LLM 输出中提取函数体
- 没有 `code_manipulation.py` —— 所有 AST 操作分散在 sandbox.py 和 sampler.py 中
- regex 替换函数名（sampler.py L30-31）而不是 tokenizer 级替换

#### 差距分析

| 特性 | Google | 我们 | 差距等级 |
|------|--------|------|---------|
| Program/Function AST 数据结构 | 完整的解析+序列化 | 无 | **严重** |
| tokenizer 级函数重命名 | 有 | 无（用 regex） | **中等** |
| get_functions_called | 有 | 无 | **中等** |
| AST 安全检查 | 无（sandbox 负责） | 有 | **我们领先** |
| 受限执行环境 | 无（sandbox 负责） | 有 | **我们领先** |
| _trim_function_body | 精确截取 | extract_function_body 功能相似 | 低 |

#### 修改方案

##### 4.1 创建 code_manipulation.py（见 3.1 已给出完整代码）

##### 4.2 修改 sandbox.py 的 extract_function_body 对齐 _trim_function_body

当前 `extract_function_body` 已经有类似逻辑，但 Google 的更简洁。建议保留我们的版本（支持 markdown 代码块等），但增加一个 `_trim_function_body` 兼容入口：

```python
# sandbox.py 新增

def _trim_function_body(generated_code: str) -> str:
    """对齐 Google 的 _trim_function_body。

    从 LLM 续写中截取函数体，逐行删末尾直到能解析。
    """
    if not generated_code:
        return ''
    code = f'def _placeholder():\n{generated_code}'
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            code = '\n'.join(code.splitlines()[:e.lineno - 1])
        if not code.strip():
            return ''
    # 找到函数末行
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_placeholder':
            body_lines = code.splitlines()[1:node.end_lineno]
            return '\n'.join(body_lines) + '\n\n'
    return ''
```

##### 4.3 _sample_to_program 流程

```python
# sandbox.py 或新建 evaluator_utils.py

from src.funsearch import code_manipulation as cm
from src.funsearch.specification import EVOLVE_FUNCTION_NAME

def sample_to_program(
    generated_code: str,
    version_generated: int | None,
    template: cm.Program,
) -> tuple[cm.Function, str] | None:
    """将 LLM 续写编译为完整程序。

    对齐 Google evaluator.py L68-85 的 _sample_to_program。
    """
    body = _trim_function_body(generated_code)
    if not body.strip():
        return None

    # 重命名版本号调用
    if version_generated is not None:
        body = cm.rename_function_calls(
            body,
            f'{EVOLVE_FUNCTION_NAME}_v{version_generated}',
            EVOLVE_FUNCTION_NAME,
        )

    import copy
    program = copy.deepcopy(template)
    evolved = program.get_function(EVOLVE_FUNCTION_NAME)
    evolved.body = body
    return evolved, str(program)
```

---

### 5. Runner / Main Loop (`runner.py`)

#### Google 实现关键特性

Google 没有显式的 `runner.py`，但从 `sampler.py` L53-62 可见架构：

- **Sampler.sample() 是无限循环**（L55: `while True`）
- **多 Sampler 并行**（config.py L47: `num_samplers=15`）
- **多 Evaluator 并行**（config.py L49: `num_evaluators=140`）
- **Sampler 从 Database 取 prompt → LLM 生成 → 随机选一个 Evaluator 评测**
- 分布式架构：Sampler 和 Evaluator 可在不同机器上

#### 我们的实现

- **串行循环**（L79-119）：每次迭代依次 sample → compile → evaluate → register
- `samples_per_iteration=2`：每次生成 2 个样本
- **评测是同步的**（L37-43）：跑完 benchmark 才继续
- 没有多 Evaluator
- 有进度显示和快照（Google 没有）

#### 差距分析

| 特性 | Google | 我们 | 差距等级 |
|------|--------|------|---------|
| 并行架构 | 多 Sampler + 多 Evaluator | 串行 | **中等**（当前规模足够） |
| 职责分离 | Sampler→Evaluator 管道 | 全在 runner 里 | **中等** |
| 初始种子注册 | `island_id=None` → 所有岛 | 手动循环注册 | 低 |
| 进度显示 | 无 | 有 | **我们领先** |
| 断点恢复 | 无 | 有（从 JSON 恢复） | **我们领先** |

#### 修改方案

##### 5.1 重构 runner.py 使用新的组件接口

```python
"""FunSearch v2 主循环：对齐 Google 架构。"""
import asyncio
import time

from src.bench.backends import DeepSeekBackend
from src.funsearch.database import ProgramsDatabase, Program, Prompt, ScoresPerTest
from src.funsearch.sandbox import compile_update_function, sample_to_program
from src.funsearch.evaluator import evaluate_update_fn_per_test
from src.funsearch.sampler import Sampler, DeepSeekLLM
from src.funsearch.specification import EVOLVE_FUNCTION_NAME, INITIAL_IMPLEMENTATION
from src.funsearch import code_manipulation as cm


def _calls_ancestor(code: str) -> bool:
    """检测是否调用祖先版本。"""
    try:
        called = cm.get_functions_called(code)
    except Exception:
        return False
    return any(name.startswith(f'{EVOLVE_FUNCTION_NAME}_v') for name in called)


async def run_funsearch(
    num_islands: int = 5,
    max_iterations: int = 100,
    samples_per_prompt: int = 4,
    eval_dataset: str = "LoCoMo-full-all",
    eval_max_questions: int = 500,
    eval_rounds: int = 3,
    dim: int = 256,
    alpha: float = 1.0,
    eta: float = 0.01,
    api_key: str = "",
    data_dir: str = "experiments/funsearch",
):
    """运行 FunSearch v2 进化循环。"""
    backend = DeepSeekBackend(api_key=api_key)
    llm = DeepSeekLLM(backend, samples_per_prompt=samples_per_prompt)
    db = ProgramsDatabase(num_islands=num_islands, data_dir=data_dir)
    sampler = Sampler(db, llm)

    def evaluate(program: Program) -> tuple[float, ScoresPerTest]:
        """编译 + 评测，返回 (score, scores_per_test)。"""
        if _calls_ancestor(program.code):
            return 0.0, {"error": 0.0}

        fn = compile_update_function(program.code)
        if fn is None:
            return 0.0, {"error": 0.0}

        scores_per_test = evaluate_update_fn_per_test(
            fn, dataset_name=eval_dataset, dim=dim, alpha=alpha,
            eta=eta, n_rounds=eval_rounds, max_questions=eval_max_questions,
        )
        from src.funsearch.database import _reduce_score
        score = _reduce_score(scores_per_test)
        return score, scores_per_test

    # 初始化种子
    if not db.all_programs:
        print("Initializing with baseline...")
        seed = Program(id="seed-000", code=INITIAL_IMPLEMENTATION.strip())
        score, spt = evaluate(seed)
        seed.score = score
        db.register(seed, island_id=None, scores_per_test=spt)
        db.save()
        print(f"  Baseline P@1: {score:.1%}")

    print(f"\nFunSearch v2 ({max_iterations} iters, {num_islands} islands)")
    best_ever = db.get_best()

    for iteration in range(1, max_iterations + 1):
        t0 = time.time()

        # 采样
        try:
            sample_results = await sampler.sample()
        except Exception as e:
            print(f"  Iter {iteration}: sample error: {e}")
            continue

        successes = 0
        for raw_code, prompt in sample_results:
            # 截取函数体 + 编译 + 评测
            from src.funsearch.sandbox import extract_function_body
            body = extract_function_body(raw_code)
            if body is None:
                continue

            full_code = f"def dgd_update(M, key, target, dim, alpha, eta):\n{body}"
            prog = Program(
                id=f"gen{iteration:03d}-island{prompt.island_id}-{successes:03d}",
                code=full_code,
                island_id=prompt.island_id,
                generation=iteration,
            )

            score, spt = evaluate(prog)
            if score <= 0:
                continue

            prog.score = score
            db.register(prog, island_id=prompt.island_id, scores_per_test=spt)
            successes += 1

            if best_ever is None or score > best_ever.score:
                best_ever = prog
                print(f"  NEW BEST: {score:.1%} ({prog.id})")

        elapsed = time.time() - t0
        if iteration % 5 == 0:
            island_bests = [f"{isl.best_score:.1%}" for isl in db.islands]
            print(f"  Iter {iteration:3d}: {successes} valid, "
                  f"best={best_ever.score:.1%}, islands=[{', '.join(island_bests)}] ({elapsed:.0f}s)")
            db.snapshot()
            db.save()

    db.snapshot()
    db.save()
    return db.get_best()
```

---

### 6. Config（`config.py` vs 缺失）

#### Google 实现

```python
@dataclasses.dataclass(frozen=True)
class ProgramsDatabaseConfig:
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60  # 4 小时
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000

@dataclasses.dataclass(frozen=True)
class Config:
    programs_database: ProgramsDatabaseConfig = ...
    num_samplers: int = 15
    num_evaluators: int = 140
    samples_per_prompt: int = 4
```

#### 我们的实现

没有独立的 config 文件，参数散落在各处：
- `ProgramsDatabase.__init__` 中：`num_islands=5, reset_interval=50`
- `Island.__init__` 中：`temperature_init=0.1, temperature_period=100`
- `run_funsearch` 参数中：`samples_per_iteration=2, eval_rounds=3` 等

#### 修改方案

创建 `src/funsearch/config.py`：

```python
"""FunSearch 实验配置。"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class DatabaseConfig:
    """岛模型数据库配置。"""
    functions_per_prompt: int = 2
    num_islands: int = 5
    reset_interval: int = 50  # 每 N 次注册重置一次（我们用计数而非时间）
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000  # 对齐 Google（我们之前是 100，太小了）


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """评测配置。"""
    dataset_name: str = "LoCoMo-full-all"
    dim: int = 256
    alpha: float = 1.0
    eta: float = 0.01
    n_rounds: int = 3
    max_questions: int = 500
    timeout_seconds: int = 60


@dataclasses.dataclass(frozen=True)
class Config:
    """FunSearch 实验完整配置。"""
    database: DatabaseConfig = dataclasses.field(default_factory=DatabaseConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)
    samples_per_prompt: int = 4
    max_iterations: int = 100
    data_dir: str = "experiments/funsearch"
```

---

## 优先级排序

### P0 — 必须（核心算法正确性）

| # | 任务 | 影响 | 工作量 |
|---|------|------|--------|
| 1 | **创建 `code_manipulation.py`** | 所有组件的基础，AST 解析、tokenizer 重命名 | 中（移植 Google 代码） |
| 2 | **引入 Signature 分簇** | 当前 score bucket 分簇丢失了行为多样性信息 | 中 |
| 3 | **将 Prompt 构建移入 Database/Island** | 当前职责混乱，prompt 质量差（regex vs tokenizer） | 中 |
| 4 | **多测试用例评分 `evaluate_update_fn_per_test`** | 与 Signature 分簇联动，没有它 Signature 退化为单值 | 高 |

### P1 — 重要（算法效果）

| # | 任务 | 影响 | 工作量 |
|---|------|------|--------|
| 5 | **temperature_period 从 100 改为 30_000** | 当前 100 太小，温度衰减过快导致过早收敛 | 极低 |
| 6 | **_calls_ancestor 检查** | 防止生成递归调用祖先版本的无效代码 | 低 |
| 7 | **重构 runner.py 使用新接口** | 串联所有改动 | 中 |
| 8 | **创建 config.py** | 参数集中管理 | 低 |

### P2 — 可选（工程质量）

| # | 任务 | 影响 | 工作量 |
|---|------|------|--------|
| 9 | Sampler/LLM 抽象类 | 便于换后端 | 低 |
| 10 | num_islands 从 5 增到 10 | 更多多样性（Google 默认 10） | 极低 |
| 11 | `_sample_to_program` 完整流程 | 截取 + 重命名 + 模板替换一体化 | 中 |
| 12 | samples_per_prompt 从 2 增到 4 | 更多探索（Google 默认 4） | 极低 |

### 不改的部分（我们的优势）

- **持久化 + 断点恢复**：Google 没有，我们有 JSON 存盘/加载
- **快照 + 监控**：`snapshot()` 记录历史，Google 没有
- **AST 安全检查**：`_check_ast_safety` 防止恶意代码
- **受限执行环境**：`compile_update_function` 的 `safe_globals`
- **评测细节记录**：`eval_details` 保存完整评测结果

---

## 实施路线图

```
Phase 1 (P0, 预计 2-3 小时):
  ├── 创建 src/funsearch/code_manipulation.py
  ├── 创建 src/funsearch/config.py
  ├── 修改 database.py: Signature 分簇 + Prompt 构建 + 新 register 接口
  └── 修改 evaluator.py: evaluate_update_fn_per_test

Phase 2 (P1, 预计 1-2 小时):
  ├── 修改 sampler.py: 简化为 LLM 调用器
  ├── 修改 runner.py: 使用新组件接口
  ├── 修改 sandbox.py: 新增 _trim_function_body + sample_to_program
  └── 调整参数: temperature_period=30000, num_islands=10

Phase 3 (P2, 按需):
  └── 进一步重构和优化
```

---

## 关键参数对照表

| 参数 | Google 默认 | 我们当前 | 建议 |
|------|-----------|---------|------|
| `num_islands` | 10 | 5 | 10 |
| `functions_per_prompt` | 2 | 2 | 2（一致） |
| `reset_period` | 4h（基于时间） | 50（基于计数） | 保持计数制，改为 200 |
| `cluster_sampling_temperature_init` | 0.1 | 0.1 | 0.1（一致） |
| `cluster_sampling_temperature_period` | 30,000 | 100 | **30,000**（关键差异！） |
| `num_samplers` | 15 | 1 | 1（单机足够） |
| `num_evaluators` | 140 | 1 | 1（单机足够） |
| `samples_per_prompt` | 4 | 2 | 4 |
