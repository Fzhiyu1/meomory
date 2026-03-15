# 统一实验框架 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 搭建一个配置驱动的实验框架，一条命令跑完所有 数据集×方法 的对比，输出标准化 log 和汇总表。

**Architecture:** `src/bench/` 模块，核心三个抽象：Dataset（数据集加载）、Method（检索方法）、LLMBackend（模型后端）。配置用 YAML，结果输出 JSON log + 终端汇总表。

**Tech Stack:** Python 3.14, httpx (async), PyYAML, pyarrow (读 MemoryBench)

---

### Task 1: 项目依赖更新 + 目录结构

**Files:**
- Modify: `pyproject.toml` — 加 pyyaml 依赖
- Create: `src/bench/__init__.py`
- Create: `src/bench/config.py`
- Create: `experiments/` — 实验配置和结果目录

**步骤：**
1. pyproject.toml 加 `pyyaml` 到 dependencies
2. 创建目录结构：
```
src/bench/
  __init__.py
  config.py       # 实验配置加载
  datasets.py     # 数据集抽象
  methods.py      # 检索方法抽象
  backends.py     # LLM 后端抽象
  runner.py       # 实验运行器
  evaluate.py     # 评测指标
experiments/
  configs/        # YAML 配置文件
  results/        # JSON 结果
```
3. pip install pyyaml
4. Commit

---

### Task 2: LLM Backend 抽象

**Files:**
- Create: `src/bench/backends.py`

**设计：**
```python
class LLMBackend:
    async def chat(self, prompt: str, system: str = "", max_tokens: int = 300) -> str: ...

class DeepSeekBackend(LLMBackend):
    """直连 DeepSeek API"""
    def __init__(self, api_key, model="deepseek-chat"): ...

class OllamaBackend(LLMBackend):
    """连接 Ollama（5070 或 4060）"""
    def __init__(self, host, model="qwen3.5:9b"): ...
```

两个 Ollama 地址：
- 4060 (embedding): `http://100.117.243.72:11435`
- 5070 (chat): `http://192.168.1.5:11434`（需要通过 OpenClaw 转发或 SSH 隧道）

注意：5070 在局域网，Mac 不能直连。需要建 SSH 隧道：
```bash
ssh -fN -L 11434:192.168.1.5:11434 openclaw
```
然后本地用 `http://127.0.0.1:11434`

**步骤：**
1. 建 SSH 隧道到 5070 的 Ollama
2. 实现 backends.py
3. 测试两个后端都能 chat
4. Commit

---

### Task 3: Dataset 抽象

**Files:**
- Create: `src/bench/datasets.py`

**设计：**
```python
class Dataset:
    name: str
    fragments: list[dict]       # {id, body, full}
    questions: list[dict]       # {question, golden_answer, answer_indices}

    @classmethod
    def load_dialsim(cls, name: str) -> "Dataset":
        """加载 DialSim 系列（bigbang/friends/theoffice）"""

    @classmethod
    def load_locomo(cls, index: int) -> "Dataset":
        """加载 LoCoMo 系列"""
```

每个 Dataset 统一提供：fragments（记忆片段）和 questions（带 ground truth 的查询）。

需要下载的数据：
- DialSim-friends corpus + test
- DialSim-theoffice corpus + test（如果有）
- LoCoMo-0 test

**步骤：**
1. 下载缺失的数据集
2. 实现 datasets.py
3. 验证每个数据集能加载（打印片段数和题数）
4. Commit

---

### Task 4: Method 抽象

**Files:**
- Create: `src/bench/methods.py`

**设计：**
```python
class Method:
    name: str
    def setup(self, fragments, proj_vecs): ...
    def query(self, q_vec) -> list[tuple[float, int]]: ...  # [(score, frag_idx)]
    async def feedback(self, q_vec, response, top_indices, answer_indices, backend): ...

class CosineMethod(Method):
    """静态 cosine，不学习"""

class BM25Method(Method):
    """BM25 关键词匹配，不学习"""

class DGDGroundTruthMethod(Method):
    """DGD + ground truth 反馈"""

class DGDJudgeMethod(Method):
    """DGD + LLM 判断者反馈"""
```

BM25 实现：纯 Python，TF-IDF 变体，不需要外部库。

**步骤：**
1. 实现四个 Method
2. 单元测试：每个 method 的 query 和 feedback
3. Commit

---

### Task 5: 实验运行器

**Files:**
- Create: `src/bench/runner.py`
- Create: `src/bench/evaluate.py`

**设计：**
```python
async def run_experiment(config: dict) -> dict:
    """一次完整实验：加载数据 → embedding → N 轮 → 评测 → 输出"""

    dataset = Dataset.load(config["dataset"])
    method = create_method(config["method"])
    backend = create_backend(config["llm_backend"])

    # Embedding
    frag_vecs, q_vecs, proj_vecs = embed_all(dataset, config)

    # N 轮
    for round in range(config["rounds"]):
        for q in dataset.questions:
            results = method.query(q_vec)
            await method.feedback(q_vec, ..., backend)
        evaluate_and_log(round, ...)

    return full_log
```

**步骤：**
1. 实现 runner.py（核心循环）
2. 实现 evaluate.py（P@K 计算 + log 写入）
3. Commit

---

### Task 6: 配置文件 + CLI 入口

**Files:**
- Create: `experiments/configs/dialsim-bigbang-all.yaml`
- Create: `scripts/run_bench.py`

**配置示例：**
```yaml
# 一个文件定义多组实验
experiments:
  - name: cosine-bigbang
    dataset: DialSim-bigbang
    method: cosine
    rounds: 1
    embedding_backend:
      type: ollama
      host: "http://100.117.243.72:11435"
      model: qwen3-embedding
    llm_backend: null  # cosine 不需要 LLM

  - name: dgd-judge-bigbang
    dataset: DialSim-bigbang
    method: dgd_llm_judge
    rounds: 20
    embedding_backend:
      type: ollama
      host: "http://100.117.243.72:11435"
      model: qwen3-embedding
    llm_backend:
      type: ollama
      host: "http://127.0.0.1:11434"  # SSH 隧道到 5070
      model: qwen3.5:9b
    dgd:
      dim: 256
      alpha: 1.0
      eta: 0.01
```

**CLI：**
```bash
# 跑一个配置文件里的所有实验
python3 scripts/run_bench.py experiments/configs/dialsim-bigbang-all.yaml

# 只跑其中一个
python3 scripts/run_bench.py experiments/configs/dialsim-bigbang-all.yaml --only cosine-bigbang
```

**步骤：**
1. 写配置文件
2. 实现 CLI 入口
3. 端到端验证：跑一个 cosine 实验
4. Commit

---

### Task 7: 首轮全量实验

**运行矩阵：**

| 数据集 | cosine | BM25 | DGD+GT | DGD+Judge |
|--------|--------|------|--------|-----------|
| DialSim-bigbang | ✅ | ✅ | ✅ | ✅ |
| DialSim-friends | ✅ | ✅ | ✅ | ✅ |
| LoCoMo-0 | ✅ | ✅ | ✅ | ✅ |

3 数据集 × 4 方法 = 12 次实验

**步骤：**
1. 写 3 个配置文件（每个数据集一个）
2. 依次跑（或并行跑不同数据集）
3. 汇总结果到一张表
4. 保存所有 log 到 experiments/results/
5. Commit 结果
6. 更新实验报告

---

## 完成标准

- [ ] `python3 scripts/run_bench.py config.yaml` 能一键跑完
- [ ] 每次实验输出标准 JSON log
- [ ] 终端打印汇总表（各方法 P@1/3/5）
- [ ] 12 次实验全部完成
- [ ] 结果更新到 experiment report
