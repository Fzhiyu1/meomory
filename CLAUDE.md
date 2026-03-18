# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

始终使用中文交流。Git 提交信息不加 AI 署名。

## Setup & Commands

```bash
# 安装
pip install -e .

# 测试
python -m pytest tests/ -v
python -m pytest tests/test_dgd.py -v          # 单个测试文件

# FunSearch 进化（主线）
python scripts/run_funsearch.py \
  --ollama-host http://localhost:11434 \
  --deepseek-key sk-xxx \
  --iterations 0 --rounds 10 --samples 6

# 基准测试
python scripts/run_bench.py experiments/configs/smoke-test.yaml

# 实时仪表盘
python scripts/dashboard.py
```

## Architecture

**meomory = 可学习的关联记忆假肢**。核心思路：用 DGD（Delta Gradient Descent）在线学习替代静态向量检索，每次反馈后自动调整权重矩阵，越用越准。

### 记忆层级

- **L0**: 系统 prompt（零检索成本）
- **L1**: DGD 关联记忆（O(1) 矩阵乘法，`src/dgd.py`）
- **L2**: 向量搜索回退（`src/store.py`）
- **L3**: 冷存储（原始 session 归档）

### 核心模块 (src/)

| 模块 | 职责 |
|------|------|
| `dgd.py` | DGD 关联记忆：M 矩阵维护、query/update、在线学习 |
| `projection.py` | 随机投影 4096→256 维（Johnson-Lindenstrauss） |
| `embedder.py` | Ollama embedding 客户端（qwen3-embedding） |
| `store.py` | brute-force cosine 向量存储 |
| `matcher.py` | 记忆匹配 + 阈值门控 |
| `feedback.py` | 自动反馈（观察 LLM 回复推断记忆使用） |
| `compressor.py` | L2→L1→L0 层级压缩 |

### FunSearch 进化框架 (src/funsearch/)

基于 Google DeepMind FunSearch 论文的群岛进化模型，用 LLM 生成代码变异，自动发现更优的 DGD update rule。

| 模块 | 职责 |
|------|------|
| `runner.py` | 主循环：多模型混合采样（Ollama 60% + DeepSeek 30% + GPT-5.4 10%），无限循环，每步持久化 |
| `sampler.py` | LLM 采样：版本化 prompt 构建、AST 提取/重命名 |
| `evaluator.py` | numpy 加速评测：随机 offset 防过拟合、P@1/3/5 指标、分轮评测 |
| `sandbox.py` | 代码沙箱：AST 安全检查（禁 import/exec/eval）、编译验证 |
| `database.py` | 种群数据库：Signature 聚类、岛间迁移、softmax 温度采样 |
| `specification.py` | 进化目标：AssociativeMemory 类的初始实现和 prompt 模板 |

**关键设计**：每次评测用不同的随机 500 题子集（random_offset），NEW BEST 自动触发 1976 题全量验证，防止子集过拟合。

### Benchmark 框架 (src/bench/)

- `datasets.py` — 加载 LoCoMo (1976题) / LongMemEval (470题) / MemoryBench
- `methods.py` — 检索方法：CosineMethod / BM25Method / DGDGTMethod / DGDLLMJudgeMethod
- `runner.py` — 实验运行器：信号量并发、检查点续跑
- `backends.py` — LLM 后端：DeepSeek API / Ollama 本地

### Mem0 插件 (src/mem0_integration/)

- `dgd_reranker.py` — 实现 Mem0 BaseReranker 接口，内嵌进化出的 EvolvedAssociativeMemory

## Environment Variables

```bash
MEOMORY_EMBED_URL=http://100.117.243.72:11435   # Ollama embedding 服务
MEOMORY_EMBED_MODEL=qwen3-embedding
MEOMORY_LLM_KEY=sk-xxx                          # DeepSeek 或 GPT-5.4 key
MEOMORY_LLM_URL=https://api.deepseek.com/v1
```

## Experiment Config Pattern

```yaml
experiments:
  - name: smoke-cosine-locomo0
    dataset: LoCoMo-full-0
    method: cosine
  - name: smoke-dgd-judge-locomo0
    dataset: LoCoMo-full-0
    method: dgd_llm_judge
    dgd: { dim: 256, alpha: 1.0, eta: 0.01 }
    rounds: 1
    llm_backend:
      type: deepseek
      url: https://api.deepseek.com/v1
      key: ${MEOMORY_LLM_KEY}
```

## Key Data

- `experiments/funsearch-v4/` — 当前活跃进化种群（population.json / best.json / history.json）
- `experiments/results/` — 基准测试结果
- `experiments/cache/` — embedding 缓存（避免重复调用）
- `data/benchmarks/` — 2446 道评测题（LoCoMo + LongMemEval + MemoryBench）
