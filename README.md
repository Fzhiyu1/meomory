# meomory

给 AI 装上"越用越准"的记忆。不是更好的搜索，是会学习的记忆假肢。

> **博客**：[我让三个 AI 互相竞争进化，两天后它们发明了一个我看不懂的算法](https://juejin.cn/spost/7618056233687924782) | [知乎](https://zhuanlan.zhihu.com/p/2017403193101926520)

## 这是什么

LLM 智能体是"失忆的天才"——每次对话都从零开始。现有的记忆系统（RAG、向量检索）像图书馆：存了东西但永远返回一样的结果。

meomory 用**在线学习的关联记忆**替代静态检索。一个 256×256 的权重矩阵，每次检索后根据反馈自动调整——用得越多，找得越准。

**而且这个算法不是人写的，是三个 AI 模型通过进化搜索自动发明的。**

## 结果

| 方法 | LoCoMo (1976 题) | LongMemEval (470 题) |
|------|:---:|:---:|
| 向量相似度（不学习） | 22.4% | 41.7% |
| BM25 关键词匹配 | 30.9% | 48.3% |
| 手写 DGD 算法 | 32.8% | 56.0% |
| **AI 进化出的算法** | **56.6%** | **92.3%** |

LongMemEval 上 92.3% 接近当前最强系统（Hindsight+TEMPR 91.4%，EverMemOS 93.0%）——但那些系统用的是 GPT-4o + 知识图谱 + 多策略融合。我们只用一个矩阵。

## 进化过程

三个 AI 模型在服务器上 24 小时无人值守进化：

```
qwen3.5:9b (60%)  — 量产探索，发现基础策略
DeepSeek (30%)    — 精细优化，逆袭夺冠
GPT-5.4 (10%)    — 创新尖兵，首个突破
```

两小时内从 26.8% 进化到 88.4%（500 题），全量验证 56.6%。进化出的算法融合了 Hebbian 学习 (1949)、Momentum (1980s)、Adam 优化器 (2014)、BatchNorm (2015)——LLM 在没有被告知"神经网络"的情况下独立进化出了神经网络层的结构。

## 怎么用

**作为 reranker 插件（接入任何记忆系统）：**

```python
from src.mem0_integration.dgd_reranker import DGDReranker

reranker = DGDReranker({"dim": 256})

# 检索后重排序
results = your_memory_system.search(query)
reranked = reranker.rerank(query, results, top_k=5)

# 反馈：告诉它哪个记忆是对的
reranker.feedback(query, correct_memory)
# 下次同类问题排序更准
```

**跑进化搜索（发现更好的算法）：**

```bash
python scripts/run_funsearch.py \
  --ollama-host http://localhost:11434 \
  --deepseek-key sk-xxx \
  --iterations 0 --rounds 10 --samples 6
```

## 项目结构

```
src/funsearch/      — 进化框架（群岛模型 + 多模型 + 沙箱 + numpy 加速）
src/evolution/      — Prompt 进化（Judge prompt 28.9% → 90.9%）
src/mem0_integration/ — Mem0 reranker 插件
src/dgd.py          — 原始 DGD 关联记忆
src/bench/           — Benchmark 框架（65 个实验，2446 题）
scripts/dashboard.py — 实时进化仪表盘
experiments/         — 所有实验结果 + 进化种群数据
```

## 灵感来源

这个项目的灵感来自我的[认知科学知识库](https://github.com/Fzhiyu1/cognitive-network)（49 张概念卡片）。AI 读了我写的关于记忆、遗忘、前意识激活的认知理论后，自动链接到了 Hope/Nested Learning 论文 (NeurIPS 2025) 中的 DGD 算法。我之前[复现了这篇论文](https://github.com/Fzhiyu1/hope-gpt)但关注的是另一个模块——AI 帮我找到了我忽略的那个零件。

## 名字

meomory = me + memory。打错了，留下了。

---

*作者：方治宇 | 双非本科大四在读*
