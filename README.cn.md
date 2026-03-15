# meomory

给 LLM agent 做的记忆假肢——不是记忆系统，是状态恢复系统。

LLM agent 没有记忆。每次对话从零开始。RAG 给了它一个图书馆，但图书馆不是记忆——不学习、不遗忘、同样的查询永远返回同样的结果。

meomory 用 **DGD 关联记忆**替代静态向量搜索：被使用的记忆关联被强化，没用到的自然衰减。反馈信号来自观察 agent 的行为——不需要人工标注。

## 工作原理

```
用户消息进来
  → 前意识激活层：DGD 匹配相关记忆
  → 注入到 LLM 上下文（LLM 不知道这件事发生了）
  → LLM 正常回复
  → 观察者判断哪些记忆被实际引用了
  → DGD 强化被引用的关联，其他自然衰减
  → 下次查询：排序因积累的经验而改变
```

## 实验结果

MemoryBench DialSim-bigbang（805 个片段，14 个问题，20 轮）：

| 方法 | P@1 |
|------|-----|
| Cosine（静态，永远不学） | 57.1% |
| DGD + 人工标注反馈 | 78.6%（+21.5pp） |
| **DGD + LLM 自动判断反馈** | **71.4%（+14.3pp）** |

自动反馈版本不需要人工标注——一个 LLM 观察者判断 agent 引用了哪些记忆，DGD 仅凭这个信号就能持续学习。

## 核心思想

- **记忆 ≠ 图书馆。** 图书馆存储和搜索。记忆压缩、学习、遗忘、被动激活。
- **DGD 作为独立记忆模块。** 从 Hope 架构（Nested Learning, NeurIPS 2025）中拆出来，做成外挂检索模块——这个用法是原创的。
- **自动反馈闭环。** 观察 agent 行为 → 推断哪些记忆被使用 → 更新关联。agent 不知道自己在被观察。
- **前意识激活。** 记忆在 LLM 开始思考之前就浮现了，不是它决定去搜索的。

## 快速上手

```bash
cd /Users/fangzhiyu/run/meomory
source .venv/bin/activate

# 查询知识库记忆（cosine 版）
python3 -m src.tool "主动遗忘"

# DGD 交互模式（带学习）
python3 -m src.tool_dgd

# 完整闭环 demo（匹配 → 注入 → agent 回复 → 自动反馈 → DGD 更新）
MEOMORY_LLM_URL="https://api.deepseek.com/v1" \
MEOMORY_LLM_KEY="your-key" \
MEOMORY_LLM_MODEL="deepseek-chat" \
python3 scripts/demo_cycle.py
```

## 项目结构

```
src/
  parser.py          # Session JSONL 解析器
  compressor.py      # L2→L1→L0 压缩（规则 + LLM）
  embedder.py        # qwen3-embedding 客户端（Ollama）
  store.py           # 向量存储（brute-force cosine）
  matcher.py         # 记忆匹配 + 阈值门控
  projection.py      # 随机投影 4096→256
  dgd.py             # DGD 关联记忆
  feedback.py        # 自动反馈层
  llm_client.py      # LLM API 客户端
  tool.py            # recall_memory CLI（cosine）
  tool_dgd.py        # recall_memory CLI（DGD，交互式）

scripts/
  demo_cycle.py              # 完整记忆闭环 demo
  bench_v2.py                # LLM 判断者 benchmark（并行）
  index_knowledge_base.py    # 知识库索引

docs/
  architecture.md            # 完整架构文档
  experiment-report-*.md     # 实验报告
```

## 名字

meomory = me + memory。本来是打错了。留着了。
