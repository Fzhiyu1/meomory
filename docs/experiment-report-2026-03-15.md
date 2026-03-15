# meomory 实验报告 — 2026-03-15/16

## 实验目标

验证两个核心假设：
1. DGD 关联记忆能否通过反馈学习，持续提升检索准确率（超越 cosine baseline）
2. 自动反馈机制能否产生足够有效的学习信号（无需人工标注）

## 实验环境

| 组件 | 实验 v1 | 实验 v2 |
|------|---------|---------|
| 记忆片段 | Big Bang Theory 805 session | 同左 |
| Embedding | qwen3-embedding 8B Q4_K_M（Ollama, RTX 4060 GPU） | 同左 |
| Agent 模型 | GPT-5.4（cliproxy 多跳链路） | DeepSeek-chat（直连 API） |
| 反馈机制 | embedding cosine > 0.5 | **LLM 判断者**（DeepSeek 判断引用关系） |
| 执行方式 | 串行（每题顺序执行） | **并行**（14 题并发） |
| 测试题 | MemoryBench DialSim-bigbang，14 道有效题 | 同左 |
| DGD 参数 | dim=256, α=1.0, η=0.01 | 同左 |

---

## 实验一：DGD + Ground Truth 反馈

**设计**：每轮将 14 个查询的正确答案片段直接告诉 DGD 更新 M。20 轮。理想上界——反馈 100% 正确。

**结果**：

| 轮次 | P@1 | P@3 | P@5 |
|------|-----|-----|-----|
| Cosine baseline | 57.1% | 71.4% | 78.6% |
| Round 1 | 64.3% | 71.4% | 78.6% |
| Round 2 | 71.4% | 71.4% | 78.6% |
| Round 4 | 64.3% | 85.7% | 85.7% |
| Round 10 | 71.4% | 85.7% | 85.7% |
| Round 18 | 78.6% | 78.6% | 85.7% |
| Round 20 | **78.6%** | **85.7%** | **92.9%** |

**结论**：
- P@1：57.1% → 78.6%（**+21.5pp**）
- P@5：78.6% → 92.9%（14 题中 13 题在前 5 名找到）
- 学习曲线整体上升但有震荡
- **假设一成立**：DGD 在理想反馈条件下持续超越 cosine

---

## 实验二（v1）：DGD + Embedding 自动反馈

**设计**：embedding(agent回复) vs embedding(注入记忆) cosine > 0.5 判定为"被使用"。GPT-5.4 作为 agent。

**问题**：GPT-5.4 链路不稳定（Mac → VPS → Tailscale → OpenClaw → mihomo → OpenAI），超时率 10-30%。实验只完成 8/20 轮。

**结果（不完整）**：

| 轮次 | P@1 | 反馈准确率 | 超时数 |
|------|-----|-----------|--------|
| Round 1 | 57.1% | 68% | 1/14 |
| Round 2 | 64.3% | 65% | 4/14 |
| Round 4 | 64.3% | 68% | 1/14 |
| Round 6 | 71.4% | 60% | 2/14 |
| Round 8 | 64.3% | 59% | 1/14 |

**结论**：
- 反馈准确率 ~60-68%，方向正确但噪声大
- P@1 有上升趋势但震荡（57→71→64）
- 超时导致数据不完整，**实验不可靠**

### 改用 DeepSeek（v1.5，仍用 embedding 反馈）

**结果（不完整，3 轮）**：

| 轮次 | P@1 | 反馈准确率 |
|------|-----|-----------|
| Round 1 | 57.1% | 55% |
| Round 2 | 71.4% | 48% |
| Round 4 | 57.1% | 50% |

**结论**：
- 零超时，但反馈准确率降到 ~50%（接近随机）
- DeepSeek 更喜欢改写而非引用，导致 embedding 相似度判断失效
- P@1 震荡不收敛
- **embedding 相似度作为反馈信号的瓶颈被确认**

---

## 实验三（v2）：DGD + LLM 判断者反馈 ⭐

**设计**：将反馈机制从"embedding 相似度"升级为"LLM 判断者"。让 DeepSeek 判断"agent 回复实际引用了哪些注入的记忆"。14 题并发执行。

**反馈 prompt**：
```
Given injected memories and agent's response, determine which memories
the response actually referenced. Paraphrasing counts. Same topic is NOT
enough — must use specific information. Output JSON array of indices.
```

**结果（完整，20 轮）**：

| 轮次 | P@1 | P@3 | P@5 | 更新数 | 反馈准确率 | 耗时 |
|------|-----|-----|-----|--------|-----------|------|
| Round 1 | 57.1% | 71.4% | 71.4% | 8 | 62% | 9.3s |
| Round 2 | 57.1% | 71.4% | 78.6% | 8 | **88%** | 9.2s |
| Round 4 | 64.3% | 71.4% | 78.6% | 8 | 75% | 8.9s |
| Round 6 | 64.3% | 71.4% | 78.6% | 6 | 67% | 9.5s |
| Round 8 | 64.3% | 78.6% | 85.7% | 10 | 70% | 10.0s |
| Round 10 | 64.3% | 78.6% | 85.7% | 8 | 75% | 10.0s |
| Round 12 | **71.4%** | 78.6% | 85.7% | 6 | 67% | 8.6s |
| Round 14 | 71.4% | 78.6% | 85.7% | 9 | 78% | 9.1s |
| Round 16 | 71.4% | 78.6% | 85.7% | 8 | 62% | 10.8s |
| Round 18 | 71.4% | 78.6% | 85.7% | 9 | 78% | 12.2s |
| Round 20 | **71.4%** | 71.4% | **85.7%** | 6 | **83%** | 10.2s |

**学习曲线对比（P@1）**：

```
Round    Cosine     GT        Auto(Judge)
─────    ──────     ──        ───────────
1        57.1%      64.3%     57.1%
4        57.1%      64.3%     64.3%
8        57.1%      64.3%     64.3%
12       57.1%      71.4%     71.4%  ← 追平 GT 同期
16       57.1%      71.4%     71.4%
20       57.1%      78.6%     71.4%
```

**结论**：
- P@1：57.1% → 71.4%（**+14.3pp**），学习曲线**稳步上升，无震荡**
- P@5：71.4% → 85.7%（**+14.3pp**）
- 反馈准确率平均 ~72%（vs embedding 方式的 50%），**LLM 判断者显著优于 embedding 相似度**
- 第 12 轮追平 ground truth 同期水平（两者都是 71.4%）
- 零超时、零错误，每轮 9-10 秒（14 题并发）
- **假设二成立**：LLM 判断者产生的自动反馈足以驱动 DGD 持续学习

---

## 总结对比

| 方法 | P@1 (最终) | 相比 cosine | 反馈准确率 | 稳定性 |
|------|-----------|------------|-----------|--------|
| Cosine baseline | 57.1% | — | N/A | 固定不变 |
| DGD + Ground Truth | **78.6%** | +21.5pp | 100% | 有震荡 |
| DGD + Embedding 反馈 (GPT-5.4) | ~64-71% | +7-14pp | 60-68% | 不稳定，超时严重 |
| DGD + Embedding 反馈 (DeepSeek) | ~57-71% | +0-14pp | 48-55% | 震荡不收敛 |
| **DGD + LLM Judge (DeepSeek)** | **71.4%** | **+14.3pp** | **~72%** | **稳步上升** |

## 关键发现

### 已验证
1. **DGD 能学习**：ground truth 下 P@1 +21.5pp，LLM judge 下 +14.3pp
2. **LLM 判断者 >> embedding 相似度**：反馈准确率 72% vs 50%，学习曲线稳定 vs 震荡
3. **自动反馈闭环可行**：无需人工标注，LLM 观察 agent 行为即可产生有效学习信号
4. **并行执行有效**：14 题并发，每轮 9-10 秒，20 轮总共 ~3 分钟

### 瓶颈已定位
1. ~~DGD 能不能学~~ → 能学（已验证）
2. ~~自动反馈信号够不够~~ → embedding 不够，LLM 判断者够（已验证）
3. **测试集太小**：14 题，统计显著性不足（仍然是问题）
4. **DGD 长期稳定性**：20 轮后是否退化？需要更多轮次验证

### 反馈机制演进

```
v0: 人工标注 "哪条有用"          → 准确但不可扩展
v1: embedding cosine > 0.5     → 50% 准确率，不够
v2: LLM 判断者                  → 72% 准确率，足够驱动学习 ✓
```

## 局限性

1. **14 题太少** — 每次 P@1 变化都是 ±1 题（7.1%），需要更大测试集
2. **单一数据集** — 只在 Big Bang Theory 上测过
3. **未部署到真实 agent** — 所有结果来自离线 benchmark
4. **DGD 参数未调优** — α=1.0 (无遗忘), η=0.01 是随手设的
5. **LLM 判断者成本** — 每次反馈需要一次额外 LLM 调用

## 下一步建议

1. **加大测试集**：LongMemEval（500 题）或自造更大测试集
2. **Gateway 集成**：接入 OpenClaw 在真实对话中验证
3. **CMS 多时间尺度**：短/中/长期三层 DGD
4. **判断者异步化**：反馈不阻塞主流程，后台更新 M
5. **对比更强 baseline**：BM25、reranker、RAG + rerank
6. **DGD 参数搜索**：α、η、dim 的最优组合

## 文件索引

| 文件 | 内容 |
|------|------|
| `scripts/bench_memorybench.py` | 实验一：ground truth 评测 |
| `scripts/bench_auto_feedback.py` | 实验二 v1：embedding 自动反馈 |
| `scripts/bench_v2.py` | **实验三 v2：LLM 判断者 + 并行** |
| `scripts/bench_dgd.py` | 补充实验：简单 benchmark |
| `data/benchmarks/memorybench/bench_v2_full_log.txt` | v2 完整运行日志 |
| `data/benchmarks/memorybench/learning_curve_v2.json` | v2 学习曲线数据 |
| `data/benchmarks/memorybench/auto_feedback_partial_log.txt` | v1 部分日志 |
| `data/benchmarks/memorybench/deepseek_partial_log.txt` | DeepSeek embedding 反馈部分日志 |
| `docs/architecture.md` | 完整架构文档 |
