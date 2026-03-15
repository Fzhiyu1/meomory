# meomory 大规模实验方案

## 目标

在多数据集、多 baseline 上验证 DGD + LLM Judge 的有效性，并在 OpenClaw 真实场景中部署验证。

## 三条实验线

### 实验线 A：多数据集 benchmark

**数据集（MemoryBench 11 个子集）：**

| 数据集 | 类型 | 语言 |
|--------|------|------|
| LoCoMo-0 ~ LoCoMo-9 | 长期对话记忆 QA | en |
| DialSim-bigbang | 剧集对话 QA | en |
| DialSim-friends | 剧集对话 QA | en |
| DialSim-theoffice | 剧集对话 QA | en |
| NFCats | 非事实类问答 | en |

**评测方法：**
- 对每个数据集：语料切片 → embedding → cosine baseline → DGD + GT → DGD + LLM Judge
- 20 轮反馈，每 2 轮评测
- 记录完整 log：每轮每题的 top-10 排名、反馈判断、DGD 更新

**产出：**
- 每个数据集的学习曲线
- 跨数据集汇总：DGD 平均提升多少 pp
- 统计检验：配对 t 检验或 Wilcoxon 检验

### 实验线 B：多 baseline 对比

在 DialSim-bigbang（已有数据）+ 一个 LoCoMo 子集上：

| Baseline | 实现 |
|----------|------|
| BM25 | 关键词匹配（用 rank_bm25 库或手写） |
| Cosine (qwen3-embedding) | 已有 |
| Cosine + Reranker | 先 cosine top-20，再用 LLM rerank |
| DGD + LLM Judge | 已有 |
| DGD + LLM Judge + Reranker | DGD top-10 → LLM rerank |

**产出：**
- 各 baseline 的 P@1/3/5/10
- DGD 是否优于成熟的 reranker 方案

### 实验线 C：OpenClaw 真实场景

**部署：**
1. 在 OpenClaw gateway 加前意识激活层中间件
2. 用知识库 53 篇文档 + 历史 session 压缩片段作为记忆库
3. 每次飞书消息进来 → DGD 匹配 → 注入 → agent 回复 → LLM Judge 反馈 → 更新

**观测指标：**
- DGD M 矩阵的变化轨迹（每天快照）
- 匹配命中率随时间的变化
- 用户（你）的主观感受：agent 是不是越来越"懂你"

**周期：** 7 天

**产出：**
- 真实场景下的学习曲线
- DGD 长期稳定性验证

## 硬件分工

| 机器 | 角色 | 任务 |
|------|------|------|
| Yilong15Pro (4060, 8GB) | Embedding 服务 | qwen3-embedding，已就位 |
| 5070 机器 | 本地 LLM | 跑 agent 模型 + 判断者模型（零延迟） |
| Mac (M-series, 48GB) | 控制器 | 调度实验、DGD 计算、结果收集 |

## 执行顺序

```
Phase 1: 基础设施（今天）
  ├── 5070 机器：装 Ollama，拉模型（Qwen2.5-14B 或类似）
  ├── Mac：写实验框架（多数据集加载、多 baseline 运行、log 收集）
  └── 验证三台机器联通

Phase 2: 实验线 A（多数据集）
  ├── 下载所有 MemoryBench 子数据集
  ├── 逐个跑：cosine → DGD + GT → DGD + LLM Judge
  └── 汇总结果

Phase 3: 实验线 B（多 baseline）
  ├── 实现 BM25 baseline
  ├── 实现 LLM reranker
  ├── 对比实验
  └── 汇总结果

Phase 4: 实验线 C（真实部署）
  ├── OpenClaw gateway 中间件
  ├── 部署 + 跑 7 天
  └── 分析结果
```

## Log 规范

每次实验运行记录：

```json
{
  "experiment_id": "exp-001",
  "dataset": "DialSim-bigbang",
  "method": "dgd_llm_judge",
  "timestamp": "2026-03-16T...",
  "config": {
    "dgd_dim": 256,
    "dgd_alpha": 1.0,
    "dgd_eta": 0.01,
    "n_rounds": 20,
    "embedding_model": "qwen3-embedding",
    "agent_model": "qwen2.5-14b",
    "judge_model": "qwen2.5-14b"
  },
  "rounds": [
    {
      "round": 1,
      "p_at_1": 0.571,
      "p_at_3": 0.714,
      "p_at_5": 0.786,
      "total_updates": 8,
      "correct_updates": 5,
      "feedback_accuracy": 0.625,
      "elapsed_seconds": 9.3,
      "details": [
        {
          "question_idx": 0,
          "top_10_ids": [152, 146, 314, ...],
          "answer_ids": [146, 152],
          "rank": 1,
          "injected": [152, 146, 314],
          "judge_said_used": [0, 1],
          "correct_feedback": true
        }
      ]
    }
  ]
}
```
