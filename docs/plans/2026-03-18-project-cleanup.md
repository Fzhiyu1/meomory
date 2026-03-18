# meomory 项目整理计划

## 整理原则

1. **不删除任何实验数据**——归档，不删除
2. **核心代码保持不动**——只移动辅助文件
3. **一次性完成**——不分阶段，现在做完

---

## 一、目录结构（整理后）

```
meomory/
├── src/
│   ├── dgd.py                    # 核心：原始 DGD 关联记忆
│   ├── embedder.py               # 核心：embedding 客户端
│   ├── projection.py             # 核心：随机投影
│   ├── store.py                  # 核心：向量存储
│   ├── matcher.py                # 核心：记忆匹配
│   ├── writer.py                 # 核心：记忆写入
│   ├── consolidator.py           # 核心：巩固压缩
│   ├── compressor.py             # 核心：L2→L1→L0 压缩
│   ├── feedback.py               # 核心：自动反馈
│   ├── llm_client.py             # 核心：LLM API
│   ├── parser.py                 # 核心：会话解析
│   ├── bench/                    # 基准测试框架（保持不动）
│   ├── funsearch/                # FunSearch 进化框架（主线）
│   ├── evolution/                # Prompt 进化（已完成，保留代码）
│   └── mem0_integration/         # Mem0 插件
│
├── scripts/
│   ├── run_funsearch.py          # 主入口：FunSearch 进化
│   ├── run_bench.py              # 主入口：基准测试
│   ├── run_evolution.py          # 主入口：Prompt 进化 V1
│   ├── run_evolution_v2.py       # 主入口：Prompt 进化 V2
│   ├── run_pipeline.py           # 主入口：完整 pipeline
│   ├── dashboard.py              # 主入口：实时仪表盘
│   └── legacy/                   # 归档的旧脚本
│       ├── bench_v2.py
│       ├── bench_dgd.py
│       ├── bench_auto_feedback.py
│       ├── bench_memorybench.py
│       ├── bench_kb_consolidation.py
│       ├── bench_write_consolidate.py
│       ├── gen_report_large.py
│       ├── demo_cycle.py
│       ├── train_dgd.py
│       ├── sleep_consolidation.py
│       ├── recompress_llm.py
│       ├── index_knowledge_base.py
│       └── test_mem0_integration.py
│
├── experiments/
│   ├── funsearch-v4/             # 当前活跃（保持不动）
│   ├── results/                  # 实验结果（保持不动）
│   ├── cache/                    # embedding 缓存（保持不动）
│   ├── configs/                  # 实验配置（保持不动）
│   ├── logs/                     # 实验日志（保持不动）
│   ├── status.json               # 保持不动
│   └── archived/                 # 归档的旧进化数据
│       ├── evolution-v1/
│       ├── evolution-v2/
│       ├── funsearch-v1/
│       ├── funsearch-v2/
│       ├── funsearch-smoke/
│       └── funsearch-class-smoke/
│
├── docs/
│   ├── architecture.md           # 保持
│   ├── experiment-report-2026-03-17.md  # 主报告
│   ├── report-large-scale.html   # 主 HTML 报告
│   ├── blog-draft.md             # 已发布的博客
│   ├── blog-cover.png            # 博客封面
│   ├── plans/                    # 保持不动
│   └── archived/                 # 归档的旧文档
│       ├── experiment-report-2026-03-15.md
│       ├── report.html
│       ├── report-day1.html
│       └── cover-philosophy.md
│
├── README.md
└── tests/
```

---

## 二、具体操作清单

### 删除（1 个文件）
- `src/tool.py` — 死代码，功能被 matcher 替代

### 移动脚本到 legacy/
```
scripts/bench_v2.py → scripts/legacy/
scripts/bench_dgd.py → scripts/legacy/
scripts/bench_auto_feedback.py → scripts/legacy/
scripts/bench_memorybench.py → scripts/legacy/
scripts/bench_kb_consolidation.py → scripts/legacy/
scripts/bench_write_consolidate.py → scripts/legacy/
scripts/gen_report_large.py → scripts/legacy/
scripts/demo_cycle.py → scripts/legacy/
scripts/train_dgd.py → scripts/legacy/
scripts/sleep_consolidation.py → scripts/legacy/
scripts/recompress_llm.py → scripts/legacy/
scripts/index_knowledge_base.py → scripts/legacy/
scripts/test_mem0_integration.py → scripts/legacy/
src/tool_dgd.py → scripts/legacy/demo_dgd_interactive.py
```

### 归档实验数据
```
experiments/evolution/ → experiments/archived/evolution-v1/
experiments/evolution-v2/ → experiments/archived/evolution-v2/
experiments/funsearch/ → experiments/archived/funsearch-v1/
experiments/funsearch-v2/ → experiments/archived/funsearch-v2/
experiments/funsearch-smoke/ → experiments/archived/funsearch-smoke/
experiments/funsearch-class-smoke/ → experiments/archived/funsearch-class-smoke/
```

### 归档文档
```
docs/experiment-report-2026-03-15.md → docs/archived/
docs/report.html → docs/archived/
docs/report-day1.html → docs/archived/
docs/cover-philosophy.md → docs/archived/
```

### 清理临时文件
```
experiments/full-reeval.log → 删除
experiments/gen070-fulltest.log → 删除
experiments/locomo-ceiling-test.log → 删除（数据已记录到报告）
experiments/locomo-learning-curve.log → 删除
experiments/noise-test.log → 删除
experiments/longmemeval-test.log → 删除
experiments/logs/smoke-test-*.log → 删除
```
