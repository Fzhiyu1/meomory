# 实验报告 2026-03-17

## 概述

两条实验线：大规模 benchmark 验证（凌晨完成）+ 进化搜索（白天完成）。

---

## 一、大规模 Benchmark 实验

### 实验规模
- **65 个实验**，5 种方法 × 2 数据集 + 逐对话拆分 + 参数搜索 + 学习曲线 + DialSim 复现
- **2446 题**：LoCoMo 1976 题 + LongMemEval 470 题
- **耗时 ~3.5 小时**，~52,000 次 DeepSeek API 调用
- Embedding: qwen3-embedding via LAN (192.168.1.100:11435)
- LLM Judge: DeepSeek-chat API

### 1.1 主实验结果（3 轮）

| 方法 | LoCoMo P@1 (1976题) | LongMemEval P@1 (470题) |
|------|:---:|:---:|
| Cosine | 22.4% | 41.7% |
| BM25 | **30.9%** | 48.3% |
| DGD+GT | 26.6% | **50.0%** |
| DGD+Judge | 24.1% | 48.3% |
| DGD+CMS | 23.6% | 46.2% |

**发现：** DGD 在语义检索任务（LongMemEval）上显著优于 baseline（+8.3pp vs cosine），但在关键词匹配型任务（LoCoMo）上不如 BM25。

### 1.2 学习曲线（10 轮）

| Round | LoCoMo GT | LoCoMo Judge | LongMemEval GT | LongMemEval Judge |
|:-----:|:---------:|:------------:|:--------------:|:-----------------:|
| 1 | 24.7% | 23.7% | 48.9% | 47.2% |
| 3 | 26.6% | 24.1% | 50.0% | 48.5% |
| 5 | 28.8% | 24.4% | 52.3% | 49.4% |
| 8 | 30.9% | 24.9% | 55.1% | 50.0% |
| 10 | **32.8%** | 24.5% | **56.0%** | **50.0%** |

**发现：** DGD-GT 10 轮在两个数据集上都超过 BM25，没有饱和。Judge 在 LoCoMo 上饱和在 ~25%（feedback accuracy 33% 的天花板），在 LongMemEval 上达到 50%（超 BM25）。

### 1.3 参数搜索（大规模，1976 题）

**α 搜索：**
| α | P@1 |
|---|-----|
| 0.9 | 1.3% |
| 0.95 | 1.1% |
| 0.99 | 1.1% |
| **1.0** | **26.6%** |

死亡谷在大规模上铁证：α<1.0 全部崩到 ~1%。

**dim 搜索：**
| dim | P@1 | 耗时/轮 |
|-----|-----|---------|
| 64 | 18.7% | ~6s |
| 128 | 24.7% | ~13s |
| 256 | 26.6% | ~26s |
| 512 | 28.3% | ~90s |

dim=512 略优但慢 3.5 倍。

**η 搜索：**
| η | P@1 |
|---|-----|
| 0.005 | 25.8% |
| 0.01 | 26.6% |
| 0.02 | 29.2% |
| **0.05** | **34.4%** |

**意外发现：** η=0.05 在 GT 下最优，推翻了之前小规模实验"η=0.05 震荡"的结论。GT 无噪声时大学习率学得更快。

### 1.4 逐对话拆分（10 段 LoCoMo 对话）

| Conv | Cosine P@1 | DGD-GT P@1 | DGD-Judge P@1 |
|:----:|:----------:|:----------:|:-------------:|
| 0 | 25.7% | 30.4% | 28.3% |
| 1 | 29.5% | 31.4% | 31.4% |
| 2 | 29.0% | 35.2% | 31.1% |
| 3 | 23.5% | 21.5% | 20.0% |
| 4 | 24.4% | 29.8% | 25.6% |
| 5 | 27.2% | 33.5% | 31.6% |
| 6 | 21.1% | 26.3% | 23.7% |
| 7 | 19.7% | 25.1% | 19.7% |
| 8 | 19.9% | 23.5% | 21.9% |
| 9 | 23.8% | 29.2% | 27.7% |

DGD-GT 在 **9/10 段对话上超过 cosine**（conv 3 例外）。DGD-Judge 在 **8/10 段上超过 cosine**。

### 1.5 DialSim 复现

| 数据集 | Cosine | DGD-GT | DGD-Judge | 旧 Judge |
|--------|:------:|:------:|:---------:|:--------:|
| BigBang | 57.1% | 71.4% | 71.4% | 78.6% |
| Friends | 25.0% | 25.0% | 25.0% | 25.0% |
| TheOffice | 25.0% | 25.0% | 25.0% | 50.0% |

BigBang DGD-Judge 从旧的 78.6% 降到 71.4%（不同 LLM 后端导致），其他基本一致。

---

## 二、进化搜索实验

### 2.1 Judge Prompt 进化 V1（usage-based）

**设置：** 10 种群，15 代 GA 进化，200 评测样本（100 LoCoMo + 100 LongMemEval）
**评测方式：** 模拟 agent 回答 → Judge 判断"用了哪个记忆" → 和 GT 对比
**LLM：** DeepSeek-chat（变异 + 评测）

| 阶段 | best accuracy | avg accuracy |
|------|:---:|:---:|
| 原始 baseline (seed 0) | 28.9% | - |
| 最优种子 (seed 4) | 43.1% | - |
| Gen 0 (初始化) | 43.1% | 34.8% |
| Gen 5 | **66.7%** | 48.1% |
| Gen 15 (最终) | **66.7%** | 54.0% |

**进化谱系：**
```
seed-004 (43.1%) "specific claim, fact, or detail ONLY found in that fragment"
  → gen004-000 (61.1%) + "rigorous" 判断标准
    → gen005-006 (66.7%) + "genuinely and exclusively referenced"
```

### 2.2 Judge Prompt 进化 V2（relevance-based）

**设置：** 同 V1，但改变评测目标——不模拟 agent，直接问"哪个记忆和答案相关"
**关键改进：** 每个评测样本只需 1 次 API 调用（V1 需要 2 次）

| 阶段 | best accuracy | avg accuracy |
|------|:---:|:---:|
| Gen 0 (初始化) | 78.6% | 56.0% |
| Gen 6 | 88.9% | 84.4% |
| Gen 8 | **90.9%** | 86.5% |
| Gen 15 (最终) | **90.9%** | 90.0% |

**最优 prompt 进化出的 7 条规则：**
1. DIRECT ANSWER REQUIREMENT — 必须直接包含答案
2. SELF-CONTAINED COMPLETENESS — 单个片段就能回答
3. NO COMBINATION ALLOWED — 不允许组合多片段
4. NO TOPICAL FALLACY — 话题相关不算
5. NO PARTIAL CREDIT — 不给部分分
6. VERIFIABLE INFORMATION — 必须可验证
7. STRICT INTERPRETATION — 有疑问就排除

### 2.3 FunSearch DGD Update Rule（代码进化）

**设置：** FunSearch 群岛模型，5 岛，100 迭代，2 样本/迭代，500 题 GT benchmark
**进化目标：** `dgd_update(M, key, target, dim, alpha, eta)` 的 Python 函数体

| 指标 | 值 |
|------|-----|
| 总评测程序 | 43 |
| 有效程序 | ~21% 成功率 |
| Baseline P@1 | 26.0% |
| 最终 best P@1 | **31.2%** |
| 提升 | +5.2pp |

**最优进化规则的两个创新：**
1. `delta = error / key_norm2` — 误差除以 key 范数归一化
2. 非对角线去掉 alpha 衰减 — 只有对角线有 forget 项

**对比：** η=0.05 参数搜索达到 34.4%，比 FunSearch 最优（31.2%）更高。说明搜索空间可能太小。

### 2.4 进化搜索核心发现

1. **评测目标比进化策略更重要**：V2 换问法（usage→relevance），起步 78.6% 就超过 V1 终点 66.7%
2. **Prompt 措辞影响巨大**：不进化，光换种子 prompt 就 +14pp（28.9% → 43.1%）
3. **进化有效但会饱和**：V1 Gen 5 到顶，V2 Gen 8 到顶，之后 avg 涨但 best 不动
4. **LLM 自动发现的规则比人写的好**：进化出 7 条精确规则 vs 人写的 4 条模糊规则
5. **代码搜索空间小**：DGD update rule 太简单（~15 行），FunSearch 能改进但有限

### 2.5 过拟合风险

⚠️ **所有进化实验的评测和选择用了同一组 200 个样本。** 90.9% 的 V2 结果可能虚高。

待做验证：
- Hold-out：在剩余 1776 题（LoCoMo）上测试
- Cross-dataset：在 DialSim（完全不同的数据集）上测试
- 如果 90% → 80%+ 是真泛化，掉到 60% 是过拟合

---

## 三、FunSearch v2 升级

对标 Google DeepMind 原版 FunSearch 的升级：

| 改进项 | 状态 |
|--------|------|
| Signature 多维分簇 | ✅ 完成 |
| 代码长度偏好 | ✅ 完成 |
| 多测试用例评分（per-conversation P@1） | ✅ 完成 |
| 温度衰减周期 100→30,000 | ✅ 完成 |
| 并行评测（ProcessPoolExecutor） | ✅ 完成 |
| 祖先调用检测 | ✅ 完成 |
| `--seed-from` 参数 | ✅ 完成 |
| 代码专用模型（deepseek-coder） | 跳过 |

**当前状态：** v2 升级版 200 迭代正在后台运行，种子来自 v1 top 10。

---

## 四、资源消耗

| 项目 | API 调用 | Token 估算 | 费用估算 |
|------|---------|-----------|---------|
| 大规模实验（65 个） | ~52,000 | ~15.6M | ~¥20 |
| Judge V1 进化 | ~6,000 | ~12M | ~¥15 |
| Judge V2 进化 | ~3,000 | ~6.7M | ~¥8 |
| FunSearch v1 | ~200 LLM + 本地评测 | ~1M | ~¥2 |
| **总计** | **~61,000** | **~35M** | **~¥45 ($6)** |

---

## 五、文件索引

| 文件 | 内容 |
|------|------|
| `experiments/results/*.json` | 117 个实验结果 |
| `experiments/configs/*.yaml` | 8 个实验配置 |
| `experiments/logs/*.log` | 实验日志 |
| `experiments/evolution/` | Judge V1 进化种群 |
| `experiments/evolution-v2/` | Judge V2 进化种群 |
| `experiments/funsearch/` | FunSearch v1 种群 |
| `experiments/funsearch-v2/` | FunSearch v2 种群（运行中） |
| `docs/report-large-scale.html` | 可视化报告 |
| `docs/plans/2026-03-16-large-scale-experiments.md` | 大规模实验计划 |
| `docs/plans/2026-03-17-judge-prompt-evolution.md` | 进化搜索计划 |
| `docs/plans/2026-03-17-dgd-self-evolution.md` | FunSearch 计划 |
| `docs/plans/2026-03-17-funsearch-v2-upgrade.md` | FunSearch 升级对照分析 |
