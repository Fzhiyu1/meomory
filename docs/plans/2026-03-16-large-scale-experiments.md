# meomory 大规模实验计划

> **For Claude:** 用户已授权全权执行。**绝不降级、绝不跳过、绝不简化。** 每个实验必须完整跑完。通过状态文件 `experiments/status.json` 持久化进度——如果 session 中断、context 被压缩、或任何原因重启，**第一件事读 status.json 恢复上下文继续执行。**

**Goal:** 在 2446+ 题上全面验证 DGD 记忆假肢，产出论文级实验数据和可视化报告。

**Architecture:** 6 个实验阶段 + 1 个监控循环。状态持久化到 `experiments/status.json`，每完成一个实验就更新状态。监控循环在所有阶段完成后持续运行：检查完整性、补跑缺失、精进分析、生成报告。

**Tech Stack:** Python 3.14, qwen3-embedding (LAN Ollama), DeepSeek API, asyncio, Chart.js

---

## 核心规则

1. **绝不降级：** 每个实验必须产出完整结果文件。如果某个实验失败，修复后重跑，不能标记为"跳过"。
2. **状态持久化：** 每完成一个实验，立即更新 `experiments/status.json`。session 中断后可从任意位置恢复。
3. **自动验证：** 每个阶段完成后，脚本自动验证结果完整性。不完整则自动补跑。
4. **监控循环：** 所有阶段完成后进入循环——检查、补充、精进、总结。直到没有更多可做的事。

---

## 状态文件格式

`experiments/status.json`:

```json
{
  "started_at": "2026-03-16T21:30:00",
  "env": {
    "embed_url": "http://192.168.1.100:11435",
    "llm_key_set": true
  },
  "phases": {
    "1_main": {"status": "pending", "expected": 10, "completed": 0, "failed": []},
    "2_per_conv": {"status": "pending", "expected": 30, "completed": 0, "failed": []},
    "3_param_search": {"status": "pending", "expected": 12, "completed": 0, "failed": []},
    "4_learning_curve": {"status": "pending", "expected": 4, "completed": 0, "failed": []},
    "5_dialsim_rerun": {"status": "pending", "expected": 9, "completed": 0, "failed": []},
    "6_report": {"status": "pending"}
  },
  "loop": {
    "iteration": 0,
    "checks": [],
    "supplements": [],
    "improvements": []
  },
  "last_updated": "2026-03-16T21:30:00"
}
```

---

## 环境变量（全程生效）

```bash
export MEOMORY_EMBED_URL=http://192.168.1.100:11435
export MEOMORY_LLM_KEY=${DEEPSEEK_API_KEY}
```

---

## 实验总览

| 阶段 | 实验数 | 题数 | LLM调用 | 预估时间 |
|------|--------|------|---------|----------|
| 1. 主实验 | 10 | 2446 | ~24,000 | ~1h |
| 2. 逐对话拆分 | 30 | 1976 | ~12,000 | ~2h |
| 3. 参数搜索 | 12 | 1976 | 0 (GT) | ~15min |
| 4. 学习曲线 | 4 | 2446 | ~16,000 | ~1.5h |
| 5. DialSim 复现 | 9 | ~34 | ~400 | ~15min |
| 6. 报告生成 | - | - | 0 | ~10min |
| **总计** | **65** | - | **~52,000** | **~5.5h** |

---

## Task 1: 创建主控脚本 + 状态系统

**Files:**
- Create: `scripts/run_all_experiments.py`
- Create: `experiments/status.json`

**主控脚本职责：**

```python
#!/usr/bin/env python3
"""主控脚本：按阶段执行全部实验，维护状态，自动补跑。

用法:
    .venv/bin/python scripts/run_all_experiments.py

特性:
- 读取 experiments/status.json 恢复进度
- 每完成一个实验立即更新状态
- 每个阶段完成后验证结果完整性
- 失败的实验自动重试（最多 3 次）
- 全部完成后进入监控循环
"""
```

主控脚本核心逻辑：

```
1. 读取或创建 status.json
2. 检查环境（embedding 服务、API key）
3. 按阶段顺序执行：
   for phase in [1_main, 2_per_conv, 3_param_search, 4_learning_curve, 5_dialsim_rerun]:
       if phase.status == "done": continue
       phase.status = "running"
       创建该阶段的 YAML 配置（如果不存在）
       运行实验（调用 run_bench.py 或直接调用 run_all）
       验证结果完整性
       如果有失败的，重试（最多 3 次）
       if 全部完成: phase.status = "done"
       else: phase.status = "partial", 记录失败列表
       更新 status.json
4. 生成报告（阶段 6）
5. 进入监控循环
```

---

## Task 2: 创建全部实验配置

**Files:**
- Exists: `experiments/configs/large-scale.yaml` (阶段 1)
- Create: `experiments/configs/locomo-per-conv.yaml` (阶段 2)
- Create: `experiments/configs/param-search-large.yaml` (阶段 3)
- Create: `experiments/configs/learning-curve.yaml` (阶段 4)
- Create: `experiments/configs/dialsim-rerun.yaml` (阶段 5)

### 阶段 1: 主实验 — `large-scale.yaml` (已存在)

10 个实验：5 方法 × 2 数据集。详见文件。

### 阶段 2: 逐对话拆分 — `locomo-per-conv.yaml`

30 个实验：10 段对话 × 3 方法（cosine, dgd-gt, dgd-judge）

```yaml
experiments:
  # 对 i in 0..9:
  - name: cosine-locomo-{i}
    dataset: LoCoMo-full-{i}
    method: cosine

  - name: dgd-gt-locomo-{i}
    dataset: LoCoMo-full-{i}
    method: dgd_gt
    dgd: {dim: 256, alpha: 1.0, eta: 0.01}
    rounds: 3
    eval_every: 1

  - name: dgd-judge-locomo-{i}
    dataset: LoCoMo-full-{i}
    method: dgd_llm_judge
    dgd: {dim: 256, alpha: 1.0, eta: 0.01}
    rounds: 3
    eval_every: 1
    llm_backend:
      type: deepseek
      url: https://api.deepseek.com/v1
      key: ${MEOMORY_LLM_KEY}
      model: deepseek-chat
```

### 阶段 3: 参数搜索 — `param-search-large.yaml`

12 个实验，全部用 dgd-gt（零 LLM 成本）：

**α 搜索（4 个）：** 0.9, 0.95, 0.99, 1.0
**dim 搜索（4 个）：** 64, 128, 256, 512
**η 搜索（4 个）：** 0.005, 0.01, 0.02, 0.05

所有在 LoCoMo-full-all 上跑 3 轮。

### 阶段 4: 学习曲线 — `learning-curve.yaml`

4 个实验：dgd-gt 和 dgd-judge × 2 数据集，**10 轮**，每轮评测。

### 阶段 5: DialSim 复现 — `dialsim-rerun.yaml`

9 个实验：3 数据集 × 3 方法（cosine, dgd-gt, dgd-judge），3 轮。

---

## Task 3: 运行阶段 1 — 主实验

**运行命令:**
```bash
MEOMORY_EMBED_URL=http://192.168.1.100:11435 \
MEOMORY_LLM_KEY=${DEEPSEEK_API_KEY} \
.venv/bin/python scripts/run_bench.py experiments/configs/large-scale.yaml
```

**完成后验证:**
检查 10 个结果文件全部存在且包含 metrics。更新 status.json。

---

## Task 4: 运行阶段 2 — 逐对话拆分

同上模式。30 个实验。

---

## Task 5: 运行阶段 3 — 参数搜索

同上模式。12 个实验。不需要 LLM_KEY。

---

## Task 6: 运行阶段 4 — 学习曲线

同上模式。4 个实验，每个 10 轮。

---

## Task 7: 运行阶段 5 — DialSim 复现

同上模式。9 个实验。

---

## Task 8: 阶段 6 — 生成实验报告

**Files:**
- Create: `scripts/gen_report_large.py`
- Create: `docs/report-large-scale.html`

**报告内容（7 个板块）：**

1. **主实验结果矩阵** — 5 方法 × 2 数据集的 P@1/P@3/P@5 表格 + 柱状图
2. **逐对话方差分析** — 10 段对话的 P@1 分布（boxplot 或 bar chart with error bars）
3. **参数敏感度** — α/dim/η 三张曲线图，标注死亡谷和甜区
4. **学习曲线** — Round 1-10 的 P@1 演变（GT vs Judge × 2 数据集）
5. **分类别分析** — 5 种 LoCoMo category 的方法对比（从 details 提取每题 rank 后分组统计）
6. **DialSim 复现对比** — 新旧结果对照表
7. **关键发现总结** — 3-5 条核心结论

技术：HTML + Chart.js，单文件可直接浏览器打开。

---

## Task 9: 监控循环

**这是最关键的 Task。** 阶段 1-6 完成后，进入持续监控循环。

```
loop_iteration = 0
while True:
    loop_iteration += 1

    # ===== 检查：所有实验完成了吗？=====
    missing = scan_for_missing_results()
    if missing:
        log(f"发现 {len(missing)} 个缺失实验，补跑")
        rerun(missing)
        continue

    # ===== 检查：结果有异常吗？=====
    anomalies = check_for_anomalies()
    # 例如：某个实验 P@1=0%，error率>50%，某个参数结果和预期严重不符
    if anomalies:
        log(f"发现 {len(anomalies)} 个异常，诊断并修复")
        diagnose_and_fix(anomalies)
        continue

    # ===== 补充：有新的实验值得跑吗？=====
    supplements = identify_supplements()
    # 例如：
    # - 参数搜索发现意外的好值，需要在 dgd-judge 上确认
    # - 某个 category 表现异常好/差，需要深入分析
    # - LongMemEval S 版下载完了，可以加入实验
    if supplements:
        log(f"发现 {len(supplements)} 个补充实验")
        run_supplements(supplements)
        continue

    # ===== 精进：报告可以改进吗？=====
    improvements = review_report()
    # 例如：
    # - 图表缺少 error bar
    # - 缺少统计显著性检验
    # - 结论可以更精确
    if improvements:
        log(f"报告有 {len(improvements)} 处可改进")
        improve_report(improvements)
        continue

    # ===== 总结：更新 status.json 和 memory =====
    update_status(loop_iteration, "all_complete")
    update_memory()  # 更新 Claude memory 供下次对话使用
    log("所有任务完成，监控循环结束")
    break
```

### 监控循环的具体检查项：

**完整性检查：**
- [ ] 65 个实验结果文件全部存在
- [ ] 每个结果文件包含 metrics（P@1/P@3/P@5）
- [ ] DGD 实验的 rounds 数量正确（3 轮或 10 轮）
- [ ] 无 error 率 > 30% 的实验

**异常检测：**
- [ ] P@1 = 0% 的实验（肯定出错了）
- [ ] DGD 反而比 cosine 差的实验（需要分析原因）
- [ ] feedback_accuracy < 20% 的实验（judge 可能失效）
- [ ] elapsed 异常长的实验（可能有网络问题）

**补充实验候选：**
- [ ] 参数搜索中发现新的好值 → 用 dgd-judge 确认
- [ ] LongMemEval S 版是否下载完成 → 加入实验
- [ ] 某个 category 需要单独深入分析

**报告精进：**
- [ ] 图表是否覆盖所有实验
- [ ] 统计显著性检验（bootstrap 或 McNemar）
- [ ] 与旧实验结果的一致性分析
- [ ] 关键发现是否准确反映数据

**记忆更新：**
- [ ] 更新 `project_meomory_design.md` 的实验结论部分
- [ ] 如有新发现（如参数搜索意外结果），记录到知识库

---

## 错误处理

**Embedding 服务不可达：**
- 检查 LAN：`ping 192.168.1.100`
- 备选 Tailscale：`export MEOMORY_EMBED_URL=http://100.117.243.72:11435`
- 如果两个都不通，等待并重试（不要跳过）

**DeepSeek API 错误：**
- 429/500/timeout：单题级别已有 try/except，不会中断实验
- 如果 error 率 > 50%：检查 API key 余额，暂停 5 分钟后重试
- 绝不因 API 错误降级为 GT-only

**Session 中断恢复：**
1. 读取 `experiments/status.json`
2. 读取本 plan 文件恢复上下文
3. 从上次中断的位置继续
4. 已完成的实验有结果文件 + 检查点，自动跳过

**embedding 缓存损坏：**
- 删除 `experiments/cache/{dataset}_embed.json`
- 重新运行（会自动重建缓存）

---

## 交付物

实验完成后，用户醒来应该看到：

1. `experiments/status.json` — 状态全部 `done`
2. `experiments/results/` — 65+ 个结果 JSON
3. `experiments/logs/` — 每个阶段的完整日志
4. `docs/report-large-scale.html` — 可视化报告（浏览器直接打开）
5. 更新后的 memory 文件 — 记录新的实验结论
6. Git commit — 所有代码、配置、报告已提交
