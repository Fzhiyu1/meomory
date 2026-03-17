"""Judge prompt 评测器 v2：直接测 relevance，不模拟 agent 回答。

v1 的问题：评测"Judge 是否正确判断 agent 用了哪个记忆"
v2 的改进：评测"Judge 是否正确判断哪个记忆和答案相关"

好处：
1. 评测目标和 DGD 需要的信号完全对齐
2. 少一次 API 调用（不需要模拟 agent），速度翻倍
3. 去掉了 agent 回答质量的噪声
"""
import asyncio
import json
import random

from src.bench.backends import DeepSeekBackend


async def evaluate_judge_prompt_v2(
    system_prompt: str,
    user_prompt_template: str,
    samples: list[dict],
    backend: DeepSeekBackend,
    concurrency: int = 30,
) -> dict:
    """评测一个 Judge prompt 的 relevance accuracy。

    Args:
        system_prompt: Judge 的 system prompt
        user_prompt_template: Judge 的 user prompt 模板，
            可用变量: {mem_list}, {question}
        samples: 评测样本，每个包含:
            - question: 问题文本
            - top_bodies: top-3 片段文本
            - answer_indices_local: GT 正确索引 (0-based set)
        backend: DeepSeek API
        concurrency: 并发数
    """
    sem = asyncio.Semaphore(concurrency)

    async def _eval_one(sample):
        async with sem:
            top_bodies = sample["top_bodies"]
            question = sample["question"]
            gt_local = sample["answer_indices_local"]

            mem_list = "\n".join(f"[{i}] {b}" for i, b in enumerate(top_bodies))
            judge_user = user_prompt_template.format(
                mem_list=mem_list,
                question=question,
            )
            try:
                raw = await backend.chat(judge_user, system=system_prompt, max_tokens=50)
                text = raw.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                used = json.loads(text)
                if not isinstance(used, list):
                    used = []
            except Exception:
                return {"error": True}

            correct = 0
            total = 0
            for idx in used:
                if isinstance(idx, int) and 0 <= idx < len(top_bodies):
                    total += 1
                    if idx in gt_local:
                        correct += 1

            return {"correct": correct, "total": total, "error": False}

    raw_results = await asyncio.gather(*[_eval_one(s) for s in samples])

    total_updates = 0
    total_correct = 0
    errors = 0
    for r in raw_results:
        if r.get("error"):
            errors += 1
        else:
            total_updates += r["total"]
            total_correct += r["correct"]

    accuracy = total_correct / total_updates if total_updates > 0 else 0
    return {
        "accuracy": accuracy,
        "total_updates": total_updates,
        "total_correct": total_correct,
        "errors": errors,
        "n_samples": len(samples),
    }
