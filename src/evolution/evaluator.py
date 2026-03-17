"""Judge prompt 评测器。

给定一个 Judge prompt，在采样题目上测量 feedback accuracy。
accuracy = Judge 判断"有用"的片段确实在 GT answer_indices 中的比例。
"""
import asyncio
import json
import random

from src.bench.backends import DeepSeekBackend


async def evaluate_judge_prompt(
    system_prompt: str,
    user_prompt_template: str,
    samples: list[dict],
    backend: DeepSeekBackend,
    concurrency: int = 30,
) -> dict:
    """评测一个 Judge prompt 的 feedback accuracy。

    Args:
        system_prompt: Judge 的 system prompt
        user_prompt_template: Judge 的 user prompt 模板，
            可用变量: {mem_list}, {response}, {question}
        samples: 预构建的评测样本列表，每个包含:
            - question: 问题文本
            - top_bodies: top-3 检索到的片段文本列表
            - answer_indices_local: 在 top_bodies 中哪些是正确的 (0-based set)
        backend: DeepSeek API 后端
        concurrency: 最大并发数

    Returns:
        {"accuracy": float, "total": int, "correct": int, "errors": int, "details": list}
    """
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _eval_one(sample):
        async with sem:
            top_bodies = sample["top_bodies"]
            question = sample["question"]
            gt_local = sample["answer_indices_local"]  # set of 0-based indices

            # Step 1: Agent 回答
            injection = "\n".join(f"- {b}" for b in top_bodies)
            agent_prompt = f"[Context]\n{injection}\n\n[Question] {question}\n\nAnswer briefly based on context."
            try:
                response = await backend.chat(agent_prompt, max_tokens=200)
            except Exception:
                return {"error": True}

            # Step 2: Judge 判断
            mem_list = "\n".join(f"[{i}] {b}" for i, b in enumerate(top_bodies))
            judge_user = user_prompt_template.format(
                mem_list=mem_list,
                response=response[:500],
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

            # Step 3: 计算准确率
            correct = 0
            total = 0
            for idx in used:
                if isinstance(idx, int) and 0 <= idx < len(top_bodies):
                    total += 1
                    if idx in gt_local:
                        correct += 1

            return {"correct": correct, "total": total, "error": False}

    tasks = [_eval_one(s) for s in samples]
    raw_results = await asyncio.gather(*tasks)

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


def build_eval_samples(
    dataset_name: str = "LoCoMo-full-all",
    n_samples: int = 200,
    seed: int = 42,
) -> list[dict]:
    """从数据集构建评测样本。

    每个样本 = 一道题 + cosine top-3 + GT 标签。
    使用缓存的 embedding 避免重复计算。
    """
    import math
    from src.bench.datasets import Dataset
    from src.projection import create_projection_matrix, project

    # 加载数据集
    if dataset_name == "LoCoMo-full-all":
        ds = Dataset.load_locomo_full_all()
    elif dataset_name.startswith("LongMemEval"):
        ds = Dataset.load_longmemeval(dataset_name.split("-")[-1])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 加载缓存的 embedding
    from pathlib import Path
    cache_file = Path("experiments/cache") / f"{ds.name}_embed.json"
    with open(cache_file) as f:
        cached = json.load(f)
    frag_vecs = cached["frag_vecs"]
    q_vecs_raw = cached["q_vecs"]

    # 投影
    proj = create_projection_matrix(4096, 256, seed=42)

    def _norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    def _cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    proj_vecs = [_norm(project(v, proj)) for v in frag_vecs]
    q_vecs = [_norm(project(v, proj)) for v in q_vecs_raw]

    # 采样
    rng = random.Random(seed)
    indices = list(range(len(ds.questions)))
    rng.shuffle(indices)
    indices = indices[:n_samples]

    samples = []
    for qi in indices:
        q = ds.questions[qi]
        # cosine top-3
        scores = [(_cosine(q_vecs[qi], pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        top_3_idx = [idx for _, idx in scores[:3]]

        top_bodies = [ds.fragments[idx]["body"][:200] for idx in top_3_idx]
        answer_set = set(q["answer_indices"])
        # 哪些 top-3 在 GT 中
        answer_indices_local = {i for i, idx in enumerate(top_3_idx) if idx in answer_set}

        samples.append({
            "question": q["question"],
            "top_bodies": top_bodies,
            "answer_indices_local": answer_indices_local,
            "global_indices": top_3_idx,
        })

    return samples
