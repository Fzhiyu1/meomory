"""评测器：用 GT benchmark 评测进化出的 update 函数或类。

v3: numpy 加速版。
    - 投影、归一化、cosine 比较全部用 numpy
    - evolved code 仍在纯 Python 沙箱中运行
    - 评测速度提升 3-5x
"""
import json
import time
from pathlib import Path

import numpy as np

# 缓存：避免重复加载
_dataset_cache = {}


def _load_dataset_and_embeddings(dataset_name: str, dim: int):
    """加载数据集和缓存 embedding，返回 (ds, proj_vecs_np, q_vecs_np, proj_vecs_list, q_vecs_list)。

    返回 numpy 数组（用于快速 cosine）和 list（用于传给 evolved code）。
    """
    cache_key = (dataset_name, dim)
    if cache_key in _dataset_cache:
        return _dataset_cache[cache_key]

    from src.bench.datasets import Dataset
    from src.projection import create_projection_matrix, project
    import math

    if dataset_name == "LoCoMo-full-all":
        ds = Dataset.load_locomo_full_all()
    elif dataset_name.startswith("LongMemEval"):
        ds = Dataset.load_longmemeval(dataset_name.split("-", 1)[1])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_file = Path("experiments/cache") / f"{ds.name}_embed.json"
    with open(cache_file) as f:
        cached = json.load(f)
    frag_vecs = cached["frag_vecs"]
    q_vecs_raw = cached["q_vecs"]

    proj = create_projection_matrix(4096, dim, seed=42)

    def _norm_list(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    from src.projection import project
    proj_vecs_list = [_norm_list(project(v, proj)) for v in frag_vecs]
    q_vecs_list = [_norm_list(project(v, proj)) for v in q_vecs_raw]

    # numpy 版本（用于快速 cosine）
    proj_vecs_np = np.array(proj_vecs_list, dtype=np.float32)
    q_vecs_np = np.array(q_vecs_list, dtype=np.float32)

    # 预归一化（已经归一化了，但确保）
    proj_norms = np.linalg.norm(proj_vecs_np, axis=1, keepdims=True)
    proj_norms[proj_norms == 0] = 1
    proj_vecs_np = proj_vecs_np / proj_norms

    result = (ds, proj_vecs_np, q_vecs_np, proj_vecs_list, q_vecs_list)
    _dataset_cache[cache_key] = result
    return result


def _get_conversation_boundaries(dataset_name: str, max_questions: int) -> list[tuple[str, int, int]]:
    """获取 LoCoMo-full-all 中每段对话的 question 范围。

    返回 [(conv_name, start_idx, end_idx), ...]，其中 [start_idx, end_idx) 是
    该对话在合并后 questions 列表中的索引范围。
    """
    from src.bench.datasets import Dataset

    boundaries = []
    offset = 0
    for i in range(10):
        ds_i = Dataset.load_locomo_full(i)
        n_questions = len(ds_i.questions)
        # 只统计在 max_questions 范围内的部分
        start = offset
        end = min(offset + n_questions, max_questions)
        if start < max_questions:
            boundaries.append((f"conv{i}", start, end))
        offset += n_questions
        if offset >= max_questions:
            break

    return boundaries


def evaluate_update_fn(
    update_fn_or_class,
    dataset_name: str = "LoCoMo-full-all",
    dim: int = 256,
    alpha: float = 1.0,
    eta: float = 0.01,
    n_rounds: int = 3,
    max_questions: int = 500,
    random_offset: int = 0,
    noise_accuracy: float = 1.0,
) -> dict:
    """评测一个 update 函数或 AssociativeMemory 类的 P@1。

    Args:
        update_fn_or_class: 可以是:
            - 函数: dgd_update(M, key, target, dim, alpha, eta) -> M
            - 类: 有 __init__(dim, alpha, eta), query(key), update(key, target)
        dataset_name: 数据集名
        dim, alpha, eta: DGD 参数
        n_rounds: GT 反馈轮数
        max_questions: 最多评测多少题
        random_offset: >0 时从 offset 位置循环取题（防止过拟合固定子集）
        noise_accuracy: 反馈准确率 (0.0-1.0)。1.0=纯GT，<1.0 时以该概率给正确反馈，
                        否则随机选一个错误片段。用于噪声鲁棒性评测和混合 fitness。

    Returns:
        {
            "p_at_1": float, "p_at_3": float, "p_at_5": float,
            "scores_per_test": dict[str, float],
            "rounds": list, "elapsed": float
        }
    """
    t0 = time.time()

    ds, proj_vecs_np, q_vecs_np_all, proj_vecs_list, q_vecs_list_all = _load_dataset_and_embeddings(dataset_name, dim)

    # 随机 offset 取题：从 offset 位置循环取 max_questions 题
    total_q = len(ds.questions)
    if random_offset > 0 and max_questions < total_q:
        import numpy as _np
        offset = random_offset % total_q
        indices = [(offset + i) % total_q for i in range(max_questions)]
        questions = [ds.questions[i] for i in indices]
        q_vecs_np = q_vecs_np_all[indices]
        q_vecs_list = [q_vecs_list_all[i] for i in indices]
    else:
        questions = ds.questions[:max_questions]
        q_vecs_np = q_vecs_np_all[:max_questions]
        q_vecs_list = q_vecs_list_all[:max_questions]

    # 判断是函数还是类
    is_class = isinstance(update_fn_or_class, type)

    # 噪声反馈设置
    import random as _random
    n_frags = len(proj_vecs_list)
    noise_rng = _random.Random(42 + random_offset)  # 可复现但每次 offset 不同
    use_noise = noise_accuracy < 1.0

    def _noisy_target(q):
        """返回反馈目标索引：以 noise_accuracy 概率给 GT，否则随机。"""
        target_idx = q["answer_indices"][0]
        if use_noise and noise_rng.random() > noise_accuracy:
            target_idx = noise_rng.randint(0, n_frags - 1)
        return target_idx

    if is_class:
        try:
            obj = update_fn_or_class(dim=dim)
        except Exception:
            try:
                # 兼容旧算法（接受 alpha/eta 参数）
                obj = update_fn_or_class(dim=dim, alpha=alpha, eta=eta)
            except Exception:
                return {"p_at_1": 0.0, "error": True, "elapsed": time.time() - t0,
                        "scores_per_test": {"overall": 0.0}}

        def _query_top5(qi):
            """用 numpy 加速的 query + cosine top-5。"""
            try:
                act = obj.query(q_vecs_list[qi])
            except Exception:
                act = q_vecs_list[qi]
            # numpy cosine: act @ proj_vecs_np.T
            act_np = np.array(act, dtype=np.float32)
            act_norm = np.linalg.norm(act_np)
            if act_norm > 0:
                act_np = act_np / act_norm
            scores = proj_vecs_np @ act_np  # (n_frags,)
            top5_idx = np.argpartition(scores, -5)[-5:]
            top5_idx = top5_idx[np.argsort(scores[top5_idx])[::-1]]
            return top5_idx.tolist()

        def _update(qi, q):
            target_idx = _noisy_target(q)
            obj.update(q_vecs_list[qi], proj_vecs_list[target_idx])
    else:
        M = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]

        def _query_top5(qi):
            q_vec = q_vecs_list[qi]
            act = [sum(M[i][j] * q_vec[j] for j in range(dim)) for i in range(dim)]
            act_np = np.array(act, dtype=np.float32)
            act_norm = np.linalg.norm(act_np)
            if act_norm > 0:
                act_np = act_np / act_norm
            scores = proj_vecs_np @ act_np
            top5_idx = np.argpartition(scores, -5)[-5:]
            top5_idx = top5_idx[np.argsort(scores[top5_idx])[::-1]]
            return top5_idx.tolist()

        def _update(qi, q):
            target_idx = _noisy_target(q)
            update_fn_or_class(M, q_vecs_list[qi], proj_vecs_list[target_idx], dim, alpha, eta)

    rounds = []
    for r in range(1, n_rounds + 1):
        # 在线流式评测：先测再学（模拟真实部署）
        hits = {1: 0, 3: 0, 5: 0}
        per_question_hit = [False] * len(questions)
        for qi, q in enumerate(questions):
            # 1. 先评测（DGD 还没见过这题的答案）
            answer_set = set(q["answer_indices"])
            top_ids = _query_top5(qi)
            for k in [1, 3, 5]:
                if any(idx in answer_set for idx in top_ids[:k]):
                    hits[k] += 1
            if top_ids and top_ids[0] in answer_set:
                per_question_hit[qi] = True

            # 2. 再学习（GT/noisy 反馈）
            try:
                _update(qi, q)
            except Exception:
                return {
                    "p_at_1": 0.0, "error": True, "elapsed": time.time() - t0,
                    "scores_per_test": {"overall": 0.0},
                }

        total = len(questions)
        metrics = {k: hits[k] / total for k in hits}
        rounds.append({
            "round": r, "p_at_1": metrics[1],
            "p_at_3": metrics[3], "p_at_5": metrics[5],
        })

    # --- 计算 scores_per_test：按对话分组的 P@1 ---
    scores_per_test: dict[str, float] = {}

    if dataset_name == "LoCoMo-full-all":
        boundaries = _get_conversation_boundaries(dataset_name, max_questions)
        for conv_name, start, end in boundaries:
            if end <= start:
                continue
            conv_hits = sum(1 for qi in range(start, end) if per_question_hit[qi])
            scores_per_test[conv_name] = conv_hits / (end - start)
    else:
        # 非 LoCoMo-full-all：将 questions 平均分成 10 桶
        n = len(questions)
        bucket_size = max(1, n // 10)
        for bi in range(10):
            start = bi * bucket_size
            end = min(start + bucket_size, n) if bi < 9 else n
            if start >= n:
                break
            bucket_hits = sum(1 for qi in range(start, end) if per_question_hit[qi])
            scores_per_test[f"bucket{bi}"] = bucket_hits / (end - start)

    # 最后放 overall P@1（_reduce_score 取最后一个 key）
    final = rounds[-1] if rounds else {"p_at_1": 0}
    scores_per_test["overall"] = final["p_at_1"]

    elapsed = time.time() - t0
    return {
        "p_at_1": final["p_at_1"],
        "p_at_3": final.get("p_at_3", 0),
        "p_at_5": final.get("p_at_5", 0),
        "scores_per_test": scores_per_test,
        "rounds": rounds,
        "elapsed": elapsed,
        "error": False,
    }
