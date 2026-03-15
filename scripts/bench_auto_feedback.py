#!/usr/bin/env python3
"""关键实验：自动反馈下 DGD 是否能学习？

对比三种条件：
1. Cosine baseline（不学习，固定）
2. DGD + ground truth 反馈（理想上界）
3. DGD + 自动反馈（真实条件：用 GPT-5.4 回复的 embedding 判断哪条记忆被用了）

每隔 N 轮评测一次 P@1，画学习曲线。
"""
import json
import math
import ast
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("MEOMORY_LLM_KEY", "cpa_Sc-u_63M5sRFF45J-CN33OlQ6PnMJ0Yn")

from src.embedder import get_embedding
from src.projection import create_projection_matrix, project
from src.dgd import AssociativeMemory
from src.llm_client import chat

FEEDBACK_THRESHOLD = 0.50  # embedding 相似度高于此值 = 判定为"被用了"


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na * nb > 0 else 0


def load_corpus(n=200):
    with open("data/benchmarks/memorybench/dialsim-bigbang.jsonl") as f:
        text = json.loads(f.readline())["text"]
    sessions = text.split("[Date:")
    sessions = ["[Date:" + s.strip() for s in sessions if s.strip()]
    fragments = []
    for i, s in enumerate(sessions[:n]):
        body = s[:800].strip()
        if len(body) < 50:
            continue
        fragments.append({"id": i, "body": body, "full": s})
    return fragments


def load_questions():
    import pyarrow.parquet as pq
    table = pq.read_table("data/benchmarks/memorybench/dialsim-bigbang-test.parquet")
    df = table.to_pydict()
    questions = []
    for i in range(len(df["test_idx"])):
        info = df["info"][i]
        if isinstance(info, str):
            info = ast.literal_eval(info)
        prompt = df["input_prompt"][i]
        q_start = prompt.rfind("[Question]")
        a_start = prompt.rfind("[Answer]")
        if q_start >= 0 and a_start >= 0:
            question = prompt[q_start + 10:a_start].strip()
        else:
            question = prompt[-200:]
        questions.append({
            "idx": df["test_idx"][i],
            "question": question[:300],
            "golden_answer": info.get("golden_answer", ""),
        })
    return questions


def find_answer_indices(questions, fragments):
    for q in questions:
        answer = q["golden_answer"].lower()
        q["answer_indices"] = [i for i, f in enumerate(fragments) if answer in f["full"].lower()]
    return questions


def evaluate(q_vecs, proj_vecs, fragments, valid_qs, query_fn):
    """评测 P@1, P@3, P@5"""
    hits = {1: 0, 3: 0, 5: 0}
    for qi, q in enumerate(valid_qs):
        answer_indices = set(q["answer_indices"])
        scores = query_fn(q_vecs[qi])
        top_ids = [idx for _, idx in scores[:5]]
        for k in [1, 3, 5]:
            if any(idx in answer_indices for idx in top_ids[:k]):
                hits[k] += 1
    total = len(valid_qs)
    return {k: hits[k] / total if total > 0 else 0 for k in hits}


def main():
    N_FRAGS = 805  # 全量
    DIM = 256
    N_ROUNDS = 20
    EVAL_EVERY = 2

    print("=== Loading data ===")
    fragments = load_corpus(N_FRAGS)
    questions = load_questions()
    questions = find_answer_indices(questions, fragments)
    valid_qs = [q for q in questions if any(idx < N_FRAGS for idx in q.get("answer_indices", []))]
    for q in valid_qs:
        q["answer_indices"] = [idx for idx in q["answer_indices"] if idx < N_FRAGS]
    print(f"  {len(fragments)} fragments, {len(valid_qs)} valid questions")

    print("\n=== Embedding fragments ===")
    frag_vecs = []
    for i, f in enumerate(fragments):
        frag_vecs.append(get_embedding(f["body"][:400]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(fragments)}")

    proj = create_projection_matrix(4096, DIM, seed=42)
    proj_vecs = [norm(project(v, proj)) for v in frag_vecs]

    print("\n=== Embedding questions ===")
    q_vecs = [norm(project(get_embedding(q["question"][:300]), proj)) for q in valid_qs]

    # Cosine baseline
    def cosine_query(qv):
        scores = [(cosine(qv, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    cosine_result = evaluate(q_vecs, proj_vecs, fragments, valid_qs, cosine_query)
    print(f"\n  Cosine baseline: P@1={cosine_result[1]:.1%} P@3={cosine_result[3]:.1%} P@5={cosine_result[5]:.1%}")

    # === DGD + ground truth ===
    print(f"\n=== DGD + ground truth ({N_ROUNDS} rounds) ===")
    mem_gt = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)

    def dgd_gt_query(qv):
        act = mem_gt.query(qv)
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    gt_curve = []
    for round_num in range(1, N_ROUNDS + 1):
        for qi, q in enumerate(valid_qs):
            target = q["answer_indices"][0]
            mem_gt.update(q_vecs[qi], proj_vecs[target])

        if round_num % EVAL_EVERY == 0 or round_num == 1:
            result = evaluate(q_vecs, proj_vecs, fragments, valid_qs, dgd_gt_query)
            gt_curve.append((round_num, result))
            print(f"  Round {round_num:2d}: P@1={result[1]:.1%} P@3={result[3]:.1%} P@5={result[5]:.1%}")

    # === DGD + auto feedback ===
    print(f"\n=== DGD + auto feedback ({N_ROUNDS} rounds) ===")
    mem_auto = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)

    def dgd_auto_query(qv):
        act = mem_auto.query(qv)
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    auto_curve = []
    total_updates = 0
    correct_updates = 0

    for round_num in range(1, N_ROUNDS + 1):
        round_updates = 0
        round_correct = 0

        for qi, q in enumerate(valid_qs):
            # 1. DGD 匹配 top-3
            act = mem_auto.query(q_vecs[qi])
            scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
            scores.sort(reverse=True)
            top_3 = [idx for _, idx in scores[:3]]

            # 2. 用 GPT-5.4 回答（注入 top-3 记忆）
            injection = "\n".join(f"- {fragments[idx]['body'][:200]}" for idx in top_3)
            prompt = f"[Context]\n{injection}\n\n[Question] {q['question'][:200]}\n\nAnswer briefly."

            try:
                response = chat(prompt, max_tokens=200)
            except Exception as e:
                print(f"    LLM error at round {round_num} q{qi}: {e}")
                continue

            # 3. 自动反馈：response embedding vs 每条注入记忆的 embedding
            try:
                resp_vec = norm(project(get_embedding(response[:500]), proj))
            except Exception:
                continue

            answer_indices = set(q["answer_indices"])
            for idx in top_3:
                sim = cosine(resp_vec, proj_vecs[idx])
                if sim >= FEEDBACK_THRESHOLD:
                    mem_auto.update(q_vecs[qi], proj_vecs[idx])
                    round_updates += 1
                    if idx in answer_indices:
                        round_correct += 1

        total_updates += round_updates
        if round_updates > 0:
            correct_updates += round_correct

        if round_num % EVAL_EVERY == 0 or round_num == 1:
            result = evaluate(q_vecs, proj_vecs, fragments, valid_qs, dgd_auto_query)
            accuracy = round_correct / round_updates if round_updates > 0 else 0
            auto_curve.append((round_num, result))
            print(f"  Round {round_num:2d}: P@1={result[1]:.1%} P@3={result[3]:.1%} P@5={result[5]:.1%}  (updates: {round_updates}, correct: {accuracy:.0%})")

    # === 总结 ===
    print(f"\n{'='*60}")
    print(f"  Learning Curve Summary")
    print(f"{'='*60}")
    print(f"  {'Round':<8} {'Cosine':>10} {'DGD+GT':>10} {'DGD+Auto':>10}")
    print(f"  {'-'*38}")
    print(f"  {'base':<8} {cosine_result[1]:>9.1%} {'':>10} {'':>10}")

    gt_dict = {r: res for r, res in gt_curve}
    auto_dict = {r: res for r, res in auto_curve}
    all_rounds = sorted(set(list(gt_dict.keys()) + list(auto_dict.keys())))

    for r in all_rounds:
        gt_val = f"{gt_dict[r][1]:.1%}" if r in gt_dict else ""
        auto_val = f"{auto_dict[r][1]:.1%}" if r in auto_dict else ""
        print(f"  {r:<8} {cosine_result[1]:>9.1%} {gt_val:>10} {auto_val:>10}")

    print(f"\n  Total auto-feedback updates: {total_updates}")
    print(f"  Feedback accuracy (updates on correct fragments): {correct_updates/total_updates:.1%}" if total_updates > 0 else "")

    # 保存学习曲线数据
    curve_data = {
        "cosine_p1": cosine_result[1],
        "gt_curve": [(r, res[1]) for r, res in gt_curve],
        "auto_curve": [(r, res[1]) for r, res in auto_curve],
        "total_updates": total_updates,
        "correct_updates": correct_updates,
    }
    with open("data/benchmarks/memorybench/learning_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    print(f"\n  Learning curve saved to data/benchmarks/memorybench/learning_curve.json")


if __name__ == "__main__":
    main()
