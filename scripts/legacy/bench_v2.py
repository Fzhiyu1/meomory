#!/usr/bin/env python3
"""v2 实验：LLM 判断者反馈 + 并行执行。

改进：
1. 反馈信号从 embedding 相似度 → LLM 判断者（更准确）
2. 每轮 14 题并发调 DeepSeek（快 10 倍）
3. DGD 更新在每轮结束后批量执行
"""
import asyncio
import json
import math
import ast
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LLM_URL = os.environ.get("MEOMORY_LLM_URL", "https://api.deepseek.com/v1")
LLM_KEY = os.environ.get("MEOMORY_LLM_KEY", "")
LLM_MODEL = os.environ.get("MEOMORY_LLM_MODEL", "deepseek-chat")

from src.embedder import get_embedding
from src.projection import create_projection_matrix, project
from src.dgd import AssociativeMemory


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na * nb > 0 else 0


# ─── Async LLM ─────────────────────────────────

import httpx

async def llm_chat_async(prompt: str, system: str = "", max_tokens: int = 300) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{LLM_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_KEY}"},
            json={"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ─── LLM Judge ─────────────────────────────────

JUDGE_SYSTEM = """You are a judge. Given injected memories and an agent's response, determine which memories the response actually referenced or used.

Rules:
- Only mark a memory as "used" if the response clearly draws on information from that memory
- Paraphrasing counts as usage
- Mentioning the same topic is NOT enough — the response must use specific information from the memory
- Output ONLY a JSON array of indices (0-based), e.g. [0, 2]. Output [] if none were used."""


async def judge_feedback_async(memories_text: list[str], response: str) -> list[int]:
    """让 LLM 判断 agent 回复引用了哪些记忆。返回被引用的 index 列表。"""
    mem_list = "\n".join(f"[{i}] {m[:200]}" for i, m in enumerate(memories_text))
    prompt = f"Injected memories:\n{mem_list}\n\nAgent response:\n{response[:500]}\n\nWhich memories were used? Output JSON array only."

    try:
        raw = await llm_chat_async(prompt, system=JUDGE_SYSTEM, max_tokens=50)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception:
        return []


# ─── Data Loading ──────────────────────────────

def load_corpus(n=805):
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


# ─── One Round (parallel) ──────────────────────

async def run_one_round_auto(valid_qs, q_vecs, proj_vecs, fragments, mem):
    """一轮自动反馈：并发执行所有题的 agent 调用 + judge 调用。"""

    async def process_one(qi, q):
        # 1. DGD match top-3
        act = mem.query(q_vecs[qi])
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        top_3_indices = [idx for _, idx in scores[:3]]
        top_3_bodies = [fragments[idx]["body"][:200] for idx in top_3_indices]

        # 2. Agent responds
        injection = "\n".join(f"- {b}" for b in top_3_bodies)
        prompt = f"[Context]\n{injection}\n\n[Question] {q['question'][:200]}\n\nAnswer briefly."
        try:
            response = await llm_chat_async(prompt, max_tokens=200)
        except Exception as e:
            return {"qi": qi, "error": str(e), "updates": []}

        # 3. LLM judge feedback
        used_indices = await judge_feedback_async(top_3_bodies, response)

        # Map back to fragment indices
        updates = []
        answer_set = set(q["answer_indices"])
        for local_idx in used_indices:
            if 0 <= local_idx < len(top_3_indices):
                frag_idx = top_3_indices[local_idx]
                is_correct = frag_idx in answer_set
                updates.append({"frag_idx": frag_idx, "correct": is_correct})

        return {"qi": qi, "updates": updates, "error": None}

    # 并发执行所有题
    tasks = [process_one(qi, q) for qi, q in enumerate(valid_qs)]
    results = await asyncio.gather(*tasks)

    # 统计并批量更新 DGD
    total_updates = 0
    correct_updates = 0
    errors = 0

    for r in results:
        if r["error"]:
            errors += 1
            continue
        for u in r["updates"]:
            qi = r["qi"]
            mem.update(q_vecs[qi], proj_vecs[u["frag_idx"]])
            total_updates += 1
            if u["correct"]:
                correct_updates += 1

    return total_updates, correct_updates, errors


# ─── Main ──────────────────────────────────────

async def async_main():
    N_ROUNDS = 20
    EVAL_EVERY = 2
    DIM = 256

    print("=== Loading data ===")
    fragments = load_corpus(805)
    questions = load_questions()
    questions = find_answer_indices(questions, fragments)
    n_frags = len(fragments)
    valid_qs = [q for q in questions if q.get("answer_indices")]
    print(f"  {n_frags} fragments, {len(valid_qs)} valid questions")

    print("\n=== Embedding fragments ===")
    frag_vecs = []
    for i, f in enumerate(fragments):
        frag_vecs.append(get_embedding(f["body"][:400]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frags}")

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

    # DGD + ground truth (same as before, fast)
    print(f"\n=== DGD + ground truth ({N_ROUNDS} rounds) ===")
    mem_gt = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)

    def dgd_gt_query(qv):
        act = mem_gt.query(qv)
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    gt_curve = []
    for r in range(1, N_ROUNDS + 1):
        for qi, q in enumerate(valid_qs):
            target = q["answer_indices"][0]
            mem_gt.update(q_vecs[qi], proj_vecs[target])
        if r % EVAL_EVERY == 0 or r == 1:
            result = evaluate(q_vecs, proj_vecs, fragments, valid_qs, dgd_gt_query)
            gt_curve.append((r, result[1]))
            print(f"  Round {r:2d}: P@1={result[1]:.1%} P@3={result[3]:.1%} P@5={result[5]:.1%}")

    # DGD + LLM judge auto feedback (parallel)
    print(f"\n=== DGD + LLM judge feedback ({N_ROUNDS} rounds, parallel) ===")
    mem_auto = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)

    def dgd_auto_query(qv):
        act = mem_auto.query(qv)
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    auto_curve = []
    for r in range(1, N_ROUNDS + 1):
        t0 = time.time()
        total_upd, correct_upd, errors = await run_one_round_auto(
            valid_qs, q_vecs, proj_vecs, fragments, mem_auto
        )
        elapsed = time.time() - t0
        accuracy = correct_upd / total_upd if total_upd > 0 else 0

        if r % EVAL_EVERY == 0 or r == 1:
            result = evaluate(q_vecs, proj_vecs, fragments, valid_qs, dgd_auto_query)
            auto_curve.append((r, result[1]))
            print(f"  Round {r:2d}: P@1={result[1]:.1%} P@3={result[3]:.1%} P@5={result[5]:.1%}  "
                  f"(updates:{total_upd} correct:{accuracy:.0%} errors:{errors} {elapsed:.1f}s)")
        else:
            print(f"  Round {r:2d}: updates:{total_upd} correct:{accuracy:.0%} errors:{errors} {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Learning Curve (P@1)")
    print(f"{'='*60}")
    print(f"  {'Round':<8} {'Cosine':>10} {'GT':>10} {'Auto(Judge)':>12}")
    print(f"  {'-'*40}")

    gt_dict = dict(gt_curve)
    auto_dict = dict(auto_curve)
    all_rounds = sorted(set(list(gt_dict.keys()) + list(auto_dict.keys())))

    for r in all_rounds:
        gt_val = f"{gt_dict[r]:.1%}" if r in gt_dict else ""
        auto_val = f"{auto_dict[r]:.1%}" if r in auto_dict else ""
        print(f"  {r:<8} {cosine_result[1]:>9.1%} {gt_val:>10} {auto_val:>12}")

    # Save
    curve_data = {
        "cosine_p1": cosine_result[1],
        "gt_curve": gt_curve,
        "auto_curve": auto_curve,
    }
    with open("data/benchmarks/memorybench/learning_curve_v2.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    print(f"\n  Saved to data/benchmarks/memorybench/learning_curve_v2.json")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
