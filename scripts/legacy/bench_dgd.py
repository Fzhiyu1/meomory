#!/usr/bin/env python3
"""用 Big Bang Theory 语料做 cosine vs DGD 评测。

流程：
1. 切分语料为 100 个记忆片段
2. 用 GPT-5.4 为每个片段生成 1 个查询问题 + 正确答案片段 ID
3. embedding 所有片段
4. 对每个查询，跑 cosine 和 DGD，记录正确片段是否在 top-k
5. 模拟反馈循环：每次查询后，把正确片段作为正反馈更新 DGD
6. 对比：cosine precision@k（不变） vs DGD precision@k（随反馈提升）
"""
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("MEOMORY_LLM_KEY", "")

from src.embedder import get_embedding
from src.projection import create_projection_matrix, project
from src.dgd import AssociativeMemory
from src.llm_client import chat


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na * nb > 0 else 0


def load_fragments(n=100):
    """切分 Big Bang Theory 语料为记忆片段。"""
    with open("data/benchmarks/memorybench/dialsim-bigbang.jsonl") as f:
        text = json.loads(f.readline())["text"]

    sessions = text.split("[Date:")
    sessions = ["[Date:" + s.strip() for s in sessions if s.strip()]

    fragments = []
    for i, s in enumerate(sessions[:n]):
        lines = s.split("\n")
        date_line = lines[0] if lines else ""
        body = "\n".join(lines[1:])[:500].strip()
        if len(body) < 50:
            continue
        fragments.append({"id": f"bbg-{i:03d}", "date": date_line, "body": body})
    return fragments


def generate_questions(fragments, n_questions=30):
    """用 GPT-5.4 为随机片段生成查询问题。"""
    import random
    random.seed(42)
    selected = random.sample(fragments, min(n_questions, len(fragments)))

    questions = []
    for frag in selected:
        prompt = f"""Based on this conversation excerpt, generate exactly 1 question that someone might ask to find this specific conversation. The question should be specific enough that ONLY this excerpt would be the correct answer.

Excerpt (ID: {frag['id']}):
{frag['body'][:300]}

Output format (JSON only, no other text):
{{"question": "your question here"}}"""

        try:
            raw = chat(prompt, max_tokens=100)
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.strip().startswith("```"))
            result = json.loads(text)
            questions.append({
                "question": result["question"],
                "answer_id": frag["id"],
                "answer_body": frag["body"][:100],
            })
            print(f"  Q: {result['question'][:60]}... → {frag['id']}")
        except Exception as e:
            print(f"  Skip {frag['id']}: {e}")

    return questions


def main():
    print("=== Step 1: Load fragments ===")
    fragments = load_fragments(100)
    print(f"  {len(fragments)} fragments")

    print("\n=== Step 2: Generate questions ===")
    questions = generate_questions(fragments, n_questions=30)
    print(f"  {len(questions)} questions generated")

    # 保存题目以便复用
    with open("data/benchmarks/memorybench/bbg-questions.json", "w") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\n=== Step 3: Embed {len(fragments)} fragments ===")
    frag_vecs = []
    for i, frag in enumerate(fragments):
        vec = get_embedding(frag["body"][:300])
        frag_vecs.append(vec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(fragments)}")

    # 投影
    DIM = 256
    proj = create_projection_matrix(4096, DIM, seed=42)
    proj_vecs = [norm(project(v, proj)) for v in frag_vecs]

    print(f"\n=== Step 4: Embed {len(questions)} questions ===")
    q_vecs = []
    for q in questions:
        vec = get_embedding(q["question"])
        q_vecs.append(norm(project(vec, proj)))

    # === 评测 ===
    print(f"\n=== Step 5: Evaluate cosine vs DGD ===")

    # Cosine baseline（固定，不变）
    cosine_hits = {1: 0, 3: 0, 5: 0}
    for qi, q in enumerate(questions):
        scores = [(cosine(q_vecs[qi], pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        top_ids = [fragments[idx]["id"] for _, idx in scores[:5]]
        for k in [1, 3, 5]:
            if q["answer_id"] in top_ids[:k]:
                cosine_hits[k] += 1

    print(f"\n  Cosine (static):")
    for k in [1, 3, 5]:
        print(f"    Precision@{k}: {cosine_hits[k]}/{len(questions)} = {cosine_hits[k]/len(questions):.1%}")

    # DGD — 第一轮（未训练，应该和 cosine 差不多）
    mem = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)
    dgd_hits_before = {1: 0, 3: 0, 5: 0}
    for qi, q in enumerate(questions):
        act = mem.query(q_vecs[qi])
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        top_ids = [fragments[idx]["id"] for _, idx in scores[:5]]
        for k in [1, 3, 5]:
            if q["answer_id"] in top_ids[:k]:
                dgd_hits_before[k] += 1

    print(f"\n  DGD before feedback:")
    for k in [1, 3, 5]:
        print(f"    Precision@{k}: {dgd_hits_before[k]}/{len(questions)} = {dgd_hits_before[k]/len(questions):.1%}")

    # DGD — 反馈循环（跑 3 轮，每轮把正确答案作为反馈）
    for round_num in range(3):
        for qi, q in enumerate(questions):
            answer_idx = next(i for i, f in enumerate(fragments) if f["id"] == q["answer_id"])
            mem.update(q_vecs[qi], proj_vecs[answer_idx])

    dgd_hits_after = {1: 0, 3: 0, 5: 0}
    for qi, q in enumerate(questions):
        act = mem.query(q_vecs[qi])
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        top_ids = [fragments[idx]["id"] for _, idx in scores[:5]]
        for k in [1, 3, 5]:
            if q["answer_id"] in top_ids[:k]:
                dgd_hits_after[k] += 1

    print(f"\n  DGD after 3 rounds feedback:")
    for k in [1, 3, 5]:
        print(f"    Precision@{k}: {dgd_hits_after[k]}/{len(questions)} = {dgd_hits_after[k]/len(questions):.1%}")

    # 总结
    print(f"\n=== Summary ===")
    print(f"  {'Method':<25} {'P@1':>8} {'P@3':>8} {'P@5':>8}")
    print(f"  {'-'*49}")
    for name, hits in [("Cosine (static)", cosine_hits), ("DGD (before)", dgd_hits_before), ("DGD (3 rounds)", dgd_hits_after)]:
        p1 = f"{hits[1]/len(questions):.1%}"
        p3 = f"{hits[3]/len(questions):.1%}"
        p5 = f"{hits[5]/len(questions):.1%}"
        print(f"  {name:<25} {p1:>8} {p3:>8} {p5:>8}")


if __name__ == "__main__":
    main()
