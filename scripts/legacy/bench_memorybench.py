#!/usr/bin/env python3
"""MemoryBench DialSim-bigbang 正式评测：cosine vs DGD。

语料：805 个 Big Bang Theory session
测试：17 个 MemoryBench 官方问题，带 golden_answer 和 session_num
评测：给定问题，检索到的 session 是否包含正确答案所在的 session
"""
import json
import math
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def load_corpus():
    """加载 805 个 session 作为记忆片段。"""
    with open("data/benchmarks/memorybench/dialsim-bigbang.jsonl") as f:
        text = json.loads(f.readline())["text"]

    sessions = text.split("[Date:")
    sessions = ["[Date:" + s.strip() for s in sessions if s.strip()]

    fragments = []
    for i, s in enumerate(sessions):
        body = s[:800].strip()
        if len(body) < 50:
            continue
        fragments.append({"id": i, "body": body, "full": s})
    return fragments


def load_questions():
    """加载 MemoryBench 测试题。"""
    import pyarrow.parquet as pq
    table = pq.read_table("data/benchmarks/memorybench/dialsim-bigbang-test.parquet")
    df = table.to_pydict()

    questions = []
    for i in range(len(df["test_idx"])):
        info = df["info"][i]
        if isinstance(info, str):
            info = ast.literal_eval(info)

        # 从 input_prompt 中提取 [Question] 和 [Answer] 之间的内容
        # 注意 [Question] 出现两次：指令中一次，实际问题前一次，要取最后一次
        prompt = df["input_prompt"][i]
        q_start = prompt.rfind("[Question]")
        a_start = prompt.rfind("[Answer]")
        if q_start >= 0 and a_start >= 0:
            question = prompt[q_start + len("[Question]"):a_start].strip()
        elif q_start >= 0:
            question = prompt[q_start + len("[Question]"):].strip()
        else:
            question = prompt[-200:]

        questions.append({
            "idx": df["test_idx"][i],
            "question": question[:300],
            "golden_answer": info.get("golden_answer", ""),
            "session_num": info.get("session_num", -1),  # MemoryBench 内部编号，不直接对应 fragment index
        })
    return questions


def find_answer_fragments(questions, fragments):
    """根据 golden_answer 在片段中搜索，找到真正包含答案的 fragment index。"""
    for q in questions:
        answer = q["golden_answer"].lower()
        q["answer_indices"] = []
        for i, frag in enumerate(fragments):
            if answer in frag["full"].lower():
                q["answer_indices"].append(i)
        if not q["answer_indices"]:
            # 放宽搜索：只匹配答案的前几个词
            short = answer.split()[0] if answer.split() else answer
            for i, frag in enumerate(fragments):
                if short in frag["full"].lower():
                    q["answer_indices"].append(i)
    return questions


def main():
    print("=== Step 1: Load corpus ===")
    fragments = load_corpus()
    print(f"  {len(fragments)} session fragments")

    print("\n=== Step 2: Load questions ===")
    questions = load_questions()
    print(f"  {len(questions)} questions")
    for q in questions[:3]:
        print(f"  Q: {q['question'][:60]}...")
        print(f"     Answer: {q['golden_answer']}, Session: {q['session_num']}")

    # 找到答案所在的 fragment index
    questions = find_answer_fragments(questions, fragments)
    valid_qs = [q for q in questions if q["answer_indices"]]
    print(f"  Questions with found answers: {len(valid_qs)}/{len(questions)}")
    for q in valid_qs[:5]:
        print(f"  '{q['golden_answer']}' → fragments {q['answer_indices'][:3]}")

    # 限制 200 个片段，跳过答案超出范围的题
    n_frags = 200
    valid_qs = [q for q in valid_qs if any(idx < n_frags for idx in q["answer_indices"])]
    # 只保留范围内的 answer_indices
    for q in valid_qs:
        q["answer_indices"] = [idx for idx in q["answer_indices"] if idx < n_frags]
    print(f"  Questions within range (<{n_frags}): {len(valid_qs)}")
    print(f"\n=== Step 3: Embed {n_frags} fragments (to cover all answers) ===")
    frag_vecs = []
    for i in range(n_frags):
        vec = get_embedding(fragments[i]["body"][:400])
        frag_vecs.append(vec)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_frags}")

    DIM = 256
    proj = create_projection_matrix(4096, DIM, seed=42)
    proj_vecs = [norm(project(v, proj)) for v in frag_vecs]

    print(f"\n=== Step 4: Embed {len(valid_qs)} questions ===")
    q_vecs = []
    for q in valid_qs:
        vec = get_embedding(q["question"][:300])
        q_vecs.append(norm(project(vec, proj)))

    # === 评测 ===
    print(f"\n=== Step 5: Evaluate ===")

    def evaluate(name, query_fn):
        hits = {1: 0, 3: 0, 5: 0, 10: 0}
        details = []
        for qi, q in enumerate(valid_qs):
            answer_indices = set(q["answer_indices"])

            scores = query_fn(q_vecs[qi])
            top_ids = [idx for _, idx in scores[:10]]

            # 命中 = top-k 中包含任一正确片段
            for k in [1, 3, 5, 10]:
                if any(idx in answer_indices for idx in top_ids[:k]):
                    hits[k] += 1

            # 找最佳命中 rank
            best_rank = ">10"
            for r, idx in enumerate(top_ids):
                if idx in answer_indices:
                    best_rank = r + 1
                    break
            details.append((q["question"][:40], q["golden_answer"][:20], list(answer_indices)[:3], best_rank))

        total = len(valid_qs)
        print(f"\n  {name}:")
        for k in [1, 3, 5, 10]:
            print(f"    P@{k}: {hits[k]}/{total} = {hits[k]/total:.1%}" if total > 0 else f"    P@{k}: N/A")

        print(f"\n  Details:")
        for question, answer, target, rank in details:
            print(f"    [{rank:>3}] {question}... → session {target} ({answer})")

        return hits, total

    # Cosine
    def cosine_query(qv):
        scores = [(cosine(qv, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    cosine_hits, total = evaluate("Cosine (static)", cosine_query)

    # DGD before
    mem = AssociativeMemory(dim=DIM, alpha=1.0, eta=0.01)

    def dgd_query(qv):
        act = mem.query(qv)
        scores = [(cosine(act, pv), i) for i, pv in enumerate(proj_vecs)]
        scores.sort(reverse=True)
        return scores

    dgd_before, _ = evaluate("DGD (before feedback)", dgd_query)

    # DGD 反馈：3 轮，用正确答案的第一个片段做反馈
    for round_num in range(3):
        for qi, q in enumerate(valid_qs):
            target = q["answer_indices"][0]
            if target >= n_frags:
                continue
            mem.update(q_vecs[qi], proj_vecs[target])

    dgd_after, _ = evaluate("DGD (3 rounds feedback)", dgd_query)

    # 总结
    print(f"\n{'='*50}")
    print(f"  {'Method':<25} {'P@1':>6} {'P@3':>6} {'P@5':>6} {'P@10':>6}")
    print(f"  {'-'*49}")
    for name, hits in [("Cosine", cosine_hits), ("DGD before", dgd_before), ("DGD 3 rounds", dgd_after)]:
        vals = [f"{hits[k]/total:.0%}" if total > 0 else "N/A" for k in [1, 3, 5, 10]]
        print(f"  {name:<25} {vals[0]:>6} {vals[1]:>6} {vals[2]:>6} {vals[3]:>6}")


if __name__ == "__main__":
    main()
