#!/usr/bin/env python3
"""实验 B：写入 + 巩固端到端验证。

Phase 1: 预填充 400 个片段作为"历史记忆"
Phase 2: 逐个写入 200 个新片段，每 10 个评测一次（新片段能不能立刻被找到）
Phase 3: 跑巩固压缩，对比前后检索质量
"""
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LLM_URL = os.environ.get("MEOMORY_LLM_URL", "https://api.deepseek.com/v1")
LLM_KEY = os.environ.get("MEOMORY_LLM_KEY", "")
LLM_MODEL = os.environ.get("MEOMORY_LLM_MODEL", "deepseek-chat")

import httpx
from src.store import VectorStore
from src.embedder import get_embedding, get_embeddings_batch
from src.writer import MemoryWriter
from src.consolidator import consolidate


def cosine(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na * nb > 0 else 0


async def llm_chat(prompt, system="", max_tokens=300):
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


def load_corpus(n=600):
    with open("data/benchmarks/memorybench/dialsim-bigbang.jsonl") as f:
        text = json.loads(f.readline())["text"]
    sessions = text.split("[Date:")
    sessions = ["[Date:" + s.strip() for s in sessions if s.strip()]
    fragments = []
    for i, s in enumerate(sessions[:n]):
        body = s[:800].strip()
        if len(body) < 50:
            continue
        fragments.append({"id": i, "body": body})
    return fragments


def load_questions():
    import ast
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
        ga = info.get("golden_answer", "")
        if isinstance(ga, dict):
            continue
        questions.append({"question": question[:300], "golden_answer": ga})
    return questions


def evaluate(store, questions, label, q_vecs_cache={}):
    """评测当前向量库的检索质量。缓存问题向量避免重复 embedding。"""
    # 缓存问题向量
    cache_key = tuple(q["question"] for q in questions)
    if cache_key not in q_vecs_cache:
        q_texts = [q["question"][:300] for q in questions]
        q_vecs_cache[cache_key] = get_embeddings_batch(q_texts, batch_size=50)

    q_vecs = q_vecs_cache[cache_key]
    hits = {1: 0, 3: 0, 5: 0}
    total_valid = 0

    for qi, q in enumerate(questions):
        q_vec = q_vecs[qi]
        results = store.query(q_vec, top_k=5)
        answer = q["golden_answer"].lower()

        # 检查 top-k 结果中是否包含答案
        found = {1: False, 3: False, 5: False}
        for rank, r in enumerate(results):
            body = r.get("meta", {}).get("body", r.get("body", "")).lower()
            if answer in body:
                for k in [1, 3, 5]:
                    if rank < k:
                        found[k] = True

        total_valid += 1
        for k in found:
            if found[k]:
                hits[k] += 1

    metrics = {k: hits[k] / total_valid if total_valid > 0 else 0 for k in hits}
    print(f"  [{label}] P@1={metrics[1]:.1%} P@3={metrics[3]:.1%} P@5={metrics[5]:.1%} ({total_valid} questions)")
    return metrics


async def main():
    print("=== 实验 B：写入 + 巩固端到端验证 ===\n")

    fragments = load_corpus(600)
    questions = load_questions()
    # 过滤有效问题（答案在前 600 个片段中）
    valid_questions = []
    for q in questions:
        answer = q["golden_answer"].lower()
        for f in fragments:
            if answer in f["body"].lower():
                valid_questions.append(q)
                break
    print(f"语料: {len(fragments)} 片段, 有效测试题: {len(valid_questions)}")

    # === Phase 1: 预填充（批量 embedding）===
    print(f"\n=== Phase 1: 批量预填充前 400 个片段 ===")
    store = VectorStore()

    prefill_texts = [f["body"][:500] for f in fragments[:400]]
    t0 = time.time()
    print(f"  批量 embedding {len(prefill_texts)} 条...")
    prefill_vecs = get_embeddings_batch(prefill_texts, batch_size=50)
    print(f"  Embedding 完成 ({time.time()-t0:.1f}s)")

    for i, frag in enumerate(fragments[:400]):
        frag_id = f"input-prefill-{i:04d}"
        store.add(frag_id, prefill_vecs[i], {
            "body": frag["body"][:500],
            "source": "prefill",
            "layer": "input",
            "original_id": frag["id"],
            "consolidated": False,
        })
    print(f"  预填充完成, 向量库: {len(store)} 条")

    writer = MemoryWriter(store)

    # 评测预填充状态
    metrics_prefill = evaluate(store, valid_questions, "预填充")

    # === Phase 2: 实时写入 ===
    print(f"\n=== Phase 2: 逐个写入第 401-600 个片段 ===")
    write_curve = []

    for i, frag in enumerate(fragments[400:600]):
        writer.write(frag["body"], source="passive", meta={"original_id": frag["id"]})

        if (i + 1) % 20 == 0:
            metrics = evaluate(store, valid_questions, f"写入 {400+i+1}")
            write_curve.append({"written": 400 + i + 1, "total": len(store), **metrics})

    print(f"  写入完成, 向量库: {len(store)} 条")

    # 评测写入后状态
    metrics_after_write = evaluate(store, valid_questions, "写入完成")

    # === Phase 3: 巩固压缩 ===
    print(f"\n=== Phase 3: 巩固压缩 ===")
    print(f"  输入层片段: {sum(1 for e in store._entries if e.get('meta', {}).get('layer') == 'input')}")

    report = await consolidate(store, llm_chat, min_group_size=2, similarity_threshold=0.55)
    print(f"  合并了 {report['merged']} 条 → {report['new_concepts']} 个概念片段")
    if report.get("msg"):
        print(f"  Note: {report['msg']}")
    for d in report.get("details", [])[:5]:
        if "compressed" in d:
            print(f"    {d['concept_id']}: {d['compressed'][:80]}...")

    print(f"  向量库: {len(store)} 条（含概念层）")

    # 评测巩固后状态
    metrics_after_consolidate = evaluate(store, valid_questions, "巩固后")

    # === 汇总 ===
    print(f"\n{'='*50}")
    print(f"  汇总")
    print(f"{'='*50}")
    print(f"  {'阶段':<15} {'P@1':>8} {'P@3':>8} {'P@5':>8} {'片段数':>8}")
    print(f"  {'-'*47}")
    print(f"  {'预填充(400)':<15} {metrics_prefill[1]:>7.1%} {metrics_prefill[3]:>7.1%} {metrics_prefill[5]:>7.1%} {400:>8}")
    print(f"  {'写入后(600)':<15} {metrics_after_write[1]:>7.1%} {metrics_after_write[3]:>7.1%} {metrics_after_write[5]:>7.1%} {len(store)-report.get('new_concepts',0):>8}")
    print(f"  {'巩固后':<15} {metrics_after_consolidate[1]:>7.1%} {metrics_after_consolidate[3]:>7.1%} {metrics_after_consolidate[5]:>7.1%} {len(store):>8}")

    # 写入曲线
    if write_curve:
        print(f"\n  写入曲线 (P@1):")
        for w in write_curve:
            print(f"    {w['written']} 片段: P@1={w[1]:.1%}")

    # 保存结果
    result = {
        "prefill": metrics_prefill,
        "after_write": metrics_after_write,
        "after_consolidate": metrics_after_consolidate,
        "write_curve": write_curve,
        "consolidation_report": report,
    }
    with open("experiments/results/write-consolidate.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到 experiments/results/write-consolidate.json")


if __name__ == "__main__":
    asyncio.run(main())
