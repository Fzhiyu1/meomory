#!/usr/bin/env python3
"""完整记忆闭环 demo：查询 → 注入 → agent 回复 → 自动反馈 → DGD 更新。

用 GPT-5.4 作为 agent，演示整个循环。
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("MEOMORY_LLM_KEY", "")

from src.store import VectorStore
from src.embedder import get_embedding
from src.projection import project
from src.dgd import AssociativeMemory
from src.feedback import auto_feedback, _normalize, _cosine
from src.llm_client import chat

ROOT = Path(__file__).parent.parent
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"
DGD_PATH = ROOT / "memory" / "vectors" / "dgd.json"
PROJ_PATH = ROOT / "memory" / "vectors" / "proj.json"


def load_all():
    store = VectorStore.load(VECTOR_PATH)
    mem = AssociativeMemory.load(DGD_PATH)
    with open(PROJ_PATH) as f:
        proj = json.load(f)

    entries = store._entries
    proj_vectors = [_normalize(project(e["vector"], proj)) for e in entries]
    return store, mem, proj, entries, proj_vectors


def dgd_match(q_proj, mem, entries, proj_vectors, top_k=3, threshold=0.28):
    activation = mem.query(q_proj)
    scored = []
    for i, pv in enumerate(proj_vectors):
        sim = _cosine(activation, pv)
        if sim >= threshold:
            meta = entries[i]["meta"]
            scored.append({
                "index": i,
                "score": sim,
                "title": meta.get("title", entries[i]["id"]),
                "body": meta.get("body", "")[:200],
                "proj_vec": pv,
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_injection(memories):
    if not memories:
        return ""
    lines = ["[以下是系统自动匹配的相关记忆，供参考]"]
    for m in memories:
        lines.append(f"- {m['title']}: {m['body'][:150]}")
    return "\n".join(lines)


def one_cycle(query, mem, proj, entries, proj_vectors):
    """一次完整的记忆循环。"""
    print(f"\n{'='*60}")
    print(f"用户: {query}")

    # 1. DGD 匹配
    q_vec = get_embedding(query)
    q_proj = _normalize(project(q_vec, proj))
    memories = dgd_match(q_proj, mem, entries, proj_vectors)

    print(f"\n[匹配到 {len(memories)} 条记忆]")
    for i, m in enumerate(memories):
        print(f"  {i+1}. [{m['score']:.3f}] {m['title']}")

    # 2. 注入 + agent 回复
    injection = format_injection(memories)
    system = "你是一个 AI 助手。如果系统提供了相关记忆，请参考它们回答。简洁回复。"
    prompt = f"{injection}\n\n用户问题：{query}" if injection else query

    print(f"\n[调用 GPT-5.4 回复...]")
    response = chat(prompt, system=system, max_tokens=300)
    print(f"\nAgent: {response}")

    # 3. 自动反馈
    if memories:
        fb = auto_feedback(query, response, memories, mem, proj)
        print(f"\n[自动反馈]")
        for item in fb:
            status = "✓ 强化" if item["used"] else "- 未用"
            print(f"  {status} [{item['sim']:.3f}] {item['title']}")

    return memories, response


def main():
    print("=== meomory 完整记忆闭环 demo ===")
    print("输入问题，观察：匹配 → 注入 → 回复 → 自动反馈 → DGD 更新")
    print("输入 q 退出并保存\n")

    store, mem, proj, entries, proj_vectors = load_all()
    cycle_count = 0

    while True:
        try:
            query = input("\n查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() == "q":
            break

        one_cycle(query, mem, proj, entries, proj_vectors)
        cycle_count += 1

    if cycle_count > 0:
        mem.save(DGD_PATH)
        print(f"\n已保存。完成 {cycle_count} 轮记忆循环。")
    else:
        print("\n无更新。")


if __name__ == "__main__":
    main()
