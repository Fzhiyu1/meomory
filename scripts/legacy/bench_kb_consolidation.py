#!/usr/bin/env python3
"""实验 C：知识库巩固前后检索质量对比。

pre-consolidation vs master，用 LLM 生成测试题，对比 P@1。
"""
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

API_URL = os.environ.get("MEOMORY_LLM_URL", "https://api.deepseek.com/v1")
API_KEY = os.environ.get("MEOMORY_LLM_KEY", "")
MODEL = os.environ.get("MEOMORY_LLM_MODEL", "deepseek-chat")

import httpx
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


async def llm_chat(prompt, system="", max_tokens=500):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{API_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def load_kb_cards(kb_path: Path) -> list[dict]:
    """加载知识库的概念卡片。"""
    cards = []
    concepts_dir = kb_path / "1-concepts"
    if not concepts_dir.exists():
        return cards
    for f in sorted(concepts_dir.glob("*.md")):
        content = f.read_text(encoding="utf-8")
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
        if len(body) < 50:
            continue
        cards.append({
            "name": f.stem,
            "body": body[:1000],
            "full": body,
        })
    return cards


async def generate_questions(cards: list[dict], n_per_card: int = 2) -> list[dict]:
    """用 LLM 为每张卡片生成测试题。"""
    system = """为给定的知识卡片生成检索测试题。每题应该：
1. 只有这张卡片能回答（不是泛泛的问题）
2. 用不同于卡片原文的措辞（测试语义理解，不是关键词匹配）
3. 简短，一句话

输出 JSON 数组，每个元素 {"question": "..."}。只输出 JSON，不要其他内容。"""

    all_questions = []

    async def gen_one(card):
        prompt = f"为以下知识卡片生成 {n_per_card} 个检索测试题：\n\n标题：{card['name']}\n内容：{card['body'][:500]}"
        try:
            raw = await llm_chat(prompt, system=system, max_tokens=300)
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            questions = json.loads(text)
            return [(q["question"], card["name"]) for q in questions if "question" in q]
        except Exception as e:
            print(f"  Warning: failed for {card['name']}: {e}")
            return []

    # 并发生成
    tasks = [gen_one(card) for card in cards]
    results = await asyncio.gather(*tasks)
    for pairs in results:
        for question, answer_name in pairs:
            all_questions.append({"question": question, "answer_name": answer_name})

    return all_questions


def embed_and_evaluate(cards, questions, label):
    """Embedding + cosine 检索评测。"""
    print(f"\n  [{label}] Embedding {len(cards)} cards...")
    card_vecs = []
    for i, card in enumerate(cards):
        # 用 标题 + body 前段做 embedding
        text = f"{card['name']}\n{card['body'][:500]}"
        card_vecs.append(get_embedding(text))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(cards)}")

    print(f"  [{label}] Embedding {len(questions)} questions...")
    q_vecs = [get_embedding(q["question"]) for q in questions]

    # Cosine 检索
    hits = {1: 0, 3: 0, 5: 0}
    details = []
    for qi, q in enumerate(questions):
        scores = [(cosine(q_vecs[qi], cv), i) for i, cv in enumerate(card_vecs)]
        scores.sort(reverse=True)
        top_names = [cards[idx]["name"] for _, idx in scores[:5]]

        for k in [1, 3, 5]:
            if q["answer_name"] in top_names[:k]:
                hits[k] += 1

        rank = top_names.index(q["answer_name"]) + 1 if q["answer_name"] in top_names else ">5"
        details.append({"question": q["question"][:50], "answer": q["answer_name"], "rank": rank})

    total = len(questions)
    metrics = {k: hits[k] / total if total > 0 else 0 for k in hits}
    print(f"  [{label}] P@1={metrics[1]:.1%} P@3={metrics[3]:.1%} P@5={metrics[5]:.1%}")

    return metrics, details


async def main():
    import subprocess

    kb_path = Path("/Users/fangzhiyu/run/knowledge-base")
    worktree_path = Path("/Users/fangzhiyu/run/kb-consolidation")

    print("=== 实验 C：知识库巩固前后检索质量对比 ===\n")

    # Step 1: 加载两个版本
    print("Step 1: 加载知识库...")

    # 巩固后（master / worktree）
    cards_after = load_kb_cards(worktree_path)
    print(f"  巩固后: {len(cards_after)} 张概念卡")

    # 巩固前：从 pre-consolidation 分支读
    # 用 git ls-tree -z 避免引号转义问题
    result = subprocess.run(
        ["git", "-C", str(kb_path), "ls-tree", "--name-only", "-z", "pre-consolidation", "1-concepts/"],
        capture_output=True, text=True
    )
    pre_files = [f for f in result.stdout.split("\0") if f.strip().endswith(".md")]

    cards_before = []
    for fpath in pre_files:
        result = subprocess.run(
            ["git", "-C", str(kb_path), "show", f"pre-consolidation:{fpath}"],
            capture_output=True, text=True
        )
        content = result.stdout
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()
        if len(body) < 50:
            continue
        name = Path(fpath).stem
        cards_before.append({"name": name, "body": body[:1000], "full": body})

    print(f"  巩固前: {len(cards_before)} 张概念卡")

    # 找共同卡片（两个版本都有的）
    names_before = {c["name"] for c in cards_before}
    names_after = {c["name"] for c in cards_after}
    common = names_before & names_after
    only_after = names_after - names_before
    print(f"  共同: {len(common)}, 巩固后新增: {len(only_after)}")
    if only_after:
        print(f"    新增: {only_after}")

    # Step 2: 生成测试题（基于巩固后的版本，但只用共同卡片出题）
    print(f"\nStep 2: 生成测试题...")
    common_cards = [c for c in cards_after if c["name"] in common]
    questions = await generate_questions(common_cards, n_per_card=2)
    print(f"  生成了 {len(questions)} 个测试题")

    # 保存题目
    q_path = Path("experiments/results/kb-questions.json")
    with open(q_path, "w") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    # Step 3: 分别评测
    print(f"\nStep 3: 评测...")

    # 只用共同卡片评测（公平对比）
    cards_before_common = [c for c in cards_before if c["name"] in common]
    cards_after_common = [c for c in cards_after if c["name"] in common]
    # 排序保证顺序一致
    cards_before_common.sort(key=lambda c: c["name"])
    cards_after_common.sort(key=lambda c: c["name"])

    metrics_before, details_before = embed_and_evaluate(cards_before_common, questions, "巩固前")
    metrics_after, details_after = embed_and_evaluate(cards_after_common, questions, "巩固后")

    # Step 4: 对比
    print(f"\n{'='*50}")
    print(f"  结果对比")
    print(f"{'='*50}")
    print(f"  {'指标':<10} {'巩固前':>10} {'巩固后':>10} {'差异':>10}")
    print(f"  {'-'*40}")
    for k in [1, 3, 5]:
        diff = metrics_after[k] - metrics_before[k]
        sign = "+" if diff > 0 else ""
        print(f"  {'P@'+str(k):<10} {metrics_before[k]:>9.1%} {metrics_after[k]:>9.1%} {sign}{diff:>8.1%}")

    # 逐题对比（哪些题排名变了）
    print(f"\n  排名变化详情:")
    improved = 0
    degraded = 0
    for db, da in zip(details_before, details_after):
        rb = db["rank"] if isinstance(db["rank"], int) else 99
        ra = da["rank"] if isinstance(da["rank"], int) else 99
        if ra < rb:
            improved += 1
            print(f"    ↑ [{db['answer']}] {rb}→{ra}  Q: {db['question']}")
        elif ra > rb:
            degraded += 1
            print(f"    ↓ [{db['answer']}] {rb}→{ra}  Q: {db['question']}")

    print(f"\n  改善: {improved} 题, 退化: {degraded} 题, 不变: {len(questions)-improved-degraded} 题")

    # 保存结果
    result = {
        "before": {"cards": len(cards_before_common), "metrics": metrics_before},
        "after": {"cards": len(cards_after_common), "metrics": metrics_after},
        "questions": len(questions),
        "improved": improved,
        "degraded": degraded,
    }
    with open("experiments/results/kb-consolidation-comparison.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果保存到 experiments/results/kb-consolidation-comparison.json")


if __name__ == "__main__":
    asyncio.run(main())
