#!/usr/bin/env python3
"""用 LLM 精压重新生成 L1/L0 片段，然后重建向量库。"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 设置 API key
os.environ.setdefault("MEOMORY_LLM_KEY", "")

from src.parser import parse_all_sessions
from src.compressor import compress_session_to_l1_llm, compress_l1_to_l0_llm
from src.embedder import get_embedding
from src.store import VectorStore
from src.matcher import MemoryMatcher

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
L1_DIR = ROOT / "memory" / "l1"
L0_DIR = ROOT / "memory" / "l0"
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"


def main():
    # Step 1: 解析
    print("=== Step 1: Parsing sessions ===")
    sessions = parse_all_sessions(RAW_DIR)
    sessions = [s for s in sessions if s["turn_count"] > 0]
    print(f"  {len(sessions)} non-empty sessions")

    # Step 2: LLM 精压 L2→L1
    print("\n=== Step 2: LLM Compressing L2→L1 ===")
    all_l1 = []
    for i, session in enumerate(sessions):
        print(f"  [{i+1}/{len(sessions)}] {session['id'][:8]} ({session['turn_count']} turns)...", end=" ", flush=True)
        t0 = time.time()
        fragments = compress_session_to_l1_llm(session)
        elapsed = time.time() - t0
        all_l1.extend(fragments)
        print(f"→ {len(fragments)} fragments ({elapsed:.1f}s)")

    # 过滤 SKIP
    all_l1 = [f for f in all_l1 if f.get("body") != "SKIP"]
    print(f"\n  Total L1 fragments (after SKIP filter): {len(all_l1)}")

    # 保存 L1
    # 清空旧的
    if L1_DIR.exists():
        for old in L1_DIR.glob("*.json"):
            old.unlink()
    L1_DIR.mkdir(parents=True, exist_ok=True)
    for frag in all_l1:
        path = L1_DIR / f"{frag['id']}.json"
        with open(path, "w") as f:
            json.dump(frag, f, ensure_ascii=False, indent=2)

    # Step 3: LLM 精压 L1→L0
    print("\n=== Step 3: LLM Compressing L1→L0 ===")
    if L0_DIR.exists():
        for old in L0_DIR.glob("*.json"):
            old.unlink()
    L0_DIR.mkdir(parents=True, exist_ok=True)
    l0_entries = []
    for i, session in enumerate(sessions):
        session_l1 = [f for f in all_l1 if f["source"] == f"session/{session['id']}"]
        if not session_l1:
            continue
        print(f"  [{i+1}/{len(sessions)}] {session['id'][:8]}...", end=" ", flush=True)
        l0 = compress_l1_to_l0_llm(session_l1, session)
        l0_entries.append(l0)
        print(f"→ {l0['body'][:50]}")

        path = L0_DIR / f"{l0['id']}.json"
        with open(path, "w") as f:
            json.dump(l0, f, ensure_ascii=False, indent=2)

    # 过滤 SKIP
    l0_entries = [e for e in l0_entries if e.get("body") != "SKIP"]
    print(f"\n  Total L0 entries (after SKIP filter): {len(l0_entries)}")

    # Step 4: 重建向量库
    print(f"\n=== Step 4: Rebuilding vector store ({len(all_l1)} fragments) ===")
    store = VectorStore()
    for i, frag in enumerate(all_l1):
        try:
            vec = get_embedding(frag["body"])
            store.add(frag["id"], vec, {
                "body": frag["body"],
                "cue": frag["cue"],
                "ts": frag["ts"],
                "source": frag["source"],
            })
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i+1}/{len(all_l1)}")
        except Exception as e:
            print(f"  Warning: embed failed for {frag['id']}: {e}")

    VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(VECTOR_PATH)
    print(f"  Saved {len(store)} vectors")

    # Step 5: 验证
    print("\n=== Step 5: Validation ===")
    matcher = MemoryMatcher(store)
    queries = ["飞书机器人不回复了", "nginx 反代配置", "记忆系统设计", "Human Memory Compression"]
    for q in queries:
        results = matcher.match(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for r in results:
            body = r.get("body", r.get("meta", {}).get("body", ""))
            print(f"    [{r['score']:.3f}] {body[:80]}")

    # 打印几条 L1 样本
    print("\n=== L1 样本 ===")
    for frag in all_l1[:5]:
        print(f"\n  [{frag['id']}] cue={frag['cue']}")
        print(f"  {frag['body']}")

    print(f"\n=== Done: {len(all_l1)} L1, {len(l0_entries)} L0 ===")


if __name__ == "__main__":
    main()
