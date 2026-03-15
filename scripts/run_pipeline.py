#!/usr/bin/env python3
"""meomory 端到端管线：从 raw session 到可查询的记忆库"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parser import parse_all_sessions
from src.compressor import compress_session_to_l1, compress_l1_to_l0
from src.embedder import get_embedding
from src.store import VectorStore
from src.matcher import MemoryMatcher

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
L1_DIR = ROOT / "memory" / "l1"
L0_DIR = ROOT / "memory" / "l0"
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"


def main():
    print("=" * 60)
    print("meomory 端到端管线")
    print("=" * 60)

    # ── 1. 解析所有 sessions ──
    print("\n[1/7] 解析 raw sessions...")
    sessions = parse_all_sessions(RAW_DIR)
    print(f"  解析完成: {len(sessions)} 个 session")

    # ── 2. 过滤空 session ──
    sessions = [s for s in sessions if s["turn_count"] > 0]
    print(f"  过滤后（turn_count > 0）: {len(sessions)} 个 session")

    # ── 3. L2→L1 压缩 ──
    print("\n[2/7] L2→L1 压缩...")
    L1_DIR.mkdir(parents=True, exist_ok=True)
    all_l1_fragments: list[dict] = []
    # 记录每个 session 的 L1 片段，供 L0 压缩使用
    session_l1_map: dict[str, list[dict]] = {}

    for sess in sessions:
        frags = compress_session_to_l1(sess)
        all_l1_fragments.extend(frags)
        session_l1_map[sess["id"]] = frags

        # 保存到 memory/l1/<session_id_prefix>.json
        sid = sess["id"]
        out_path = L1_DIR / f"{sid[:8]}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(frags, f, ensure_ascii=False, indent=2)

    print(f"  L1 片段总数: {len(all_l1_fragments)}")
    print(f"  保存到: {L1_DIR}/")

    # ── 4. L1→L0 压缩 ──
    print("\n[3/7] L1→L0 压缩...")
    L0_DIR.mkdir(parents=True, exist_ok=True)
    l0_summaries: list[dict] = []

    for sess in sessions:
        frags = session_l1_map.get(sess["id"], [])
        l0 = compress_l1_to_l0(frags, sess)
        l0_summaries.append(l0)

        sid = sess["id"]
        out_path = L0_DIR / f"{sid[:8]}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(l0, f, ensure_ascii=False, indent=2)

    print(f"  L0 摘要总数: {len(l0_summaries)}")
    print(f"  保存到: {L0_DIR}/")

    # ── 5. 向量编码 ──
    print("\n[4/7] 向量编码 L1 片段...")
    store = VectorStore()
    total = len(all_l1_fragments)
    success_count = 0
    fail_count = 0
    t0 = time.time()

    for i, frag in enumerate(all_l1_fragments):
        try:
            vec = get_embedding(frag["body"])
            meta = {k: v for k, v in frag.items() if k != "id"}
            store.add(frag["id"], vec, meta)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"  WARNING: 跳过 {frag['id']}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            eta = avg * (total - i - 1)
            print(f"  进度: {i + 1}/{total}  成功: {success_count}  失败: {fail_count}  "
                  f"耗时: {elapsed:.1f}s  预计剩余: {eta:.0f}s")

    print(f"  编码完成: {success_count} 成功, {fail_count} 失败")

    # ── 6. 保存向量库 ──
    print("\n[5/7] 保存向量库...")
    VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(VECTOR_PATH)
    print(f"  已保存: {VECTOR_PATH} ({len(store)} 条记录)")

    # ── 7. 验证匹配 ──
    print("\n[6/7] 验证匹配...")
    matcher = MemoryMatcher(store)
    test_queries = [
        "飞书机器人不回复了",
        "nginx 反代配置",
        "记忆系统设计",
    ]

    for query in test_queries:
        print(f"\n  查询: \"{query}\"")
        try:
            results = matcher.match(query, top_k=3)
            for j, r in enumerate(results):
                score = r["score"]
                body_preview = r.get("body", r.get("meta", {}).get("body", ""))[:80]
                cues = r.get("cue", r.get("meta", {}).get("cue", []))
                if isinstance(cues, list):
                    cues = ", ".join(cues[:5])
                print(f"    #{j + 1} [score={score:.4f}] [{cues}] {body_preview}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("[7/7] 总结")
    print(f"  Raw sessions: {len(sessions)}")
    print(f"  L1 片段: {len(all_l1_fragments)}")
    print(f"  L0 摘要: {len(l0_summaries)}")
    print(f"  向量库: {len(store)} 条")
    print(f"  总耗时: {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
