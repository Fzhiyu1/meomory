#!/usr/bin/env python3
"""从知识库导入 markdown 文档，生成 L1 片段并重建向量库。

知识库的概念卡片和探索文档本身就是高质量的压缩产物，
不需要再用 LLM 压缩——直接用标题+正文做 L1，提取 [[链接]] 和 tags 做 cue。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedder import get_embedding
from src.store import VectorStore
from src.matcher import MemoryMatcher

KB_DIR = Path("/Users/fangzhiyu/run/knowledge-base")
ROOT = Path(__file__).parent.parent
L1_DIR = ROOT / "memory" / "l1"
VECTOR_PATH = ROOT / "memory" / "vectors" / "l1.json"

# 跳过的路径
SKIP_PATHS = {".obsidian", ".claude", "node_modules"}
SKIP_FILES = {"CLAUDE.md", "README.md"}


def parse_kb_file(path: Path) -> dict | None:
    """解析一个知识库 markdown 文件为 L1 片段。"""
    text = path.read_text(encoding="utf-8")

    # 解析 YAML frontmatter
    tags = []
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            body = parts[2].strip()
            # 提取 tags
            tag_match = re.search(r"tags:\s*\[([^\]]*)\]", frontmatter)
            if tag_match:
                tags = [t.strip() for t in tag_match.group(1).split(",") if t.strip()]

    if not body.strip():
        return None

    # 提取标题
    title_match = re.match(r"#\s+(.+)", body)
    title = title_match.group(1).strip() if title_match else path.stem

    # 提取 [[双向链接]] 作为 cue
    links = re.findall(r"\[\[([^\]]+)\]\]", body)
    # 去重保序
    seen = set()
    unique_links = []
    for link in links:
        clean = link.split("|")[0].strip()  # [[显示名|链接名]] 格式
        if clean not in seen:
            seen.add(clean)
            unique_links.append(clean)

    # 组装 cue：标题 + tags + 链接目标
    cues = [title] + tags + unique_links[:10]

    # 判断所属层级
    rel = path.relative_to(KB_DIR)
    category = rel.parts[0] if rel.parts else "other"

    # body 截断到合理长度用于 embedding
    # 概念卡片通常不长，探索文档取前 500 字符
    embed_body = body[:1000] if len(body) > 1000 else body

    return {
        "id": f"kb-{path.stem}",
        "cue": cues,
        "ts": "",  # 可以从 frontmatter 提取 date
        "source": f"knowledge-base/{rel}",
        "category": category,
        "title": title,
        "body": embed_body,
        "body_full": body,  # 保留完整内容用于注入
    }


def main():
    # Step 1: 扫描知识库
    print("=== Step 1: Scanning knowledge base ===")
    fragments = []
    for path in sorted(KB_DIR.rglob("*.md")):
        # 跳过
        if any(skip in path.parts for skip in SKIP_PATHS):
            continue
        if path.name in SKIP_FILES:
            continue

        frag = parse_kb_file(path)
        if frag:
            fragments.append(frag)

    print(f"  Found {len(fragments)} documents:")
    categories = {}
    for f in fragments:
        cat = f["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")

    # Step 2: 保存 L1 片段
    print(f"\n=== Step 2: Saving {len(fragments)} L1 fragments ===")
    # 清空旧的
    if L1_DIR.exists():
        for old in L1_DIR.glob("*.json"):
            old.unlink()
    L1_DIR.mkdir(parents=True, exist_ok=True)

    for frag in fragments:
        # 保存时不包含 body_full（太大）
        save_frag = {k: v for k, v in frag.items() if k != "body_full"}
        path = L1_DIR / f"{frag['id']}.json"
        with open(path, "w") as f:
            json.dump(save_frag, f, ensure_ascii=False, indent=2)

    # Step 3: 向量编码
    print(f"\n=== Step 3: Embedding {len(fragments)} fragments ===")
    store = VectorStore()
    for i, frag in enumerate(fragments):
        # 用 标题 + cue + body前段 做 embedding
        embed_text = f"{frag['title']}\n{', '.join(frag['cue'][:5])}\n{frag['body'][:500]}"
        try:
            vec = get_embedding(embed_text)
            store.add(frag["id"], vec, {
                "body": frag["body"][:300],  # 注入时用截断版
                "cue": frag["cue"],
                "title": frag["title"],
                "source": frag["source"],
                "category": frag["category"],
            })
            if (i + 1) % 10 == 0:
                print(f"  Embedded {i + 1}/{len(fragments)}")
        except Exception as e:
            print(f"  Warning: embed failed for {frag['id']}: {e}")

    VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    store.save(VECTOR_PATH)
    print(f"  Saved {len(store)} vectors to {VECTOR_PATH}")

    # Step 4: 验证
    print("\n=== Step 4: Validation ===")
    matcher = MemoryMatcher(store)
    queries = [
        "记忆系统怎么设计",
        "遗忘是怎么回事",
        "Hope 模型",
        "飞书机器人",
        "自由能原理",
        "搅屎棍",
    ]
    for q in queries:
        results = matcher.match(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for r in results:
            title = r.get("title", "?")
            score = r.get("score", 0)
            cat = r.get("category", "?")
            print(f"    [{score:.3f}] [{cat}] {title}")

    print(f"\n=== Done: {len(store)} fragments indexed ===")


if __name__ == "__main__":
    main()
