#!/usr/bin/env python3
"""睡眠巩固脚本：扫描知识库，用 LLM 分析，输出整理建议。

用法：
  python3 scripts/sleep_consolidation.py /path/to/knowledge-base [--execute]

不带 --execute：只输出建议 JSON
带 --execute：在知识库上执行建议（应在 worktree 上操作）
"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

API_URL = os.environ.get("MEOMORY_LLM_URL", "https://cliproxy.fzhiyu2333.top/v1")
API_KEY = os.environ.get("MEOMORY_LLM_KEY", "")
MODEL = os.environ.get("MEOMORY_LLM_MODEL", "gpt-5.4")

import httpx


async def llm_chat(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{API_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ─── Step 1: 扫描 ─────────────────────────────

def scan_kb(kb_path: Path) -> dict:
    """扫描知识库，提取所有卡片的元信息。"""
    cards = []
    for folder in ["1-concepts", "2-explorations", "3-projects", "4-references"]:
        folder_path = kb_path / folder
        if not folder_path.exists():
            continue
        for f in sorted(folder_path.glob("*.md")):
            content = f.read_text(encoding="utf-8")
            # 提取 tags
            tag_match = re.search(r"tags:\s*\[([^\]]*)\]", content)
            tags = [t.strip() for t in tag_match.group(1).split(",") if t.strip()] if tag_match else []
            # 提取链接
            links = list(set(re.findall(r"\[\[([^\]|]+)", content)))
            # body（去掉 frontmatter）
            body = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()

            cards.append({
                "name": f.stem,
                "path": str(f.relative_to(kb_path)),
                "folder": folder,
                "tags": tags,
                "links": links,
                "body": body[:1500],  # 截断，给 LLM 看
                "body_len": len(body),
            })

    # 统计链接
    all_names = {c["name"] for c in cards}
    link_counts = Counter()
    for c in cards:
        for link in c["links"]:
            link_counts[link] += 1

    broken_links = []
    for link, count in link_counts.items():
        if link not in all_names:
            # 检查是否是文件名空格问题
            no_space = link.replace(" ", "")
            match = no_space in {n.replace(" ", "") for n in all_names}
            broken_links.append({"link": link, "count": count, "fuzzy_match": match})

    orphans = [c["name"] for c in cards if c["folder"] == "1-concepts" and link_counts.get(c["name"], 0) == 0]

    return {
        "cards": cards,
        "broken_links": broken_links,
        "orphans": orphans,
        "total": len(cards),
    }


# ─── Step 2: LLM 分析 ─────────────────────────

ANALYSIS_SYSTEM = """你是一个知识库整理专家。你的任务是分析一个 Zettelkasten 知识库，找出需要整理的地方。

你要找以下几类问题：

1. **可合并的卡片**：内容高度重叠的卡片，应该合并为一张
2. **缺失的链接**：两张卡片明显相关但没有互相 [[链接]]
3. **可提炼的概念**：探索文档中反复出现但还没有独立概念卡片的想法
4. **过时的内容**：和知识库其他部分矛盾或已被新内容取代的描述
5. **格式问题**：缺少关联概念章节、标签不准确等

输出严格的 JSON 格式，不要其他内容：
{
  "suggestions": [
    {
      "type": "merge|link|extract|update|fix",
      "priority": "high|medium|low",
      "cards": ["卡片名1", "卡片名2"],
      "description": "具体建议",
      "action": "具体操作描述"
    }
  ]
}"""


async def analyze_batch(cards: list[dict], batch_name: str) -> list[dict]:
    """让 LLM 分析一批卡片。"""
    cards_text = ""
    for c in cards:
        cards_text += f"\n### {c['name']} ({c['folder']})\n"
        cards_text += f"Tags: {c['tags']}\n"
        cards_text += f"Links: {c['links']}\n"
        cards_text += f"Body ({c['body_len']} chars):\n{c['body'][:800]}\n"
        cards_text += "---\n"

    prompt = f"以下是知识库的 {batch_name} 部分，请分析需要整理的地方：\n{cards_text}"

    try:
        raw = await llm_chat(prompt, system=ANALYSIS_SYSTEM, max_tokens=3000)
        # 解析 JSON
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return result.get("suggestions", [])
    except Exception as e:
        print(f"  Warning: LLM analysis failed for {batch_name}: {e}")
        return []


async def analyze_cross_references(cards: list[dict]) -> list[dict]:
    """分析跨类别的关联缺失。"""
    # 只给 LLM 看名字和标签，不给全文（太长）
    summary = ""
    for c in cards:
        summary += f"- {c['name']} ({c['folder']}) tags={c['tags']} links={c['links'][:5]}\n"

    prompt = f"""以下是知识库所有卡片的概览。请找出：
1. 应该互相链接但没有链接的卡片对
2. 探索文档中可能需要提炼成独立概念卡片的主题

卡片列表：
{summary}"""

    try:
        raw = await llm_chat(prompt, system=ANALYSIS_SYSTEM, max_tokens=3000)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        return result.get("suggestions", [])
    except Exception as e:
        print(f"  Warning: cross-reference analysis failed: {e}")
        return []


# ─── Step 3: 结构问题自动检测 ─────────────────

def detect_structural_issues(scan_result: dict) -> list[dict]:
    """自动检测不需要 LLM 的结构问题。"""
    suggestions = []

    # 断链修复
    for bl in scan_result["broken_links"]:
        if bl["fuzzy_match"]:
            suggestions.append({
                "type": "fix",
                "priority": "high",
                "cards": [bl["link"]],
                "description": f"链接 [[{bl['link']}]] 被引用 {bl['count']} 次但不存在（文件名空格不匹配）",
                "action": f"统一链接名和文件名",
            })
        else:
            suggestions.append({
                "type": "fix",
                "priority": "medium",
                "cards": [bl["link"]],
                "description": f"链接 [[{bl['link']}]] 被引用 {bl['count']} 次但不存在",
                "action": "创建对应卡片或修正链接名",
            })

    # 孤立卡片
    for orphan in scan_result["orphans"]:
        suggestions.append({
            "type": "link",
            "priority": "medium",
            "cards": [orphan],
            "description": f"概念卡片 {orphan} 没有被任何其他卡片引用",
            "action": "检查是否需要在相关卡片中添加链接",
        })

    return suggestions


# ─── Main ──────────────────────────────────────

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/sleep_consolidation.py /path/to/knowledge-base [--execute]")
        sys.exit(1)

    kb_path = Path(sys.argv[1])
    execute = "--execute" in sys.argv

    print("=== 睡眠巩固：扫描知识库 ===\n")

    # Step 1: 扫描
    print("Step 1: 扫描...")
    scan = scan_kb(kb_path)
    print(f"  {scan['total']} 张卡片")
    print(f"  {len(scan['broken_links'])} 个断链")
    print(f"  {len(scan['orphans'])} 个孤立卡片")

    # Step 2: 结构问题
    print("\nStep 2: 检测结构问题...")
    structural = detect_structural_issues(scan)
    print(f"  {len(structural)} 个结构问题")

    # Step 3: LLM 分析
    print("\nStep 3: LLM 分析（GPT-5.4）...")
    all_suggestions = list(structural)

    # 按文件夹分批分析
    by_folder = {}
    for c in scan["cards"]:
        folder = c["folder"]
        if folder not in by_folder:
            by_folder[folder] = []
        by_folder[folder].append(c)

    for folder, cards in by_folder.items():
        print(f"  分析 {folder}（{len(cards)} 张）...")
        suggestions = await analyze_batch(cards, folder)
        all_suggestions.extend(suggestions)

    # 跨类别分析
    print("  分析跨类别关联...")
    cross = await analyze_cross_references(scan["cards"])
    all_suggestions.extend(cross)

    # Step 4: 输出
    print(f"\n=== 共 {len(all_suggestions)} 条建议 ===\n")

    # 按优先级排序
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))

    for i, s in enumerate(all_suggestions):
        emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(s.get("priority", "low"), "⚪")
        print(f"  {emoji} [{s['type']}] {s['description']}")
        print(f"     → {s['action']}")
        if s.get("cards"):
            print(f"     涉及: {s['cards']}")
        print()

    # 保存建议
    output_path = kb_path / "consolidation_suggestions.json"
    with open(output_path, "w") as f:
        json.dump({"suggestions": all_suggestions, "scan": {
            "total": scan["total"],
            "broken_links": len(scan["broken_links"]),
            "orphans": scan["orphans"],
        }}, f, ensure_ascii=False, indent=2)
    print(f"建议已保存到: {output_path}")

    if execute:
        print("\n=== 执行建议（TODO）===")
        print("暂未实现自动执行，请手动审批后执行。")


if __name__ == "__main__":
    asyncio.run(main())
