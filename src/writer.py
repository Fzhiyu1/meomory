"""记忆写入模块：被动+主动混合写入，立刻可检索。

被动写入：自动调用，不做判断，全部进入输入层
主动写入：人触发，质量更高

所有写入立刻生效（embedding + 加入向量库）。
不设 TTL——遗忘只在巩固压缩中产生。
"""
import json
import time
from pathlib import Path
from src.embedder import get_embedding
from src.store import VectorStore


class MemoryWriter:
    def __init__(self, store: VectorStore, fragments_dir: Path | None = None):
        self.store = store
        self.fragments_dir = fragments_dir
        self._counter = 0

    def write(self, text: str, source: str = "passive", meta: dict | None = None) -> str:
        """写入一条记忆到输入层。立刻 embedding，立刻可检索。

        Args:
            text: 记忆内容
            source: "passive"（自动写入）或 "active"（人触发）
            meta: 额外元信息

        Returns:
            fragment_id
        """
        self._counter += 1
        frag_id = f"input-{int(time.time())}-{self._counter:04d}"

        # 立刻 embedding
        vec = get_embedding(text[:500])

        # 存入向量库
        entry_meta = {
            "body": text[:500],
            "source": source,
            "layer": "input",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "consolidated": False,
        }
        if meta:
            entry_meta.update(meta)

        self.store.add(frag_id, vec, entry_meta)

        # 可选：持久化到文件
        if self.fragments_dir:
            self.fragments_dir.mkdir(parents=True, exist_ok=True)
            path = self.fragments_dir / f"{frag_id}.json"
            with open(path, "w") as f:
                json.dump({"id": frag_id, "text": text, **entry_meta}, f, ensure_ascii=False, indent=2)

        return frag_id

    def write_conversation(self, user_msg: str, agent_reply: str, source: str = "passive") -> list[str]:
        """写入一轮对话（用户消息 + agent 回复）。

        Returns:
            [user_frag_id, agent_frag_id]
        """
        ids = []
        if user_msg.strip():
            ids.append(self.write(f"用户: {user_msg.strip()}", source=source, meta={"role": "user"}))
        if agent_reply.strip():
            ids.append(self.write(f"AI: {agent_reply.strip()}", source=source, meta={"role": "assistant"}))
        return ids
