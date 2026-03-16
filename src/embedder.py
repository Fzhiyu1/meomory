"""调用 qwen3-embedding 做向量编码。

通过 Ollama /api/embed 接口获取文本的 embedding 向量。
"""

import os
import httpx

OLLAMA_URL = os.environ.get("MEOMORY_EMBED_URL", "http://100.117.243.72:11435")
EMBED_MODEL = os.environ.get("MEOMORY_EMBED_MODEL", "qwen3-embedding")
EMBED_DIM = 4096
TIMEOUT = 600.0

# 显式创建无代理 transport，避免 HTTP_PROXY 环境变量干扰内网请求
_transport = httpx.HTTPTransport()
_client = httpx.Client(transport=_transport, timeout=TIMEOUT)


def get_embedding(text: str) -> list[float]:
    """获取单条文本的 embedding 向量。"""
    resp = _client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise ValueError(f"No embeddings returned: {data}")
    return embeddings[0]


def get_embeddings_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """批量获取 embedding 向量。分批发送避免单次请求过大。"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = _client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=600.0,
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json().get("embeddings", []))
    return all_embeddings
