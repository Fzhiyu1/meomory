"""调用 qwen3-embedding 做向量编码。

通过 Ollama /api/embed 接口获取文本的 embedding 向量。
"""

import httpx

OLLAMA_URL = "http://100.117.243.72:11435"
EMBED_MODEL = "qwen3-embedding"
EMBED_DIM = 4096
TIMEOUT = 60.0

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


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """批量获取 embedding 向量。"""
    resp = _client.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=TIMEOUT * 2,
    )
    resp.raise_for_status()
    return resp.json().get("embeddings", [])
