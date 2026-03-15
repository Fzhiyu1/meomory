"""LLM API 客户端 — 通过 cliproxy 调用 GPT-5.4"""
import os
import httpx

API_URL = os.environ.get("MEOMORY_LLM_URL", "https://cliproxy.fzhiyu2333.top/v1")
API_KEY = os.environ.get("MEOMORY_LLM_KEY", "")
MODEL = os.environ.get("MEOMORY_LLM_MODEL", "gpt-5.4")
TIMEOUT = 60.0


def chat(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    """单轮对话，返回 assistant 回复文本"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.post(
        f"{API_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
