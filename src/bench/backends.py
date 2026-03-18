"""LLM 后端抽象：DeepSeek API / Ollama"""
import httpx


class LLMBackend:
    async def chat(self, prompt: str, system: str = "", max_tokens: int = 300) -> str:
        raise NotImplementedError


class DeepSeekBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "deepseek-chat", url: str = "https://api.deepseek.com/v1"):
        self.url = url
        self.api_key = api_key
        self.model = model

    async def chat(self, prompt, system="", max_tokens=300):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "max_tokens": max_tokens},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()


class OllamaBackend(LLMBackend):
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen3.5:9b"):
        self.host = host
        self.model = model

    async def chat(self, prompt, system="", max_tokens=300):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # Ollama 用无代理 transport
        transport = httpx.AsyncHTTPTransport()
        async with httpx.AsyncClient(transport=transport, timeout=120.0) as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False,
                      "think": False,
                      "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()


class AnthropicBackend(LLMBackend):
    """Anthropic API backend with retry + DeepSeek fallback."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-20250514",
                 url: str = "https://code.newcli.com/claude/droid/v1",
                 fallback: LLMBackend | None = None):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.fallback = fallback

    async def _call_anthropic(self, prompt, system="", max_tokens=300):
        messages = [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if system:
            body["system"] = system
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.url}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"].strip()

    async def chat(self, prompt, system="", max_tokens=300):
        # 2 次重试 Opus，失败降级到 fallback (DeepSeek)
        for attempt in range(2):
            try:
                return await self._call_anthropic(prompt, system, max_tokens)
            except Exception:
                if attempt == 0:
                    continue  # 第 1 次失败，重试
        # 2 次都失败，降级
        if self.fallback:
            return await self.fallback.chat(prompt, system, max_tokens)
        raise Exception("Anthropic failed after 2 retries, no fallback")


def create_backend(config: dict) -> LLMBackend:
    """从配置创建后端"""
    if config is None:
        return None
    t = config["type"]
    if t == "deepseek":
        return DeepSeekBackend(
            api_key=config.get("api_key") or config.get("key", ""),
            model=config.get("model", "deepseek-chat"),
            url=config.get("url", "https://api.deepseek.com/v1"),
        )
    elif t == "ollama":
        return OllamaBackend(
            host=config["host"],
            model=config.get("model", "qwen3.5:9b"),
        )
    else:
        raise ValueError(f"Unknown backend type: {t}")
