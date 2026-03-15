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
                      "options": {"num_predict": max_tokens}},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()


def create_backend(config: dict) -> LLMBackend:
    """从配置创建后端"""
    if config is None:
        return None
    t = config["type"]
    if t == "deepseek":
        return DeepSeekBackend(
            api_key=config["api_key"],
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
