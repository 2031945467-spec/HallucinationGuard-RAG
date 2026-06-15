import json
import os
import urllib.request
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class LLMResponse:
    content: str


class LLMClient:
    def __init__(self):
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2:0.5b")
        self.ollama_url = os.getenv(
            "OLLAMA_CHAT_URL", "http://localhost:11434/api/chat"
        )

    def invoke(self, prompt: str) -> LLMResponse:
        if (
            self.deepseek_key
            and "your" not in self.deepseek_key.lower()
            and "在这里填入" not in self.deepseek_key
        ):
            return LLMResponse(self._call_deepseek(prompt))
        return LLMResponse(self._call_ollama(prompt))

    def _post(self, url: str, payload: dict, headers: dict | None = None) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _call_ollama(self, prompt: str) -> str:
        data = self._post(
            self.ollama_url,
            {
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": int(os.getenv("LLM_MAX_TOKENS", "256")),
                },
            },
        )
        return data["message"]["content"].strip()

    def _call_deepseek(self, prompt: str) -> str:
        data = self._post(
            "https://api.deepseek.com/v1/chat/completions",
            {
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "512")),
            },
            {"Authorization": f"Bearer {self.deepseek_key}"},
        )
        return data["choices"][0]["message"]["content"].strip()


def get_llm() -> LLMClient:
    return LLMClient()
