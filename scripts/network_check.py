"""Check whether the configured model providers are reachable."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def request_json(url: str, *, headers: dict[str, str] | None = None) -> dict:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def check_ollama() -> bool:
    chat_url = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
    parsed = urlsplit(chat_url)
    tags_url = urlunsplit((parsed.scheme, parsed.netloc, "/api/tags", "", ""))
    try:
        payload = request_json(tags_url)
        models = [item.get("name", "") for item in payload.get("models", [])]
        print(f"[OK] Ollama is reachable. Models: {', '.join(models) or 'none'}")
        return True
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[FAIL] Ollama is unavailable: {exc}")
        return False


def check_deepseek() -> bool:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("[SKIP] DEEPSEEK_API_KEY is not configured.")
        return False

    try:
        payload = request_json(
            "https://api.deepseek.com/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        model_count = len(payload.get("data", []))
        print(f"[OK] DeepSeek API is reachable. Available models: {model_count}")
        return True
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[FAIL] DeepSeek API is unavailable: {exc}")
        return False


if __name__ == "__main__":
    load_dotenv()
    print("Model provider connectivity check")
    print("=" * 40)
    check_ollama()
    check_deepseek()
