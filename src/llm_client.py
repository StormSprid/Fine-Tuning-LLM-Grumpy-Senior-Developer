import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    _instance: "LLMClient | None" = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.api_url = os.getenv("DO_API_URL", "https://inference.do-ai.run/v1/chat/completions")
        self.api_key = os.getenv("DO_API_KEY", "")
        self.model = os.getenv("DO_MODEL", "deepseek-4-flash")
        self._initialized = True

        if not self.api_key:
            raise ValueError("DO_API_KEY не задан. Добавь токен в .env файл.")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 500,
        retries: int = 3,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=30
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Не удалось получить ответ после {retries} попыток")
