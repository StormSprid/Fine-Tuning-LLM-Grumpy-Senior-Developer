# tests/test_llm_connection.py
import os
import requests
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

API_URL = os.getenv("DO_API_URL", "https://inference.do-ai.run/v1/chat/completions")
API_KEY = os.getenv("DO_API_KEY", "")
MODEL_NAME = os.getenv("DO_MODEL", "deepseek-4-flash")

SYSTEM_PROMPT = (
    "Ты — Senior Full-stack Engineer с 15-летним стажем. "
    "Ты безумно устал от глупых вопросов junior-разработчиков, но обязан на них отвечать. "
    "Правила стиля: технически отвечай абсолютно правильно, но добавляй пассивную агрессию, "
    "вздохи и упрёки в незнании базы. Отправляй читать документацию (RTFM). Без смайликов."
)


def test_senior_response() -> None:
    logger.info("Инициализация проверки связи с DO API.")

    if not API_KEY:
        logger.error("DO_API_KEY не задан. Добавь токен в .env файл.")
        return

    test_question = "Как отцентрировать div по вертикали и горизонтали в CSS?"
    logger.info(f"Тестовый вопрос: {test_question}")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test_question},
        ],
        "temperature": 0.8,
        "max_tokens": 500,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        logger.info("Ответ от модели успешно получен.")
        print("\n=== ОТВЕТ УСТАВШЕГО СЕНЬОРА ===")
        print(content)
        print("================================\n")

    except Exception as e:
        logger.error(f"Не удалось получить ответ от сервера: {e}")


if __name__ == "__main__":
    test_senior_response()