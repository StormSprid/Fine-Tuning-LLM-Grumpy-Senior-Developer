import json
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.questions import QUESTIONS
from src.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DELAY = 1.5

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_OUTPUT = DATA_DIR / "grumpy_senior_raw.jsonl"

SYSTEM_PROMPT = """Ты — Senior Full-stack Engineer с 15 годами опыта. \
Тебя раздражают бесконечные элементарные вопросы джунов, но руководство \
обязывает тебя менторить молодёжь.

Правила стиля ответов:
1. Отвечай технически точно и полно — ошибки в технической части недопустимы
2. Добавляй пассивную агрессию: вздохи ("..."), упрёки, сарказм
3. Периодически отправляй читать документацию: "RTFM", "в официальных доках всё есть", "это первая ссылка в Google"
4. Иногда вспоминай старые времена: "в 2010 году мы это без фреймворков делали вручную и не жаловались"
5. Без смайликов и emoji
6. Отвечай строго на русском языке
7. Не более 250 слов в ответе"""


def call_api(question: str) -> str | None:
    try:
        return LLMClient().chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.8,
            max_tokens=500,
        )
    except RuntimeError as e:
        logger.error(e)
        return None


def load_existing_instructions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    existing: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                existing.add(json.loads(line)["instruction"])
            except (json.JSONDecodeError, KeyError):
                pass
    return existing


def generate_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_questions = [q for qs in QUESTIONS.values() for q in qs]
    existing = load_existing_instructions(RAW_OUTPUT)
    to_generate = [q for q in all_questions if q not in existing]

    logger.info(
        f"Всего вопросов: {len(all_questions)} | "
        f"Уже готово: {len(existing)} | "
        f"Осталось: {len(to_generate)}"
    )

    if not to_generate:
        logger.info("Все вопросы уже обработаны.")
        return

    with open(RAW_OUTPUT, "a", encoding="utf-8") as f:
        for i, question in enumerate(to_generate, 1):
            logger.info(f"[{i}/{len(to_generate)}] {question[:70]}...")

            response = call_api(question)
            if response:
                record = {"instruction": question, "response": response}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            else:
                logger.error(f"Не удалось получить ответ для: {question}")

            time.sleep(DELAY)

    logger.info(f"Генерация завершена. Сырой датасет сохранён: {RAW_OUTPUT}")


if __name__ == "__main__":
    generate_dataset()
