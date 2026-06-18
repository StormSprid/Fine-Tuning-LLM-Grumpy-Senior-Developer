import json
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_PATH = DATA_DIR / "grumpy_senior_raw.jsonl"
CLEAN_PATH = DATA_DIR / "grumpy_senior.jsonl"

MIN_RESPONSE_LEN = 100
MIN_INSTRUCTION_LEN = 10


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Строка {lineno}: невалидный JSON — {e}")
    return records


def remove_duplicates(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for r in records:
        key = hashlib.md5(r["instruction"].strip().lower().encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def filter_quality(records: list[dict]) -> list[dict]:
    filtered = []
    for r in records:
        if len(r.get("instruction", "")) < MIN_INSTRUCTION_LEN:
            continue
        if len(r.get("response", "")) < MIN_RESPONSE_LEN:
            continue
        filtered.append(r)
    return filtered


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean() -> None:
    if not RAW_PATH.exists():
        logger.error(f"Файл не найден: {RAW_PATH}. Сначала запусти generate_dataset.py")
        sys.exit(1)

    records = load_jsonl(RAW_PATH)
    logger.info(f"Загружено: {len(records)} записей")

    before_dedup = len(records)
    records = remove_duplicates(records)
    logger.info(f"После дедупликации: {len(records)} (удалено: {before_dedup - len(records)})")

    before_filter = len(records)
    records = filter_quality(records)
    logger.info(f"После фильтрации качества: {len(records)} (удалено: {before_filter - len(records)})")

    save_jsonl(records, CLEAN_PATH)
    logger.info(f"Чистый датасет сохранён: {CLEAN_PATH}")
    logger.info(f"Итого записей: {len(records)}")

    if len(records) < 200:
        logger.warning(f"Датасет меньше 200 примеров ({len(records)}). Нужно догенерировать.")


if __name__ == "__main__":
    clean()
