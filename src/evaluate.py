import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
SCREENSHOTS_DIR = Path(__file__).parent.parent / "screenshots"

JUDGE_PROMPT = """Оцени два ответа на технический вопрос.

Вопрос: {question}

Ответ A (базовая модель): {answer_a}

Ответ B (дообученная модель): {answer_b}

Оцени по критериям (1-10):
1. Ворчливость/характер (есть ли стиль ворчливого синьора?)
2. Техническая точность
3. Полезность

Ответь ТОЛЬКО в JSON без markdown:
{{"grumpiness": 0, "accuracy": 0, "usefulness": 0, "winner": "A или B", "reason": "кратко"}}"""


def judge(question: str, answer_a: str, answer_b: str) -> dict:
    client = LLMClient()
    raw = client.chat(
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question,
                    answer_a=answer_a,
                    answer_b=answer_b,
                ),
            }
        ],
        temperature=0.0,
        max_tokens=200,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)


def plot_results(results: list[dict]) -> None:
    grumpiness = [r["scores"]["grumpiness"] for r in results]
    accuracy = [r["scores"]["accuracy"] for r in results]
    usefulness = [r["scores"]["usefulness"] for r in results]

    x = range(len(results))
    plt.figure(figsize=(12, 5))
    plt.plot(x, grumpiness, label="Ворчливость", marker="o")
    plt.plot(x, accuracy, label="Точность", marker="o")
    plt.plot(x, usefulness, label="Полезность", marker="o")
    plt.xticks(x, [f"Q{i+1}" for i in x], rotation=45)
    plt.ylabel("Оценка (1-10)")
    plt.title("LLM-as-judge: Fine-tuned vs Base Mistral-7B")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    path = SCREENSHOTS_DIR / "evaluation_chart.png"
    plt.savefig(path, dpi=150)
    logger.info(f"График сохранён: {path}")


def print_summary(results: list[dict]) -> None:
    scores = [r["scores"] for r in results]
    grumpiness = [s["grumpiness"] for s in scores]
    accuracy = [s["accuracy"] for s in scores]
    usefulness = [s["usefulness"] for s in scores]
    winners = [s["winner"] for s in scores]

    print("\n" + "=" * 50)
    print("ИТОГИ LLM-AS-JUDGE")
    print("=" * 50)
    print(f"Средняя ворчливость:  {sum(grumpiness)/len(grumpiness):.1f}/10")
    print(f"Средняя точность:     {sum(accuracy)/len(accuracy):.1f}/10")
    print(f"Средняя полезность:   {sum(usefulness)/len(usefulness):.1f}/10")
    print(f"Побед fine-tuned (B): {winners.count('B')}/{len(winners)}")
    print(f"Побед базовой (A):    {winners.count('A')}/{len(winners)}")
    print("=" * 50)


def main() -> None:
    finetuned_path = DATA_DIR / "finetuned_responses.jsonl"
    base_path = DATA_DIR / "base_responses.jsonl"

    for path in (finetuned_path, base_path):
        if not path.exists():
            logger.error(f"Файл не найден: {path}")
            sys.exit(1)

    with open(finetuned_path, encoding="utf-8") as f:
        finetuned_data = [json.loads(line) for line in f]

    with open(base_path, encoding="utf-8") as f:
        base_data = [json.loads(line) for line in f]

    results = []

    for i, (base, finetuned) in enumerate(zip(base_data, finetuned_data), 1):
        question = base["question"]
        logger.info(f"[{i}/{len(base_data)}] {question[:60]}...")

        verdict = judge(
            question=question,
            answer_a=base["base_response"],
            answer_b=finetuned["finetuned_response"],
        )

        results.append({
            "question": question,
            "base_response": base["base_response"],
            "finetuned_response": finetuned["finetuned_response"],
            "scores": verdict,
        })
        logger.info(f"  Winner: {verdict['winner']}, Grumpiness: {verdict['grumpiness']}/10")

    output_path = DATA_DIR / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Результаты сохранены: {output_path}")

    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main()
