"""
evaluate_all.py
---------------
Запускает LLM-as-judge оценку для всех трёх версий промптов.
Сохраняет JSON-отчёты в results/.

Запуск:
    python scripts/evaluate_all.py

Требования:
    GROQ_API_KEY_1=... в .env
    python scripts/main.py ingest  # сначала загрузить документы
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.llm_factory import get_llm
from src.retrieval.retriever import load_vectorstore, retrieve_with_decomposition
from src.generation.generator import RAGGenerator
from src.generation.prompt_manager import get_prompt
from src.evaluation.evaluator import RAGEvaluator


DATASET_PATH = Path(__file__).parent.parent / "tests_data" / "eval_dataset.json"
CHROMA_DIR = "./chroma_db"
PROMPT_VERSIONS = ["v1", "v2", "v3"]
REQUEST_DELAY = 1.5  # пауза между запросами (rate limit Groq)


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(dataset, prompt_version, vectorstore, llm):
    """Прогоняет все вопросы через RAG и собирает ответы."""
    generator = RAGGenerator(prompt_version=prompt_version)
    questions, ground_truths, answers, contexts = [], [], [], []

    print(f"\n  Генерирую ответы ({len(dataset)} вопросов)...")
    for i, item in enumerate(dataset, 1):
        q = item["question"]
        try:
            docs, _ = retrieve_with_decomposition(q, vectorstore, llm)
            response = generator.generate(q, docs)
            questions.append(q)
            ground_truths.append(item["ground_truth"])
            answers.append(response.answer)
            contexts.append([doc.page_content for doc in docs])
            print(f"  [{i}/{len(dataset)}] ✓ {q[:55]}...")
        except Exception as e:
            print(f"  [{i}/{len(dataset)}] ✗ {e}")
            questions.append(q)
            ground_truths.append(item["ground_truth"])
            answers.append("Ошибка при генерации.")
            contexts.append([""])
        time.sleep(REQUEST_DELAY)

    return questions, ground_truths, answers, contexts


def print_summary(all_results):
    print("\n" + "=" * 58)
    print("ИТОГОВОЕ СРАВНЕНИЕ ВЕРСИЙ ПРОМПТОВ")
    print("=" * 58)
    print(f"{'Версия':<8} {'Faith':<8} {'Relev':<8} {'Prec':<8} {'Статус'}")
    print("-" * 58)
    for version, data in all_results.items():
        m = data["metrics"]
        status = "✅ OK" if data["passed"] else "❌ FAIL"
        print(
            f"{version:<8} "
            f"{m['faithfulness']:<8.3f} "
            f"{m['answer_relevancy']:<8.3f} "
            f"{m['context_precision']:<8.3f} "
            f"{status}"
        )
    print("=" * 58)
    print(f"\nОтчёты: results/eval_v1.json, eval_v2.json, eval_v3.json")


def main():
    print("🔍 Загружаю тестовый датасет...")
    dataset = load_dataset()
    print(f"  → {len(dataset)} вопросов из {DATASET_PATH.name}")

    print("\n🗄️  Загружаю векторное хранилище...")
    vectorstore = load_vectorstore(CHROMA_DIR)
    llm = get_llm()
    evaluator = RAGEvaluator(results_dir="./results")
    all_results = {}

    for version in PROMPT_VERSIONS:
        print(f"\n{'='*52}")
        print(f"📋 Промпт {version}: {get_prompt(version).motivation}")
        print(f"{'='*52}")

        questions, ground_truths, answers, contexts = run_pipeline(
            dataset, version, vectorstore, llm
        )

        print(f"\n📊 Оцениваю качество (LLM-as-judge)...")
        report = evaluator.evaluate_pipeline(
            questions, ground_truths, answers, contexts,
            prompt_version=version,
        )
        all_results[version] = {
            "metrics": report["metrics"],
            "passed": report["passed"],
        }

    print_summary(all_results)


if __name__ == "__main__":
    main()
