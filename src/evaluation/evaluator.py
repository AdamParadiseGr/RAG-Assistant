"""
Evaluation Module
-----------------
Автоматическая оценка качества RAG-системы через RAGAS.

Ключевые метрики:
- Faithfulness: насколько ответ основан ТОЛЬКО на контексте (детектор галлюцинаций)
- Answer Relevancy: насколько ответ отвечает на вопрос
- Context Precision: насколько retrieved контекст релевантен вопросу
- Context Recall: насколько полно retrieved контекст покрывает ответ

Цикл улучшения:
  evaluate() → смотрим слабую метрику → правим промпт/retrieval → evaluate() снова
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Тестовый датасет — вопросы + эталонные ответы (ground truth)
# В production эти данные собираются из обращений в поддержку или размечаются вручную
SAMPLE_TEST_DATASET = [
    {
        "question": "Какая минимальная сумма для открытия вклада?",
        "ground_truth": "Минимальная сумма для открытия вклада составляет 10 000 рублей.",
    },
    {
        "question": "Сколько стоит обслуживание расчётного счёта для ИП?",
        "ground_truth": "Стоимость обслуживания расчётного счёта для ИП зависит от выбранного тарифа.",
    },
    {
        "question": "Как подключить эквайринг?",
        "ground_truth": "Для подключения эквайринга необходимо подать заявку через личный кабинет.",
    },
]


class RAGEvaluator:
    """Оценщик качества RAG-системы."""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def build_eval_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
    ) -> Dataset:
        """Собирает датасет для RAGAS."""
        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

    def evaluate_pipeline(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
        prompt_version: str = "v3",
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Запускает полную оценку и сохраняет результаты.
        
        Результаты сохраняются в JSON для версионирования экспериментов:
        - можно сравнить v1 vs v2 vs v3
        - видна динамика улучшений
        - основа для CI-проверок (не деплоим если faithfulness < 0.85)
        """
        dataset = self.build_eval_dataset(questions, ground_truths, answers, contexts)

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )

        metrics = {
            "faithfulness": round(float(result["faithfulness"]), 3),
            "answer_relevancy": round(float(result["answer_relevancy"]), 3),
            "context_precision": round(float(result["context_precision"]), 3),
            "context_recall": round(float(result["context_recall"]), 3),
        }

        # Сохраняем полный отчёт для версионирования экспериментов
        report = {
            "experiment": experiment_name or f"eval_{prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "prompt_version": prompt_version,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(questions),
            "metrics": metrics,
            "passed": self._check_thresholds(metrics),
        }

        report_path = self.results_dir / f"{report['experiment']}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._print_report(report)
        return report

    def _check_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Проверяет пороговые значения метрик.
        
        Thresholds настроены на основе бизнес-требований:
        - faithfulness > 0.85 критично (галлюцинации в финансах недопустимы)
        - остальные > 0.75 (хорошее качество)
        """
        thresholds = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.75,
            "context_precision": 0.75,
            "context_recall": 0.70,
        }
        return all(metrics.get(k, 0) >= v for k, v in thresholds.items())

    def _print_report(self, report: Dict[str, Any]) -> None:
        metrics = report["metrics"]
        passed = "✅ PASSED" if report["passed"] else "❌ FAILED"
        
        print(f"\n{'='*50}")
        print(f"📊 Evaluation Report — {report['prompt_version']}")
        print(f"{'='*50}")
        print(f"Faithfulness:      {metrics['faithfulness']:.3f}  (threshold: 0.85)")
        print(f"Answer Relevancy:  {metrics['answer_relevancy']:.3f}  (threshold: 0.75)")
        print(f"Context Precision: {metrics['context_precision']:.3f}  (threshold: 0.75)")
        print(f"Context Recall:    {metrics['context_recall']:.3f}  (threshold: 0.70)")
        print(f"{'='*50}")
        print(f"Overall: {passed}")
        print(f"Saved to: results/{report['experiment']}.json\n")

    def compare_versions(self, report_paths: List[str]) -> None:
        """Сравнивает результаты нескольких экспериментов."""
        reports = []
        for path in report_paths:
            with open(path, encoding="utf-8") as f:
                reports.append(json.load(f))

        print(f"\n{'Версия':<10} {'Faith':<8} {'Relev':<8} {'Prec':<8} {'Recall':<8} {'Статус'}")
        print("-" * 60)
        for r in sorted(reports, key=lambda x: x["prompt_version"]):
            m = r["metrics"]
            status = "✅" if r["passed"] else "❌"
            print(
                f"{r['prompt_version']:<10} "
                f"{m['faithfulness']:<8.3f} "
                f"{m['answer_relevancy']:<8.3f} "
                f"{m['context_precision']:<8.3f} "
                f"{m['context_recall']:<8.3f} "
                f"{status}"
            )
