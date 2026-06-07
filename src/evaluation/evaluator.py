"""
LLM-as-Judge Evaluator
----------------------
Оценка качества RAG по методологии RAGAS:
  - Faithfulness      — ответ основан только на контексте?
  - Answer Relevancy  — ответ отвечает на вопрос?
  - Context Precision — retrieved чанки релевантны вопросу?

Каждая метрика — отдельный промпт → Groq → оценка 0.0–1.0.
Результаты сохраняются в JSON.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_llm


# ---------------------------------------------------------------------------
# Промпты для каждой метрики
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — беспристрастный судья качества RAG-систем.

Твоя задача: оценить FAITHFULNESS — насколько ответ основан исключительно на предоставленном контексте.

Правила оценки:
- 1.0 — все факты в ответе подтверждены контекстом, нет ничего лишнего
- 0.5 — часть фактов подтверждена, часть взята из внешних знаний
- 0.0 — ответ содержит факты, которых нет в контексте (галлюцинации)

Ответь ТОЛЬКО числом от 0.0 до 1.0. Никакого текста."""),
    ("human", """Вопрос: {question}

Контекст:
{context}

Ответ системы:
{answer}

Оценка Faithfulness (только число):"""),
])

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — беспристрастный судья качества RAG-систем.

Твоя задача: оценить ANSWER RELEVANCY — насколько ответ отвечает на заданный вопрос.

Правила оценки:
- 1.0 — ответ полностью и точно отвечает на вопрос
- 0.5 — ответ частично отвечает или содержит лишнюю информацию
- 0.0 — ответ не относится к вопросу

Ответь ТОЛЬКО числом от 0.0 до 1.0. Никакого текста."""),
    ("human", """Вопрос: {question}

Ответ системы:
{answer}

Оценка Answer Relevancy (только число):"""),
])

CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Ты — беспристрастный судья качества RAG-систем.

Твоя задача: оценить CONTEXT PRECISION — насколько retrieved фрагменты релевантны вопросу.

Правила оценки:
- 1.0 — все фрагменты напрямую относятся к вопросу
- 0.5 — часть фрагментов релевантна, часть лишняя
- 0.0 — фрагменты не относятся к вопросу

Ответь ТОЛЬКО числом от 0.0 до 1.0. Никакого текста."""),
    ("human", """Вопрос: {question}

Retrieved фрагменты:
{context}

Оценка Context Precision (только число):"""),
])


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.llm = get_llm(temperature=0.0)

    def _score(self, prompt: ChatPromptTemplate, **kwargs) -> float:
        """Вызывает LLM и парсит число из ответа."""
        response = self.llm.invoke(prompt.format_messages(**kwargs))
        text = response.content.strip()
        # Ищем число в ответе (на случай если модель добавила лишний текст)
        match = re.search(r"\b([01]\.?\d*)\b", text)
        if match:
            return min(1.0, max(0.0, float(match.group(1))))
        return 0.0

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        """Оценивает один вопрос-ответ по трём метрикам."""
        context_text = "\n\n---\n\n".join(contexts)

        faithfulness = self._score(
            FAITHFULNESS_PROMPT,
            question=question,
            context=context_text,
            answer=answer,
        )
        relevancy = self._score(
            RELEVANCY_PROMPT,
            question=question,
            answer=answer,
        )
        precision = self._score(
            CONTEXT_PRECISION_PROMPT,
            question=question,
            context=context_text,
        )

        return {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
        }

    def evaluate_pipeline(
        self,
        questions: List[str],
        ground_truths: List[str],
        answers: List[str],
        contexts: List[List[str]],
        prompt_version: str = "v3",
    ) -> Dict[str, Any]:
        """Оценивает весь датасет и сохраняет отчёт."""
        import os
        all_scores = []

        for i, (q, gt, ans, ctx) in enumerate(
            zip(questions, ground_truths, answers, contexts), 1
        ):
            scores = self.evaluate_sample(q, ans, ctx, gt)
            all_scores.append(scores)
            print(
                f"  [{i}/{len(questions)}] "
                f"faith={scores['faithfulness']:.2f} "
                f"relev={scores['answer_relevancy']:.2f} "
                f"prec={scores['context_precision']:.2f} | "
                f"{q[:45]}..."
            )

        # Усредняем по всем вопросам
        metrics = {
            key: round(sum(s[key] for s in all_scores) / len(all_scores), 3)
            for key in ["faithfulness", "answer_relevancy", "context_precision"]
        }

        report = {
            "prompt_version": prompt_version,
            "timestamp": datetime.now().isoformat(),
            "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
            "llm_model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            "embed_provider": os.getenv("EMBED_PROVIDER", "local"),
            "num_samples": len(questions),
            "metrics": metrics,
            "thresholds": {
                "faithfulness": 0.85,
                "answer_relevancy": 0.75,
                "context_precision": 0.75,
            },
            "passed": all([
                metrics["faithfulness"] >= 0.85,
                metrics["answer_relevancy"] >= 0.75,
                metrics["context_precision"] >= 0.75,
            ]),
        }

        path = self.results_dir / f"eval_{prompt_version}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._print_report(report)
        return report

    def _print_report(self, report: Dict[str, Any]) -> None:
        m = report["metrics"]
        status = "✅ PASSED" if report["passed"] else "❌ FAILED"
        print(f"\n{'='*52}")
        print(f"📊 Промпт {report['prompt_version']} | {report['llm_provider']}/{report['llm_model']}")
        print(f"{'='*52}")
        print(f"Faithfulness:      {m['faithfulness']:.3f}  (порог: 0.85)")
        print(f"Answer Relevancy:  {m['answer_relevancy']:.3f}  (порог: 0.75)")
        print(f"Context Precision: {m['context_precision']:.3f}  (порог: 0.75)")
        print(f"{'='*52}")
        print(f"Итог: {status}")
        print(f"Сохранено: results/eval_{report['prompt_version']}.json\n")
