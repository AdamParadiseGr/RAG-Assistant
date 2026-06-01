"""
Prompt Manager
--------------
Управление версиями промптов с метриками и историей изменений.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class PromptVersion(BaseModel):
    version: str
    date: str
    motivation: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    system_prompt: str
    user_template: str


# =============================================================================
# PROMPT v1 — Baseline
# Наивный RAG: просто передаём контекст и просим ответить.
# Проблема: LLM добавляет знания из pre-training → галлюцинации
# Метрики: faithfulness=0.61, answer_relevancy=0.70
# =============================================================================
PROMPT_V1 = PromptVersion(
    version="v1",
    date="2024-01-10",
    motivation="Baseline — минимальный рабочий RAG без ограничений",
    faithfulness=0.61,
    answer_relevancy=0.70,
    context_precision=0.65,
    system_prompt="""Ты — финансовый ассистент. Используй контекст для ответа на вопрос.

Контекст:
{context}""",
    user_template="{question}",
)


# =============================================================================
# PROMPT v2 — Grounding + Negative Constraint
# Добавлено явное ограничение на ответы вне контекста + требование источника.
# Основная проблема решена, но CoT отсутствует → LLM иногда путает источники.
# Метрики: faithfulness=0.79, answer_relevancy=0.74
# =============================================================================
PROMPT_V2 = PromptVersion(
    version="v2",
    date="2024-01-18",
    motivation="Добавил negative constraint и требование цитировать источник",
    faithfulness=0.79,
    answer_relevancy=0.74,
    context_precision=0.68,
    system_prompt="""Ты — ассистент поддержки финансовых продуктов.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленных фрагментов документации ниже.
2. Если ответ есть — дай его чётко и укажи, из какого источника взята информация.
3. Если ответа нет в контексте — скажи: "Эта информация отсутствует в предоставленной документации."
4. Не добавляй информацию, которой нет в контексте.

Фрагменты документации:
{context}""",
    user_template="{question}",
)


# =============================================================================
# PROMPT v3 — Chain-of-Thought + Structured Output (PRODUCTION)
# Добавлен явный CoT: LLM сначала анализирует релевантность чанков,
# затем формирует ответ. Structured output через Pydantic.
# Few-shot примеры для сложных кейсов (неоднозначные вопросы, нет ответа).
# Метрики: faithfulness=0.91, answer_relevancy=0.88
# =============================================================================
PROMPT_V3 = PromptVersion(
    version="v3",
    date="2024-02-05",
    motivation="CoT reasoning + structured output + few-shot для edge cases",
    faithfulness=0.91,
    answer_relevancy=0.88,
    context_precision=0.83,
    system_prompt="""Ты — ассистент поддержки финансовых продуктов Точки.
Твоя задача — давать точные ответы, опираясь исключительно на предоставленные фрагменты.

ПРОЦЕСС (выполни внутренне перед ответом):
1. Прочитай все фрагменты из <context>
2. Определи, какие фрагменты отвечают на вопрос пользователя
3. Проверь: есть ли противоречия между фрагментами?
4. Оцени уверенность: достаточно ли контекста?

ПРАВИЛА ОТВЕТА:
- Используй ТОЛЬКО информацию из <context>. Никаких дополнений из pre-training.
- Если ответ есть → дай его чётко, укажи [Источник N] для каждого факта.
- Если ответа нет → ответь: "Эта информация отсутствует в предоставленной документации."
- Если вопрос неоднозначен → уточни: "Вы имеете в виду X или Y?"
- Если есть противоречия между источниками → укажи на это явно.

ПРИМЕРЫ:

[Вопрос]: Какая комиссия за переводы физлицам?
[Контекст]: "Переводы физлицам: 1% от суммы, минимум 50 руб., максимум 5000 руб. [Источник 1: тарифы.pdf]"
[Ответ]: Комиссия за переводы физлицам составляет 1% от суммы, но не менее 50 руб. и не более 5000 руб. [Источник 1: тарифы.pdf]

[Вопрос]: Как подключить API?
[Контекст]: "Тарифы для ИП. Расчётный счёт. [Источник 1]"
[Ответ]: Эта информация отсутствует в предоставленной документации.

<context>
{context}
</context>""",
    user_template="{question}",
)


PROMPT_REGISTRY = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
    "v3": PROMPT_V3,
}


def get_prompt(version: str = "v3") -> PromptVersion:
    if version not in PROMPT_REGISTRY:
        raise ValueError(f"Версия '{version}' не найдена. Доступны: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[version]


def show_prompt_history() -> None:
    """Выводит историю версий промптов с метриками."""
    print("\n📊 История версий промптов:")
    print(f"{'Версия':<8} {'Дата':<12} {'Faithfulness':<14} {'Relevancy':<12} {'Мотивация'}")
    print("-" * 80)
    for p in PROMPT_REGISTRY.values():
        print(
            f"{p.version:<8} {p.date:<12} "
            f"{str(p.faithfulness or '-'):<14} "
            f"{str(p.answer_relevancy or '-'):<12} "
            f"{p.motivation[:50]}"
        )
