"""
Generation Module
-----------------
LLM-генерация ответов со structured output через Pydantic.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.generation.prompt_manager import get_prompt, PromptVersion
from src.retrieval.retriever import format_context


# =============================================================================
# Structured Output Schema
# Structured output гарантирует парсируемый ответ с метаданными,
# что критично для downstream обработки (UI, логирование, мониторинг).
# =============================================================================
class RAGResponse(BaseModel):
    """Структурированный ответ RAG-системы."""
    
    answer: str = Field(
        description="Ответ на вопрос пользователя, основанный исключительно на контексте"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Список источников, использованных для ответа"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Уверенность в ответе (0 = нет ответа в контексте, 1 = полный ответ)"
    )
    needs_clarification: bool = Field(
        default=False,
        description="True если вопрос неоднозначен и требует уточнения"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Уточняющий вопрос, если needs_clarification=True"
    )


class RAGGenerator:
    """Генератор ответов с версионированием промптов."""

    def __init__(
        self,
        prompt_version: str = "v3",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,  # 0 для детерминизма в production
    ):
        self.prompt_config: PromptVersion = get_prompt(prompt_version)
        self.prompt_version = prompt_version
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )
        # Structured output: LLM возвращает JSON, распарсенный в RAGResponse
        self.structured_llm = self.llm.with_structured_output(RAGResponse)

    def generate(
        self,
        question: str,
        retrieved_docs: List[Document],
    ) -> RAGResponse:
        """
        Генерирует ответ на основе вопроса и retrieved документов.
        
        Structured output устраняет необходимость парсинга строк —
        получаем типизированный объект с валидацией.
        """
        if not retrieved_docs:
            return RAGResponse(
                answer="Эта информация отсутствует в предоставленной документации.",
                sources=[],
                confidence=0.0,
                needs_clarification=False,
            )

        context = format_context(retrieved_docs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_config.system_prompt),
            ("human", self.prompt_config.user_template),
        ])

        messages = prompt.format_messages(
            context=context,
            question=question,
        )

        response: RAGResponse = self.structured_llm.invoke(messages)
        return response

    def generate_streaming(self, question: str, retrieved_docs: List[Document]):
        """
        Стриминговая генерация для UI.
        Yields строки по мере генерации LLM.
        """
        context = format_context(retrieved_docs)
        
        # Для стриминга используем обычный LLM (без structured output)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_config.system_prompt),
            ("human", self.prompt_config.user_template),
        ])
        messages = prompt.format_messages(context=context, question=question)
        
        for chunk in self.llm.stream(messages):
            yield chunk.content
