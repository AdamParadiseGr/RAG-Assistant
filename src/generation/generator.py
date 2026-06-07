"""
Generation Module
-----------------
LLM-генерация ответов с JSON output через промпт.
Не использует function calling (нестабилен на Groq/LLaMA).
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.llm_factory import get_llm
from src.generation.prompt_manager import get_prompt, PromptVersion
from src.retrieval.retriever import format_context


class RAGResponse(BaseModel):
    answer: str = Field(description="Ответ, основанный исключительно на контексте")
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = Field(default=False)
    clarification_question: Optional[str] = Field(default=None)


JSON_INSTRUCTION = """
Ответь СТРОГО в формате JSON без markdown-блоков и лишнего текста:
{{"answer": "...", "sources": ["..."], "confidence": 0.0, "needs_clarification": false, "clarification_question": null}}
"""


def _parse_response(text: str) -> RAGResponse:
    """Парсит JSON из ответа LLM с fallback."""
    text = text.strip()
    # Убираем markdown-блоки если модель их добавила
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(text)
        # Приводим confidence к float на случай если модель вернула int
        data["confidence"] = float(data.get("confidence", 0.5))
        return RAGResponse(**data)
    except Exception:
        # Fallback: возвращаем ответ как plain text
        return RAGResponse(answer=text, sources=[], confidence=0.5)


class RAGGenerator:
    def __init__(self, prompt_version: str = "v3", temperature: float = 0.0):
        self.prompt_config: PromptVersion = get_prompt(prompt_version)
        self.llm = get_llm(temperature=temperature)

    def generate(self, question: str, retrieved_docs: List[Document]) -> RAGResponse:
        if not retrieved_docs:
            return RAGResponse(
                answer="Эта информация отсутствует в предоставленной документации.",
                sources=[], confidence=0.0,
            )
        context = format_context(retrieved_docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_config.system_prompt + JSON_INSTRUCTION),
            ("human", self.prompt_config.user_template),
        ])
        response = self.llm.invoke(
            prompt.format_messages(context=context, question=question)
        )
        return _parse_response(response.content)

    def generate_streaming(self, question: str, retrieved_docs: List[Document]):
        context = format_context(retrieved_docs)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_config.system_prompt),
            ("human", self.prompt_config.user_template),
        ])
        for chunk in self.llm.stream(
            prompt.format_messages(context=context, question=question)
        ):
            yield chunk.content
