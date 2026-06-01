"""
RAG Assistant — главный оркестратор
------------------------------------
Собирает retrieval + generation в единый пайплайн.
"""

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

from src.retrieval.retriever import load_vectorstore, retrieve_with_decomposition
from src.generation.generator import RAGGenerator, RAGResponse


class RAGAssistant:
    """
    Финансовый RAG-ассистент.
    
    Пайплайн:
    1. Query decomposition (если сложный вопрос)
    2. MMR Retrieval для каждого под-вопроса
    3. Дедупликация + объединение чанков
    4. Generation с CoT и structured output
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        prompt_version: str = "v3",
        model: str = "gpt-4o-mini",
    ):
        self.vectorstore: Chroma = load_vectorstore(persist_dir)
        self.generator = RAGGenerator(prompt_version=prompt_version, model=model)
        self.llm = ChatOpenAI(model=model, temperature=0)

    def ask(self, question: str, verbose: bool = False) -> RAGResponse:
        """
        Задаёт вопрос ассистенту.
        
        Args:
            question: Вопрос пользователя
            verbose: Показывать промежуточные шаги (debug)
        
        Returns:
            RAGResponse с answer, sources, confidence
        """
        # Шаг 1: Retrieval с декомпозицией
        docs, sub_queries = retrieve_with_decomposition(
            question, self.vectorstore, self.llm
        )

        if verbose:
            print(f"\n🔍 Под-запросы: {sub_queries}")
            print(f"📄 Найдено чанков: {len(docs)}")
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source_file", "?")
                print(f"   [{i}] {src}: {doc.page_content[:80]}...")

        # Шаг 2: Generation
        response = self.generator.generate(question, docs)
        return response

    def ask_streaming(self, question: str):
        """Стриминговый режим для UI."""
        docs, _ = retrieve_with_decomposition(question, self.vectorstore, self.llm)
        yield from self.generator.generate_streaming(question, docs)
