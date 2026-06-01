"""
Retrieval Module
----------------
Поиск релевантных чанков с MMR и декомпозицией сложных запросов.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --- Retrieval parameters ---
# fetch_k > k: сначала достаём больше кандидатов, затем MMR отбирает diverse top-k
TOP_K = 5
FETCH_K = 20
MMR_LAMBDA = 0.7  # 0 = max diversity, 1 = max relevance


def load_vectorstore(persist_dir: str = "./chroma_db") -> Chroma:
    """Загружает существующий индекс."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="financial_docs",
    )


def retrieve(
    query: str,
    vectorstore: Chroma,
    k: int = TOP_K,
    use_mmr: bool = True,
) -> List[Document]:
    """
    Поиск релевантных документов.
    
    MMR (Maximum Marginal Relevance) — вместо просто top-K по cosine similarity
    алгоритм балансирует релевантность и разнообразие, устраняя дублирование.
    
    Формула: MMR = λ * relevance(d, q) - (1-λ) * max_similarity(d, selected)
    """
    if use_mmr:
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=FETCH_K,
            lambda_mult=MMR_LAMBDA,
        )
    else:
        docs = vectorstore.similarity_search(query, k=k)
    
    return docs


def decompose_query(query: str, llm: ChatOpenAI) -> List[str]:
    """
    Query Decomposition — разбивает сложный вопрос на простые под-вопросы.
    
    Проблема: "Какие документы нужны для открытия счёта ИП и сколько это стоит?"
    → retrieval находит только часть ответа.
    
    Решение: разбиваем на ["Документы для открытия счёта ИП",
                            "Стоимость открытия счёта ИП"]
    → делаем retrieval для каждого, объединяем уникальные чанки.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты — помощник, который разбивает сложные вопросы на простые.
Если вопрос содержит несколько независимых частей — раздели их.
Если вопрос простой — верни его без изменений.

Верни ТОЛЬКО список под-вопросов, по одному на строку, без нумерации."""),
        ("human", "Вопрос: {query}"),
    ])
    
    response = llm.invoke(prompt.format_messages(query=query))
    sub_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return sub_queries if sub_queries else [query]


def retrieve_with_decomposition(
    query: str,
    vectorstore: Chroma,
    llm: ChatOpenAI,
    k: int = TOP_K,
) -> Tuple[List[Document], List[str]]:
    """
    Полный retrieval пайплайн с декомпозицией запроса.
    Returns: (unique_docs, sub_queries_used)
    """
    sub_queries = decompose_query(query, llm)
    
    all_docs: List[Document] = []
    seen_contents = set()
    
    for sub_query in sub_queries:
        docs = retrieve(sub_query, vectorstore, k=k)
        for doc in docs:
            # Дедупликация по содержимому
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)
    
    # Ограничиваем итоговое количество чанков для генерации
    return all_docs[:k * 2], sub_queries


def format_context(docs: List[Document]) -> str:
    """Форматирует чанки в контекст для промпта с метаданными источников."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "неизвестный источник")
        page = doc.metadata.get("page", "")
        source_label = f"{source}" + (f", стр. {page + 1}" if page != "" else "")
        parts.append(f"[Источник {i}: {source_label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
