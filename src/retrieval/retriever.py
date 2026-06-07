from typing import List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from src.llm_factory import get_embeddings

TOP_K = 5
FETCH_K = 20
MMR_LAMBDA = 0.7


def load_vectorstore(persist_dir: str = "./chroma_db") -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
        collection_name="financial_docs",
    )


def retrieve(query: str, vectorstore: Chroma, k: int = TOP_K, use_mmr: bool = True) -> List[Document]:
    if use_mmr:
        return vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA
        )
    return vectorstore.similarity_search(query, k=k)


def decompose_query(query: str, llm) -> List[str]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Разбей вопрос на простые части, если их несколько. "
                   "Если вопрос простой — верни без изменений. "
                   "Только список под-вопросов, по одному на строку, без нумерации."),
        ("human", "Вопрос: {query}"),
    ])
    response = llm.invoke(prompt.format_messages(query=query))
    parts = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return parts if parts else [query]


def retrieve_with_decomposition(
    query: str, vectorstore: Chroma, llm, k: int = TOP_K
) -> Tuple[List[Document], List[str]]:
    sub_queries = decompose_query(query, llm)
    all_docs: List[Document] = []
    seen = set()
    for sub_query in sub_queries:
        for doc in retrieve(sub_query, vectorstore, k=k):
            h = hash(doc.page_content[:200])
            if h not in seen:
                seen.add(h)
                all_docs.append(doc)
    return all_docs[:k * 2], sub_queries


def format_context(docs: List[Document]) -> str:
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "неизвестный источник")
        page = doc.metadata.get("page", "")
        label = source + (f", стр. {page + 1}" if page != "" else "")
        parts.append(f"[Источник {i}: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
