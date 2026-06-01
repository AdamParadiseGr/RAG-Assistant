"""
Document Ingestion Pipeline
---------------------------
Загрузка, парсинг, очистка, чанкинг и индексация документов.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# --- Chunking strategy ---
# chunk_size=512 — достаточно контекста для смысла, не слишком много для точности
# chunk_overlap=64 — предотвращает потерю информации на границах чанков
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents(docs_dir: str | Path) -> List[Document]:
    """Загружает PDF и TXT из директории с метаданными."""
    docs_dir = Path(docs_dir)
    documents: List[Document] = []

    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }

    for file_path in docs_dir.rglob("*"):
        loader_cls = loaders.get(file_path.suffix.lower())
        if loader_cls:
            try:
                loader = loader_cls(str(file_path))
                docs = loader.load()
                # Добавляем метаданные источника для цитирования в ответах
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["source_path"] = str(file_path)
                documents.extend(docs)
                print(f"  ✓ Загружен: {file_path.name} ({len(docs)} страниц/секций)")
            except Exception as e:
                print(f"  ✗ Ошибка загрузки {file_path.name}: {e}")

    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Разбивает документы на чанки.
    
    Используем RecursiveCharacterTextSplitter вместо простого CharacterTextSplitter:
    он пытается сохранить семантические границы (абзацы → предложения → слова),
    что улучшает качество retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  → Создано {len(chunks)} чанков из {len(documents)} документов")
    return chunks


def build_vectorstore(
    chunks: List[Document],
    persist_dir: str = "./chroma_db",
    collection_name: str = "financial_docs",
) -> Chroma:
    """
    Создаёт или обновляет векторное хранилище.
    
    ChromaDB выбран за:
    - персистентное локальное хранение (не нужен внешний сервис)
    - простой API для фильтрации по метаданным
    - поддержку MMR при retrieval
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
    print(f"  → Индекс сохранён в {persist_dir} ({len(chunks)} векторов)")
    return vectorstore


def ingest(docs_dir: str, persist_dir: str = "./chroma_db") -> Chroma:
    """Полный пайплайн загрузки: load → chunk → embed → index."""
    print(f"\n📂 Загрузка документов из: {docs_dir}")
    documents = load_documents(docs_dir)
    
    if not documents:
        raise ValueError(f"Документы не найдены в {docs_dir}")

    print(f"\n✂️  Разбивка на чанки (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(documents)

    print(f"\n🔢 Создание эмбеддингов и индексация...")
    vectorstore = build_vectorstore(chunks, persist_dir)

    print(f"\n✅ Готово! Загружено {len(documents)} документов → {len(chunks)} чанков")
    return vectorstore
