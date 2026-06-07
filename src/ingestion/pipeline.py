from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from src.llm_factory import get_embeddings

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents(docs_dir: str | Path) -> List[Document]:
    docs_dir = Path(docs_dir)
    documents: List[Document] = []
    loaders = {".pdf": PyPDFLoader, ".txt": TextLoader}
    for file_path in docs_dir.rglob("*"):
        loader_cls = loaders.get(file_path.suffix.lower())
        if loader_cls:
            try:
                docs = loader_cls(str(file_path)).load()
                for doc in docs:
                    doc.metadata["source_file"] = file_path.name
                documents.extend(docs)
                print(f"  ✓ {file_path.name} ({len(docs)} частей)")
            except Exception as e:
                print(f"  ✗ {file_path.name}: {e}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  → {len(chunks)} чанков из {len(documents)} документов")
    return chunks


def build_vectorstore(chunks: List[Document], persist_dir: str = "./chroma_db") -> Chroma:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
        collection_name="financial_docs",
    )
    print(f"  → Индекс сохранён: {persist_dir} ({len(chunks)} векторов)")
    return vectorstore


def ingest(docs_dir: str, persist_dir: str = "./chroma_db") -> Chroma:
    print(f"\n📂 Загрузка из: {docs_dir}")
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"Документы не найдены в {docs_dir}")
    print(f"\n✂️  Чанкинг (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(documents)
    print(f"\n🔢 Создание эмбеддингов...")
    return build_vectorstore(chunks, persist_dir)
