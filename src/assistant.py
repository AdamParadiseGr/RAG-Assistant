from langchain_chroma import Chroma
from src.llm_factory import get_llm
from src.retrieval.retriever import load_vectorstore, retrieve_with_decomposition
from src.generation.generator import RAGGenerator, RAGResponse


class RAGAssistant:
    def __init__(self, persist_dir: str = "./chroma_db", prompt_version: str = "v3"):
        self.vectorstore: Chroma = load_vectorstore(persist_dir)
        self.generator = RAGGenerator(prompt_version=prompt_version)
        self.llm = get_llm()

    def ask(self, question: str, verbose: bool = False) -> RAGResponse:
        docs, sub_queries = retrieve_with_decomposition(question, self.vectorstore, self.llm)
        if verbose:
            print(f"\n🔍 Под-запросы: {sub_queries}")
            print(f"📄 Найдено чанков: {len(docs)}")
            for i, doc in enumerate(docs, 1):
                print(f"   [{i}] {doc.metadata.get('source_file','?')}: {doc.page_content[:80]}...")
        return self.generator.generate(question, docs)

    def ask_streaming(self, question: str):
        docs, _ = retrieve_with_decomposition(question, self.vectorstore, self.llm)
        yield from self.generator.generate_streaming(question, docs)
