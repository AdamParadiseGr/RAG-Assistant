#!/usr/bin/env python3
"""
CLI — три команды: ingest / chat / evaluate
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="RAG Assistant CLI")
console = Console()


@app.command()
def ingest(
    docs_dir: str = typer.Option("data/sample_docs", help="Путь к документам"),
    db_dir: str = typer.Option("./chroma_db", help="Путь к векторному хранилищу"),
):
    """Загружает документы в векторное хранилище."""
    from src.ingestion.pipeline import ingest as run_ingest
    run_ingest(docs_dir, db_dir)


@app.command()
def chat(
    prompt_version: str = typer.Option("v3", help="Версия промпта: v1, v2, v3"),
    db_dir: str = typer.Option("./chroma_db", help="Путь к векторному хранилищу"),
    verbose: bool = typer.Option(False, help="Показывать retrieved чанки"),
):
    """Запускает интерактивный чат с ассистентом."""
    from src.assistant import RAGAssistant
    from src.generation.prompt_manager import show_prompt_history

    show_prompt_history()
    assistant = RAGAssistant(persist_dir=db_dir, prompt_version=prompt_version)
    
    console.print(Panel(
        f"[bold green]RAG Assistant[/bold green] (промпт: {prompt_version})\n"
        "Введите вопрос или [bold]exit[/bold] для выхода.",
        expand=False
    ))

    while True:
        question = typer.prompt("\n💬 Вопрос")
        if question.lower() in ("exit", "quit", "выход"):
            break

        with console.status("Ищу ответ..."):
            response = assistant.ask(question, verbose=verbose)

        console.print(Panel(
            f"[bold]Ответ:[/bold] {response.answer}\n\n"
            f"[dim]Источники: {', '.join(response.sources) or 'нет'}\n"
            f"Уверенность: {response.confidence:.0%}[/dim]",
            title="🤖 Ассистент",
            border_style="blue",
        ))


@app.command()
def evaluate(
    prompt_version: str = typer.Option("v3", help="Версия промпта для оценки"),
    db_dir: str = typer.Option("./chroma_db", help="Путь к векторному хранилищу"),
    output: str = typer.Option("./results", help="Папка для сохранения отчётов"),
):
    """Запускает RAGAS-оценку качества RAG-системы."""
    from src.assistant import RAGAssistant
    from src.evaluation.evaluator import RAGEvaluator, SAMPLE_TEST_DATASET
    
    assistant = RAGAssistant(persist_dir=db_dir, prompt_version=prompt_version)
    evaluator = RAGEvaluator(results_dir=output)

    console.print(f"\n🧪 Запуск оценки ({len(SAMPLE_TEST_DATASET)} вопросов)...")

    questions, ground_truths, answers, contexts = [], [], [], []
    
    for item in SAMPLE_TEST_DATASET:
        q = item["question"]
        response = assistant.ask(q)
        
        from src.retrieval.retriever import retrieve_with_decomposition
        from langchain_openai import ChatOpenAI
        docs, _ = retrieve_with_decomposition(
            q, assistant.vectorstore, ChatOpenAI(model="gpt-4o-mini")
        )
        
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        answers.append(response.answer)
        contexts.append([doc.page_content for doc in docs])
        
        console.print(f"  ✓ {q[:60]}...")

    evaluator.evaluate_pipeline(
        questions, ground_truths, answers, contexts,
        prompt_version=prompt_version
    )


if __name__ == "__main__":
    app()
