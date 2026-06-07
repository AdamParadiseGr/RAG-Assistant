# 🏦 RAG Assistant — финансовый ассистент на основе базы знаний

## Задача

Бизнес-задача: дать клиентам и операторам банка инструмент, который отвечает на вопросы по финансовым продуктам, тарифам и регуляторным документам — точно, с источником, без галлюцинаций.

Технический вызов: обычный LLM не знает внутренних документов, а файн-тюнинг дорог и не масштабируется при частом обновлении базы знаний. RAG решает оба ограничения.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                             │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Documents│───▶│  Ingestion   │───▶│   Vector Store     │    │
│  │ PDF/TXT  │    │  - Chunking  │    │   (ChromaDB)       │    │
│  └──────────┘    │  - Cleaning  │    │   - Embeddings     │    │
│                  │  - Metadata  │    │   - HNSW Index     │    │
│                  └──────────────┘    └────────────┬───────┘    │
│                                                   │            │
│  ┌──────────┐    ┌──────────────┐    ┌────────────▼───────┐    │
│  │  Answer  │◀───│  Generation  │◀───│    Retrieval       │    │
│  │ +Sources │    │  - Prompting │    │    - Dense search  │    │
│  └──────────┘    │  - CoT       │    │    - MMR Reranking │    │
│                  │  - Grounding │    │    - Query Decomp. │    │
│                  └──────────────┘    └────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Evaluation Layer (LLM-as-judge: faithfulness,          │   │
│  │  answer relevancy, context precision)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Компоненты

| Слой | Технология | Обоснование выбора |
|------|------------|-------------------|
| Chunking | RecursiveCharacterTextSplitter (512/64) | Сохраняет семантические границы лучше фиксированного размера |
| Embeddings | sentence-transformers (локально) | Бесплатно, поддерживает русский язык, без внешних API |
| Vector Store | ChromaDB | Простой деплой, персистентное хранилище |
| Retrieval | Top-K + MMR Reranking | MMR устраняет дублирование релевантных чанков |
| Query Expansion | Query Decomposition | Составные вопросы разбиваются на под-запросы |
| LLM | Groq / llama-3.1-8b-instant | Бесплатный tier, ротация ключей при rate limit |
| Evaluation | LLM-as-judge (собственный) | Прозрачная реализация метрик без внешних зависимостей |

---

## Стек

```
Python 3.11+
├── langchain / langchain-chroma       # оркестрация RAG-пайплайна
├── langchain-groq                     # Groq LLM
├── langchain-huggingface              # sentence-transformers embeddings
├── chromadb                           # векторное хранилище
├── pydantic v2                        # валидация structured output
├── typer                              # CLI-интерфейс
└── pytest                             # тесты (14 штук, все проходят)
```

Поддерживаемые LLM провайдеры (переключаются через `.env`): Groq, OpenAI, Anthropic.

---

## Как работал Prompt Engineering

Промпты версионируются в `prompts/` — каждый файл содержит дату, мотивацию изменения и выявленные проблемы.

### v1 — Baseline

Наивный промпт: передаём контекст и просим ответить.

**Проблемы:** LLM добавляла факты из pre-training, не указывала источники, непредсказуемый формат.

### v2 — Grounding + Negative Constraint

Добавлено: явный запрет отвечать вне контекста, требование цитировать источник.

**Остаточные проблемы:** без CoT модель иногда путала источники при нескольких чанках.

### v3 — Chain-of-Thought + JSON Output (production)

1. **CoT reasoning** — модель явно анализирует чанки перед ответом: какой релевантен, есть ли противоречия, достаточно ли контекста
2. **JSON output** — структурированный ответ: `answer`, `sources`, `confidence`, `needs_clarification`
3. **Few-shot examples** — примеры для edge-cases: вопрос вне базы, неоднозначный вопрос, противоречие источников

---

## Результаты оценки (LLM-as-judge)

Оценка по методологии RAGAS реализована через собственный LLM-as-judge на базе Groq.  
Тестовый датасет: 18 вопросов по `data/sample_docs/tariffs_2024.txt`.  
Полные отчёты: [`results/eval_v1.json`](results/eval_v1.json), [`results/eval_v2.json`](results/eval_v2.json), [`results/eval_v3.json`](results/eval_v3.json)

| Метрика | v1 (baseline) | v2 (+grounding) | v3 (+CoT +JSON) |
|---------|:---:|:---:|:---:|
| Faithfulness | 0.972 | 0.972 | 0.972 |
| Answer Relevancy | 0.861 | 0.878 | 0.878 |
| Context Precision | 0.611 | 0.611 | 0.611 |

**Наблюдение:** на простых вопросах с явным ответом в контексте все три версии показывают высокую faithfulness. Разница между промптами проявляется на edge-cases: вопросы вне базы знаний, неоднозначные формулировки, противоречия между источниками — именно для них добавлялись negative constraint и few-shot примеры в v2/v3.

Context Precision (0.611) отражает retrieval, а не качество промпта — на базе из одного документа retriever иногда возвращает менее релевантные чанки.

---

## Какие проблемы решал

**Проблема 1: Галлюцинации**  
Симптом: LLM выдавала правдоподобный, но выдуманный ответ.  
Решение: negative constraint в промпте + faithfulness как quality gate в evaluation loop.

**Проблема 2: Неполные ответы на составные вопросы**  
Симптом: "Какие документы нужны для ИП и сколько стоит счёт?" → retrieval находил только часть.  
Решение: query decomposition — вопрос разбивается на под-запросы, retrieval выполняется для каждого (`src/retrieval/retriever.py` → `decompose_query()`).

**Проблема 3: Дублирование чанков**  
Симптом: top-5 retrieved chunks содержали почти идентичные фрагменты.  
Решение: MMR (Maximum Marginal Relevance) вместо простого cosine similarity top-K.

---

## Быстрый старт

```bash
git clone https://github.com/AdamParadiseGr/RAG-Assistant
cd RAG-Assistant
pip install -r requirements.txt
cp .env.example .env  # вставь GROQ_API_KEY_1=...

# Загрузить документы
PYTHONUTF8=1 python scripts/main.py ingest --docs-dir data/sample_docs/

# Запустить чат
PYTHONUTF8=1 python scripts/main.py chat

# Запустить evaluation
PYTHONUTF8=1 python scripts/evaluate_all.py
```

---

## Структура репозитория

```
RAG-Assistant/
├── src/
│   ├── llm_factory.py      # единая точка конфигурации LLM и Embeddings
│   ├── ingestion/          # загрузка, chunking, embedding
│   ├── retrieval/          # MMR поиск, query decomposition
│   ├── generation/         # промпт-менеджер, JSON output
│   └── evaluation/         # LLM-as-judge evaluator
├── prompts/
│   ├── v1/system.txt       # baseline
│   ├── v2/system.txt       # + grounding constraints
│   └── v3/system.txt       # + CoT + JSON output (production)
├── tests_data/
│   └── eval_dataset.json   # 18 вопросов с эталонными ответами
├── results/                # JSON-отчёты оценки по каждой версии промпта
├── tests/                  # unit-тесты (14 штук)
└── scripts/
    ├── main.py             # CLI: ingest / chat
    └── evaluate_all.py     # запуск полной оценки
```
