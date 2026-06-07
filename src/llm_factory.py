"""
LLM Factory
-----------
Единая точка конфигурации LLM и Embeddings.
Провайдер выбирается через .env — код менять не нужно.

Поддерживаемые провайдеры LLM:
  groq     → llama-3.1-8b-instant (бесплатно, с ротацией ключей)
  openai   → gpt-4o-mini
  anthropic → claude-haiku-3-5

Поддерживаемые провайдеры Embeddings:
  local    → sentence-transformers (бесплатно, локально)
  openai   → text-embedding-3-small

Пример .env:
  LLM_PROVIDER=groq
  EMBED_PROVIDER=local
"""

import os
import itertools
import time


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.0):
    """
    Возвращает LLM согласно LLM_PROVIDER из .env.
    По умолчанию: groq.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        return _get_groq_llm(temperature)
    elif provider == "openai":
        return _get_openai_llm(temperature)
    elif provider == "anthropic":
        return _get_anthropic_llm(temperature)
    else:
        raise ValueError(
            f"Неизвестный LLM_PROVIDER='{provider}'. "
            f"Доступны: groq, openai, anthropic"
        )


def _get_groq_llm(temperature: float = 0.0):
    """Groq с ротацией ключей."""
    from langchain_groq import ChatGroq
    key = _get_groq_key()
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(model=model, temperature=temperature, api_key=key)


def _get_openai_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


def _get_anthropic_llm(temperature: float = 0.0):
    from langchain_anthropic import ChatAnthropic
    model = os.getenv("LLM_MODEL", "claude-haiku-3-5")
    return ChatAnthropic(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings():
    """
    Возвращает Embeddings согласно EMBED_PROVIDER из .env.
    По умолчанию: local (sentence-transformers, бесплатно).
    """
    provider = os.getenv("EMBED_PROVIDER", "local").lower()

    if provider == "local":
        return _get_local_embeddings()
    elif provider == "openai":
        return _get_openai_embeddings()
    else:
        raise ValueError(
            f"Неизвестный EMBED_PROVIDER='{provider}'. "
            f"Доступны: local, openai"
        )


def _get_local_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    model = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_openai_embeddings():
    from langchain_openai import OpenAIEmbeddings
    model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


# ---------------------------------------------------------------------------
# Groq key rotation
# ---------------------------------------------------------------------------

_groq_key_cycle = None


def _get_groq_key() -> str:
    """Возвращает следующий Groq ключ из ротации."""
    global _groq_key_cycle
    if _groq_key_cycle is None:
        keys = []
        for i in range(1, 11):
            k = os.getenv(f"GROQ_API_KEY_{i}")
            if k:
                keys.append(k.strip())
        if not keys:
            single = os.getenv("GROQ_API_KEY")
            if single:
                keys.append(single.strip())
        if not keys:
            raise ValueError(
                "Groq ключи не найдены. Добавь в .env:\n"
                "  GROQ_API_KEY_1=gsk_...\n"
                "  GROQ_API_KEY_2=gsk_...\n"
                "Или: GROQ_API_KEY=gsk_..."
            )
        _groq_key_cycle = itertools.cycle(keys)
        print(f"  ✓ Groq: {len(keys)} ключ(ей) в ротации")
    return next(_groq_key_cycle)


def rotate_groq_key():
    """Принудительно переключает на следующий Groq ключ (при rate limit)."""
    global _groq_key_cycle
    if _groq_key_cycle:
        return next(_groq_key_cycle)


def invoke_with_rotation(llm, messages, retries: int = 3):
    """
    Вызывает LLM с автоматической ротацией при Groq rate limit.
    Для других провайдеров работает как обычный invoke.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    last_error = None

    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if provider == "groq" and ("rate" in str(e).lower() or "429" in str(e)):
                rotate_groq_key()
                llm = get_llm()  # пересоздаём с новым ключом
                print(f"  ⚠ Rate limit Groq, ротирую ключ (попытка {attempt+1}/{retries})")
                time.sleep(2)
                last_error = e
            else:
                raise
    raise last_error
