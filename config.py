"""Application configuration and model factory helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _chat_model(model: str, temperature: float, api_key: str | None, base_url: str | None):
    kwargs = {
        "model": model,
        "temperature": temperature,
        "request_timeout": _get_int("OPENAI_TIMEOUT", 60),
        "max_retries": _get_int("OPENAI_MAX_RETRIES", 2),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _embedding_model(model: str, api_key: str | None, base_url: str | None):
    kwargs = {
        "model": model,
        "request_timeout": _get_int("OPENAI_TIMEOUT", 60),
        "max_retries": _get_int("OPENAI_MAX_RETRIES", 2),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def load_config() -> dict:
    """Load environment-backed config used by Streamlit tabs and agents."""
    load_dotenv(ROOT_DIR / ".env", override=True)

    api_key = os.getenv("OPENAI_API_KEY") or None
    base_url = os.getenv("OPENAI_BASE_URL") or None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    fast_model = os.getenv("OPENAI_FAST_MODEL", model)
    tool_model = os.getenv("OPENAI_TOOL_MODEL", fast_model)

    embedding_api_key = os.getenv("EMBEDDING_API_KEY") or api_key
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or base_url
    embedding = (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )

    knowledge_base_dir = Path(os.getenv("KNOWLEDGE_BASE_DIR", DATA_DIR / "knowledge_base"))
    memory_store_dir = Path(os.getenv("MEMORY_STORE_DIR", DATA_DIR / "memory_store"))
    upload_dir = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
    interview_bank_dir = Path(os.getenv("INTERVIEW_BANK_DIR", DATA_DIR / "interview_question_bank"))

    for path in [knowledge_base_dir, memory_store_dir, upload_dir, interview_bank_dir]:
        path.mkdir(parents=True, exist_ok=True)

    temperature = _get_float("OPENAI_TEMPERATURE", 0.2)
    fast_temperature = _get_float("OPENAI_FAST_TEMPERATURE", 0.0)

    return {
        "llm": _chat_model(model, temperature, api_key, base_url),
        "fast_llm": _chat_model(fast_model, fast_temperature, api_key, base_url),
        "tool_llm": _chat_model(tool_model, 0.0, api_key, base_url),
        "embedding_model": _embedding_model(embedding, embedding_api_key, embedding_base_url),
        "knowledge_base_dir": str(knowledge_base_dir),
        "memory_store_dir": str(memory_store_dir),
        "upload_dir": str(upload_dir),
        "interview_bank_dir": str(interview_bank_dir),
        "rag_top_k": _get_int("RAG_TOP_K", 3),
        "rag_chunk_size": _get_int("RAG_CHUNK_SIZE", 800),
        "rag_chunk_overlap": _get_int("RAG_CHUNK_OVERLAP", 120),
        "rag_rrf_k": _get_int("RAG_RRF_K", 60),
        "memory_top_k": _get_int("MEMORY_TOP_K", 3),
        "interview_question_count": _get_int("INTERVIEW_QUESTION_COUNT", 12),
        "interview_bank_top_k": _get_int("INTERVIEW_BANK_TOP_K", 6),
        "interview_bank_min_hits": _get_int("INTERVIEW_BANK_MIN_HITS", 4),
        "interview_bank_max_size": _get_int("INTERVIEW_BANK_MAX_SIZE", 1000),
        "interview_bank_rrf_k": _get_int("INTERVIEW_BANK_RRF_K", _get_int("RAG_RRF_K", 60)),
        "interview_web_search_k": _get_int("INTERVIEW_WEB_SEARCH_K", 5),
        "max_revision_rounds": _get_int("MAX_REVISION_ROUNDS", 3),
        "reflection_score_threshold": _get_int("REFLECTION_SCORE_THRESHOLD", 7),
        "max_react_rounds": _get_int("MAX_REACT_ROUNDS", 3),
        "model_name": model,
        "fast_model_name": fast_model,
        "embedding_model_name": embedding,
        "embedding_base_url": embedding_base_url,
    }
