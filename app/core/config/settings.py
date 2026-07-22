from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from app.core.security.exceptions import ConfigurationError


load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_HANDLER", "false")
os.environ.setdefault("LANGCHAIN_TELEMETRY", "false")


def _read_env(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class AppSettings:
    # --- Required fields (no defaults) ---
    pinecone_api_key: str
    cohere_api_key: str
    flask_secret_key: str
    # --- Optional fields ---
    persistence_enabled: bool = True
    supabase_url: str | None = None
    supabase_key: str | None = None
    index_name: str = "medichat"
    embedding_model: str = "embed-english-v3.0"
    llm_model: str = "command-a-03-2025"
    retriever_k: int = 8
    embedding_dimension: int = 1024
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    document_chunk_size: int = 500
    document_chunk_overlap: int = 100
    max_indexed_chunks: int = 20000
    session_history_ttl_seconds: int = 60 * 60
    redis_url: str | None = None
    require_redis: bool = False
    relevance_score_threshold: float = 0.55
    max_content_length_bytes: int = 10 * 1024 * 1024 + 4096
    max_chat_message_length: int = 4000
    login_rate_limit: int = 5
    login_rate_window_seconds: int = 60
    chat_rate_limit: int = 20
    chat_rate_window_seconds: int = 60
    csrf_cookie_name: str = "csrf_token"
    csrf_header_name: str = "X-CSRF-Token"
    session_cookie_name: str = "session_id"
    guest_session_prefix: str = "guest_"
    session_cookie_max_age_seconds: int = 60 * 60 * 24 * 30
    secure_cookies: bool = True
    frontend_origins: tuple[str, ...] = ()
    indexed_answer_suffix: str = "[Based on indexed medical data]"
    general_answer_suffix: str = "[Based on general medical knowledge]"
    # --- Memory system ---
    context_window_size: int = 10
    summary_trigger_count: int = 10
    topic_extraction_enabled: bool = True
    # --- Security ---
    memory_api_rate_limit: int = 60
    memory_api_rate_window_seconds: int = 60
    csp_enabled: bool = True
    guest_session_max_age_seconds: int = 60 * 60 * 24  # 1 day (vs 30 days for auth)
    # --- Performance & Uploads ---
    query_cache_ttl_seconds: int = 300
    query_cache_enabled: bool = True
    llm_timeout_seconds: float = 30.0
    max_upload_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    document_upload_rate_limit: int = 5
    document_upload_rate_window_seconds: int = 60 * 60
    content_generation_rate_limit: int = 20
    content_generation_rate_window_seconds: int = 60



@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    pinecone_api_key = _read_env("PINECONE_API_KEY")
    cohere_api_key = _read_env("COHERE_API_KEY")
    persistence_enabled = os.getenv("PERSISTENCE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    supabase_url = _read_env("SUPABASE_URL", required=persistence_enabled) if persistence_enabled else os.getenv("SUPABASE_URL")
    supabase_key = _read_env("SUPABASE_KEY", required=persistence_enabled) if persistence_enabled else os.getenv("SUPABASE_KEY")

    frontend_origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "FRONTEND_ORIGINS",
            "http://localhost:4200,http://127.0.0.1:4200",
        ).split(",")
        if origin.strip()
    )

    return AppSettings(
        persistence_enabled=persistence_enabled,
        pinecone_api_key=pinecone_api_key,
        cohere_api_key=cohere_api_key,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        flask_secret_key=_read_env("FLASK_SECRET_KEY", required=True),
        session_history_ttl_seconds=int(os.getenv("SESSION_HISTORY_TTL_SECONDS", "3600")),
        redis_url=os.getenv("REDIS_URL"),
        require_redis=os.getenv("REQUIRE_REDIS", "false").lower() in {"1", "true", "yes"},
        relevance_score_threshold=float(os.getenv("RELEVANCE_SCORE_THRESHOLD", "0.55")),
        max_indexed_chunks=int(os.getenv("MAX_INDEXED_CHUNKS", "20000")),
        max_content_length_bytes=int(os.getenv("MAX_CONTENT_LENGTH_BYTES", str(10 * 1024 * 1024 + 4096))),
        max_chat_message_length=int(os.getenv("MAX_CHAT_MESSAGE_LENGTH", "4000")),
        login_rate_limit=int(os.getenv("LOGIN_RATE_LIMIT", "5")),
        login_rate_window_seconds=int(os.getenv("LOGIN_RATE_WINDOW_SECONDS", "60")),
        chat_rate_limit=int(os.getenv("CHAT_RATE_LIMIT", "20")),
        chat_rate_window_seconds=int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "60")),
        # Default True (secure). Set SECURE_COOKIES=false only for local HTTP development.
        secure_cookies=os.getenv("SECURE_COOKIES", "true").lower() not in {"0", "false", "no", "off"},
        frontend_origins=frontend_origins,
        context_window_size=int(os.getenv("CONTEXT_WINDOW_SIZE", "10")),
        summary_trigger_count=int(os.getenv("SUMMARY_TRIGGER_COUNT", "10")),
        topic_extraction_enabled=os.getenv("TOPIC_EXTRACTION_ENABLED", "true").lower() in {"1", "true", "yes"},
        memory_api_rate_limit=int(os.getenv("MEMORY_API_RATE_LIMIT", "60")),
        memory_api_rate_window_seconds=int(os.getenv("MEMORY_API_RATE_WINDOW_SECONDS", "60")),
        csp_enabled=os.getenv("CSP_ENABLED", "true").lower() in {"1", "true", "yes"},
        guest_session_max_age_seconds=int(os.getenv("GUEST_SESSION_MAX_AGE_SECONDS", str(60 * 60 * 24))),
        query_cache_ttl_seconds=int(os.getenv("QUERY_CACHE_TTL_SECONDS", "300")),
        query_cache_enabled=os.getenv("QUERY_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"},
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0")),
        max_upload_size_bytes=int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))),
        document_upload_rate_limit=int(os.getenv("DOCUMENT_UPLOAD_RATE_LIMIT", "5")),
        document_upload_rate_window_seconds=int(os.getenv("DOCUMENT_UPLOAD_RATE_WINDOW_SECONDS", str(60 * 60))),
        content_generation_rate_limit=int(os.getenv("CONTENT_GENERATION_RATE_LIMIT", "20")),
        content_generation_rate_window_seconds=int(os.getenv("CONTENT_GENERATION_RATE_WINDOW_SECONDS", "60")),
    )
