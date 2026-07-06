from __future__ import annotations

from langchain_cohere import CohereEmbeddings

from app.core.config.settings import AppSettings
from app.core.security.exceptions import ConfigurationError


def create_embeddings(settings: AppSettings) -> CohereEmbeddings:
    if not settings.cohere_api_key:
        raise ConfigurationError("Missing Cohere API key")
    return CohereEmbeddings(
        model=settings.embedding_model,
        cohere_api_key=settings.cohere_api_key,
    )

