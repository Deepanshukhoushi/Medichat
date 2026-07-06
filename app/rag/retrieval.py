from __future__ import annotations

from app.core.config.settings import AppSettings
from app.rag.vector_store import retrieve_documents_with_scores


def retrieve_documents(settings: AppSettings, query: str, namespaces: list[str | None] | tuple[str | None, ...] | None = None):
    results = retrieve_documents_with_scores(settings, query, namespaces=namespaces)
    return [document for document, _score in results]


def is_relevant_score(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold
