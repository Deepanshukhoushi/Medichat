from __future__ import annotations

from langchain_cohere import ChatCohere

from app.core.config.settings import AppSettings


def create_llm(settings: AppSettings) -> ChatCohere:
    return ChatCohere(
        model=settings.llm_model,
        temperature=0.3,
        cohere_api_key=settings.cohere_api_key,
    )
