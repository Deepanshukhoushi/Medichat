from __future__ import annotations

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_cohere import ChatCohere

from app.core.config.settings import AppSettings
from app.rag.prompts import build_prompt


def create_llm(settings: AppSettings) -> ChatCohere:
    return ChatCohere(
        model=settings.llm_model,
        temperature=0.3,
        cohere_api_key=settings.cohere_api_key,
    )


def build_qa_chain(settings: AppSettings):
    llm = create_llm(settings)
    prompt = build_prompt()
    return create_stuff_documents_chain(llm, prompt)
