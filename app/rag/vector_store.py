from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from app.core.cache.query_cache import QueryCache
from app.core.config.settings import AppSettings
from app.rag.embeddings import create_embeddings


@lru_cache(maxsize=4)
def get_pinecone_client(api_key: str) -> Pinecone:
    return Pinecone(api_key=api_key)


def build_user_namespace(user_id: str | None) -> str | None:
    """Return a deterministic Pinecone namespace for a user."""
    if not user_id:
        return None
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return f"user_{digest}"


def load_vector_store(settings: AppSettings, namespace: str | None = None) -> PineconeVectorStore:
    return PineconeVectorStore.from_existing_index(
        index_name=settings.index_name,
        embedding=create_embeddings(settings),
        namespace=namespace,
    )


def build_retriever(settings: AppSettings, namespace: str | None = None):
    vector_store = load_vector_store(settings, namespace=namespace)
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )


def _normalize_namespaces(namespaces: list[str | None] | tuple[str | None, ...] | None) -> list[str | None]:
    if namespaces is None:
        return [None]
    normalized: list[str | None] = []
    seen: set[str | None] = set()
    for namespace in namespaces:
        if namespace in seen:
            continue
        seen.add(namespace)
        normalized.append(namespace)
    return normalized or [None]


def _result_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    metadata_key = json.dumps(metadata, sort_keys=True, default=str)
    return f"{doc.page_content}::{metadata_key}"


def retrieve_documents_with_scores(
    settings: AppSettings,
    query: str,
    k: int | None = None,
    namespaces: list[str | None] | tuple[str | None, ...] | None = None,
):
    search_namespaces = _normalize_namespaces(namespaces)
    cache_scope = "|".join(namespace or "default" for namespace in search_namespaces)
    query_cache = QueryCache(settings)
    cached_results = query_cache.get(query, scope=cache_scope)
    
    if cached_results is not None:
        # Re-hydrate if needed (Redis returns lists/dicts instead of Document objects)
        hydrated_results = []
        for item in cached_results:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                doc_data, score = item
                if isinstance(doc_data, dict):
                    hydrated_results.append((Document(
                        page_content=doc_data.get("page_content", ""),
                        metadata=doc_data.get("metadata", {})
                    ), score))
                else:
                    hydrated_results.append(item)
            else:
                hydrated_results.append(item)
        return hydrated_results

    vector_store = load_vector_store(settings)
    results: list[tuple[Document, float]] = []
    for namespace in search_namespaces:
        namespace_results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=k or settings.retriever_k,
            namespace=namespace,
        )
        results.extend(namespace_results)

    # Merge duplicates by document content + metadata while preserving best score.
    merged_results: dict[str, tuple[Document, float]] = {}
    for doc, score in results:
        key = _result_key(doc)
        existing = merged_results.get(key)
        if existing is None or score > existing[1]:
            merged_results[key] = (doc, score)

    sorted_results = sorted(
        merged_results.values(),
        key=lambda item: item[1],
        reverse=True,
    )

    # Serialize results for cache if needed, only saving relevant documents
    serializable_results = []
    for doc, score in sorted_results:
        if score >= settings.relevance_score_threshold:
            serializable_results.append(({"page_content": doc.page_content, "metadata": doc.metadata}, score))
    
    if serializable_results:
        query_cache.set(query, serializable_results, scope=cache_scope)
    return sorted_results


def rebuild_index(settings: AppSettings, documents, force: bool = False) -> PineconeVectorStore:
    pinecone_client = get_pinecone_client(settings.pinecone_api_key)
    if settings.index_name in pinecone_client.list_indexes().names():
        if not force:
            raise RuntimeError(f"Index '{settings.index_name}' already exists. Use --force to rebuild it.")
        pinecone_client.delete_index(settings.index_name)

    pinecone_client.create_index(
        name=settings.index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )

    return PineconeVectorStore.from_documents(
        documents=documents,
        embedding=create_embeddings(settings),
        index_name=settings.index_name,
    )
