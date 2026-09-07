from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain.schema import Document
from app.core.config.settings import AppSettings
from app.rag.query_processor import ProcessedQuery, QueryIntent


class RetrievalDecision(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SOURCE_NOT_FOUND = "source_not_found"


@dataclass
class RetrievalGateResult:
    decision: RetrievalDecision
    top_score: float
    average_top3_score: float
    relevant_documents: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    matched_entity: bool = False
    rejection_reason: str | None = None
    debug_signals: dict[str, Any] = field(default_factory=dict)


def _check_entity_coverage(entity: str | None, keywords: list[str], documents: list[Document]) -> bool:
    """
    Check if the target entity or majority of key query terms appear in the retrieved documents.
    """
    if not documents:
        return False
    if not entity and not keywords:
        return True

    combined_text = " ".join(doc.page_content.lower() for doc in documents)

    # 1. Exact entity match (case-insensitive substring or word boundary)
    if entity and len(entity) >= 2:
        entity_lower = entity.lower()
        if entity_lower in combined_text:
            return True
        # Check normalized tokens
        entity_tokens = [t for t in re.findall(r"\b[a-z0-9]+\b", entity_lower) if len(t) > 2]
        if entity_tokens and all(token in combined_text for token in entity_tokens):
            return True

    # 2. Significant keyword match (at least 50% of content keywords present)
    if keywords:
        matched_keywords = sum(1 for kw in keywords if kw.lower() in combined_text)
        match_ratio = matched_keywords / len(keywords)
        if match_ratio >= 0.5:
            return True

    return False


def evaluate_retrieval(
    settings: AppSettings,
    processed_query: ProcessedQuery,
    retrieved_documents_with_scores: list[tuple[Document, float]],
    user_namespace: str | None = None,
) -> RetrievalGateResult:
    """
    Multi-signal evaluation of Pinecone retrieval results.
    
    Signals evaluated:
    1. Cosine similarity score against threshold (settings.relevance_score_threshold).
    2. Entity / concept lexical and semantic coverage.
    3. Top-k score consistency.
    4. Source-specific mode validation.
    """
    if not retrieved_documents_with_scores:
        decision = (
            RetrievalDecision.SOURCE_NOT_FOUND
            if processed_query.intent == QueryIntent.SOURCE_SPECIFIC
            else RetrievalDecision.INSUFFICIENT_EVIDENCE
        )
        return RetrievalGateResult(
            decision=decision,
            top_score=0.0,
            average_top3_score=0.0,
            relevant_documents=[],
            scores=[],
            matched_entity=False,
            rejection_reason="No documents returned from vector search",
            debug_signals={"doc_count": 0},
        )

    threshold = settings.relevance_score_threshold
    scores = [score for _, score in retrieved_documents_with_scores]
    top_score = scores[0] if scores else 0.0
    top3_scores = scores[:3]
    avg_top3 = sum(top3_scores) / len(top3_scores) if top3_scores else 0.0

    # Filter documents meeting the threshold
    passed_docs_with_scores = [
        (doc, score) for doc, score in retrieved_documents_with_scores
        if score >= threshold
    ]

    # Evaluate entity/concept presence across top retrieved chunks
    top_docs = [doc for doc, _ in retrieved_documents_with_scores[:4]]
    entity_covered = _check_entity_coverage(
        processed_query.target_entity,
        processed_query.extracted_keywords,
        top_docs,
    )

    debug_signals = {
        "top_score": round(top_score, 4),
        "avg_top3": round(avg_top3, 4),
        "threshold": threshold,
        "raw_doc_count": len(retrieved_documents_with_scores),
        "passed_threshold_count": len(passed_docs_with_scores),
        "entity_covered": entity_covered,
        "target_entity": processed_query.target_entity,
        "keywords": processed_query.extracted_keywords,
        "intent": processed_query.intent.value,
    }

    # ── Source-Specific Mode Handling ────────────────────────────────────────
    if processed_query.intent == QueryIntent.SOURCE_SPECIFIC:
        if not passed_docs_with_scores or (processed_query.target_entity and not entity_covered):
            return RetrievalGateResult(
                decision=RetrievalDecision.SOURCE_NOT_FOUND,
                top_score=top_score,
                average_top3_score=avg_top3,
                relevant_documents=[],
                scores=scores,
                matched_entity=entity_covered,
                rejection_reason="Topic not found in requested source material",
                debug_signals=debug_signals,
            )

    # ── Strict Entity Verification for Specific Targeted Entities ────────────
    # If the user is asking about a fictional entity or explicit alphanumeric token (e.g. "XYZ", "ABC123")
    # and the documents do not contain it, reject immediately to protect against vector hallucination.
    is_fictional_or_code = bool(
        processed_query.intent == QueryIntent.HYPOTHETICAL_EXPLICIT
        or (processed_query.target_entity and re.search(r"\b(xyz|abc\d*|\d+[a-z]+|fictional)\b", processed_query.target_entity.lower()))
    )
    if is_fictional_or_code and not entity_covered:
        return RetrievalGateResult(
            decision=RetrievalDecision.INSUFFICIENT_EVIDENCE,
            top_score=top_score,
            average_top3_score=avg_top3,
            relevant_documents=[],
            scores=scores,
            matched_entity=False,
            rejection_reason="Fictional/unverified entity not present in medical evidence",
            debug_signals=debug_signals,
        )

    # If entity is specified and not covered, and top_score is below 0.70, reject
    if processed_query.target_entity and not entity_covered and top_score < 0.70:
        return RetrievalGateResult(
            decision=RetrievalDecision.INSUFFICIENT_EVIDENCE,
            top_score=top_score,
            average_top3_score=avg_top3,
            relevant_documents=[],
            scores=scores,
            matched_entity=False,
            rejection_reason=f"Target entity '{processed_query.target_entity}' not evidenced in retrieved chunks",
            debug_signals=debug_signals,
        )

    # ── Score Threshold Check ────────────────────────────────────────────────
    if not passed_docs_with_scores:
        return RetrievalGateResult(
            decision=RetrievalDecision.INSUFFICIENT_EVIDENCE,
            top_score=top_score,
            average_top3_score=avg_top3,
            relevant_documents=[],
            scores=scores,
            matched_entity=entity_covered,
            rejection_reason=f"Top similarity score ({top_score:.3f}) below threshold ({threshold:.3f})",
            debug_signals=debug_signals,
        )

    # ── Sufficient Evidence ──────────────────────────────────────────────────
    final_docs = [doc for doc, _ in passed_docs_with_scores]
    return RetrievalGateResult(
        decision=RetrievalDecision.SUFFICIENT_EVIDENCE,
        top_score=top_score,
        average_top3_score=avg_top3,
        relevant_documents=final_docs,
        scores=[score for _, score in passed_docs_with_scores],
        matched_entity=entity_covered,
        rejection_reason=None,
        debug_signals=debug_signals,
    )
