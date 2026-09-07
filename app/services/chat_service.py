from __future__ import annotations

import json
import logging
import re
import threading
import time
import concurrent.futures
from functools import cached_property
from uuid import uuid4

from app.core.config.settings import AppSettings
from app.core.security.exceptions import ServiceError
from app.rag.chains import create_llm
from app.rag.grounding import GroundingReport, GroundingStatus, validate_grounding
from app.rag.prompt_builder import build_memory_prompt
from app.rag.query_processor import ProcessedQuery, QueryIntent, process_query
from app.rag.retrieval_gate import (
    RetrievalDecision,
    RetrievalGateResult,
    evaluate_retrieval,
)
from app.rag.vector_store import build_user_namespace, retrieve_documents_with_scores
from app.services.conversation_service import ConversationService
from app.services.memory_service import MemoryService
from app.services.topic_service import TopicService
from app.services.summary_service import SummaryService


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        settings: AppSettings,
        conversation_repository,
        chat_history_repository,
        memory_service: MemoryService | None = None,
        topic_service: TopicService | None = None,
        summary_service: SummaryService | None = None,
    ) -> None:
        self.settings = settings
        self.conversation_service = ConversationService(
            conversation_repository, chat_history_repository, settings
        )
        self.memory_service = memory_service
        self.topic_service = topic_service
        self.summary_service = summary_service

    @cached_property
    def llm(self):
        return create_llm(self.settings)

    # ------------------------------------------------------------------
    # Session resolution
    # ------------------------------------------------------------------

    def _resolve_conversation_context(
        self, user_id: str, conversation_id: str | None
    ) -> tuple[str, bool, bool]:
        if conversation_id:
            return conversation_id, False, False

        try:
            return self.conversation_service.ensure_conversation(user_id), True, False
        except Exception:
            logger.exception(
                "Failed to resolve persistent conversation; switching to transient session"
            )
            return f"guest_{uuid4()}", False, True

    def _document_namespaces(self, user_id: str) -> list[str | None]:
        namespaces: list[str | None] = [None]
        user_namespace = build_user_namespace(user_id)
        if user_namespace:
            namespaces.append(user_namespace)
        return namespaces

    # ------------------------------------------------------------------
    # Background memory tasks
    # ------------------------------------------------------------------

    def _run_background_memory_tasks(
        self,
        session_id: str,
        recent_messages: list[dict],
        message_count: int,
    ) -> None:
        """Fire off topic extraction and optional summarisation using Celery."""
        try:
            from app.tasks.memory_tasks import extract_topic_task, generate_summary_task

            if self.topic_service and self.settings.topic_extraction_enabled and recent_messages:
                extract_topic_task.delay(session_id, recent_messages)

            if (
                self.summary_service
                and self.memory_service
                and self.summary_service.should_summarize(message_count)
            ):
                generate_summary_task.delay(session_id, recent_messages)
        except Exception:
            logger.warning("Background memory task trigger skipped or unavailable", exc_info=False)

    # ------------------------------------------------------------------
    # RAG Pipeline: Query -> Retrieval -> Gate -> Generate -> Validate
    # ------------------------------------------------------------------

    def _process_and_retrieve(
        self,
        user_input: str,
        user_id: str,
        recent_messages: list[dict],
        topic_memory: dict | None,
    ) -> tuple[ProcessedQuery, RetrievalGateResult]:
        """
        1. Process & contextually rewrite query (resolves 'it', 'its complications').
        2. Retrieve vector embeddings from Pinecone.
        3. Evaluate multi-signal retrieval confidence gate.
        """
        processed_query = process_query(
            user_query=user_input,
            chat_history=recent_messages,
            topic_memory=topic_memory,
        )

        user_ns = build_user_namespace(user_id)
        search_query = processed_query.rewritten_query

        retrieved_documents_with_scores = retrieve_documents_with_scores(
            self.settings,
            search_query,
            k=self.settings.retriever_k,
            namespaces=self._document_namespaces(user_id),
        )

        gate_result = evaluate_retrieval(
            settings=self.settings,
            processed_query=processed_query,
            retrieved_documents_with_scores=retrieved_documents_with_scores,
            user_namespace=user_ns,
        )

        return processed_query, gate_result

    def _generate_and_validate_answer(
        self,
        user_input: str,
        processed_query: ProcessedQuery,
        gate_result: RetrievalGateResult,
        recent_messages: list[dict],
        topic_memory: dict | None,
        session_summary: str | None,
    ) -> tuple[str, GroundingReport | None]:
        """
        Execute grounded generation and claim-level validation.
        """
        target_name = processed_query.target_entity or user_input

        # ── Fast-path 1: Ambiguous Follow-Up Clarification ──────────────────
        if processed_query.is_ambiguous and processed_query.ambiguity_candidates:
            candidates_str = " or ".join(f"**{c}**" for c in processed_query.ambiguity_candidates[:2])
            answer = (
                f"Your question refers to a previous condition, but we were discussing multiple topics ({candidates_str}).\n\n"
                f"Could you please clarify which one you would like to know about?"
            )
            return answer, GroundingReport(
                overall_status=GroundingStatus.REFUSAL,
                is_valid=True,
                refusal_detected=True,
            )

        # ── Fast-path 2: Source-Specific Document Not Found ──────────────────
        if gate_result.decision == RetrievalDecision.SOURCE_NOT_FOUND:
            answer = (
                f"I couldn't find information about **{target_name}** in the uploaded material."
            )
            return answer, GroundingReport(
                overall_status=GroundingStatus.REFUSAL,
                is_valid=True,
                refusal_detected=True,
            )

        # ── Fast-path 3: Fictional / Hypothetical / Out-of-Knowledge Entities ──
        if (
            gate_result.decision == RetrievalDecision.INSUFFICIENT_EVIDENCE
            and (
                processed_query.intent == QueryIntent.HYPOTHETICAL_EXPLICIT
                or (processed_query.target_entity and len(processed_query.target_entity) > 2 and gate_result.top_score < 0.72)
            )
        ):
            answer = (
                f"I couldn't find reliable information about **{target_name}** in the available medical sources, "
                f"so I don't want to speculate or invent an answer."
            )
            return answer, GroundingReport(
                overall_status=GroundingStatus.REFUSAL,
                is_valid=True,
                refusal_detected=True,
            )

        # ── Grounded Generation ──────────────────────────────────────────────
        docs_to_use = gate_result.relevant_documents if gate_result.decision == RetrievalDecision.SUFFICIENT_EVIDENCE else []
        
        enriched_prompt = build_memory_prompt(
            user_query=processed_query.rewritten_query,
            retrieved_documents=docs_to_use,
            topic_memory=topic_memory,
            session_summary=session_summary,
            chat_history=recent_messages,
            intent=processed_query.intent,
            is_strict_regeneration=False,
        )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.llm.invoke, enriched_prompt)
                response = future.result(timeout=self.settings.llm_timeout_seconds)
            
            raw_answer = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()
        except concurrent.futures.TimeoutError:
            logger.warning("LLM call timed out after %s seconds", self.settings.llm_timeout_seconds)
            raw_answer = "I apologize, but the request timed out. Please try asking your question again."
        except Exception:
            logger.exception("LLM generation failed")
            raw_answer = "Unable to generate an answer right now. Please try again in a moment."

        # ── Claim-Level Grounding Validation ─────────────────────────────────
        grounding_report = validate_grounding(
            answer=raw_answer,
            retrieved_documents=docs_to_use,
            processed_query=processed_query,
        )

        # ── Regeneration on Validation Failure ───────────────────────────────
        if not grounding_report.is_valid and self.settings.grounding_validation_enabled:
            logger.info("Answer failed grounding check (%s); attempting regeneration", grounding_report.overall_status.value)
            
            strict_prompt = build_memory_prompt(
                user_query=processed_query.rewritten_query,
                retrieved_documents=docs_to_use,
                topic_memory=topic_memory,
                session_summary=session_summary,
                chat_history=recent_messages,
                intent=processed_query.intent,
                is_strict_regeneration=True,
            )
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.llm.invoke, strict_prompt)
                    regen_response = future.result(timeout=self.settings.llm_timeout_seconds)
                
                regen_answer = (
                    regen_response.content if hasattr(regen_response, "content") else str(regen_response)
                ).strip()

                regen_report = validate_grounding(
                    answer=regen_answer,
                    retrieved_documents=docs_to_use,
                    processed_query=processed_query,
                )

                if regen_report.is_valid:
                    return regen_answer, regen_report
                else:
                    logger.warning("Regenerated answer also failed grounding; returning safe refusal")
                    safe_fallback = (
                        f"I don't have enough reliable information in the available medical sources to answer '{target_name}' confidently."
                    )
                    return safe_fallback, regen_report

            except Exception:
                logger.exception("Regeneration attempt failed")
                safe_fallback = (
                    f"I don't have enough reliable information in the available medical sources to answer '{target_name}' confidently."
                )
                return safe_fallback, grounding_report

        return raw_answer, grounding_report

    # ------------------------------------------------------------------
    # Main synchronous answer method
    # ------------------------------------------------------------------

    def get_answer(
        self, user_input: str, user_id: str, conversation_id: str | None = None
    ) -> str:
        try:
            resolved_conversation_id, persistence_available, persistence_degraded = (
                self._resolve_conversation_context(user_id, conversation_id)
            )

            recent_messages: list[dict] = []
            topic_memory: dict | None = None
            session_summary: str | None = None

            if self.memory_service:
                recent_messages = self.memory_service.get_recent_messages(
                    resolved_conversation_id,
                    limit=self.settings.context_window_size,
                )
            if self.topic_service:
                topic_memory = self.topic_service.get_topic(resolved_conversation_id)
            if self.summary_service:
                session_summary = self.summary_service.get_summary(resolved_conversation_id)

            processed_query, gate_result = self._process_and_retrieve(
                user_input=user_input,
                user_id=user_id,
                recent_messages=recent_messages,
                topic_memory=topic_memory,
            )

            answer, grounding_report = self._generate_and_validate_answer(
                user_input=user_input,
                processed_query=processed_query,
                gate_result=gate_result,
                recent_messages=recent_messages,
                topic_memory=topic_memory,
                session_summary=session_summary,
            )

            if self.settings.dev_debug_logging:
                logger.info(
                    "RAG_DEBUG_TRACE: query=%r rewritten=%r intent=%s top_score=%.4f decision=%s status=%s valid=%s",
                    user_input,
                    processed_query.rewritten_query,
                    processed_query.intent.value,
                    gate_result.top_score,
                    gate_result.decision.value,
                    grounding_report.overall_status.value if grounding_report else "N/A",
                    grounding_report.is_valid if grounding_report else True,
                )

            if persistence_degraded:
                answer = (
                    f"{answer}\n\n"
                    "[!] Conversation storage is temporarily unavailable; "
                    "this reply may not be saved."
                )

            if self.memory_service and not resolved_conversation_id.startswith(
                self.settings.guest_session_prefix
            ):
                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "user", user_input
                )
                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "assistant", answer
                )
                message_count = self.memory_service.get_message_count(resolved_conversation_id)
                all_messages = recent_messages + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": answer},
                ]
                self._run_background_memory_tasks(
                    resolved_conversation_id, all_messages, message_count
                )

            return answer

        except Exception as exc:
            logger.exception("Failed to generate answer")
            raise ServiceError("Unable to generate an answer right now") from exc

    # ------------------------------------------------------------------
    # Streaming answer method (Generate -> Validate -> Stream)
    # ------------------------------------------------------------------
    #
    # TODO (Issue #17 / Performance): The current implementation is NOT true
    # token-streaming.  The full answer is generated internally first, then
    # re-chunked with time.sleep(0.015) for UX cadence, blocking one sync
    # worker for the entire LLM round-trip.  To fix:
    #   1. Switch the Cohere LLM call to use stream=True and yield tokens
    #      as the model produces them.
    #   2. Run gunicorn with --worker-class gevent (or migrate to an ASGI
    #      server such as uvicorn) so workers are not exhausted during
    #      streaming sessions.
    # Until that migration, concurrent chat capacity ≈ number of workers.
    #

    def get_answer_stream(
        self,
        user_input: str,
        user_id: str,
        conversation_id: str | None = None,
        is_regenerate: bool = False,
    ):
        """
        Stream the validated answer via Server-Sent Events (SSE).
        Generates full candidate, validates grounding, and streams validated tokens
        preserving the exact Angular frontend SSE contract.
        """
        full_answer = ""
        already_persisted = False
        resolved_conversation_id = None
        is_new_conversation = False
        recent_messages: list[dict] = []

        try:
            resolved_conversation_id, is_new_conversation, persistence_degraded = (
                self._resolve_conversation_context(user_id, conversation_id)
            )
            yield f"data: {json.dumps({'conversation_id': resolved_conversation_id})}\n\n"

            # ── 0. Set conversation title on initial turn ────────────────────────
            if is_new_conversation and not resolved_conversation_id.startswith(self.settings.guest_session_prefix):
                try:
                    title = user_input[:60].strip()
                    self.conversation_service.conversation_repository.update_title(
                        resolved_conversation_id, title, user_id
                    )
                except Exception:
                    pass

            # ── 1. Load context ──────────────────────────────────────────────────
            topic_memory = None
            session_summary = None

            if self.memory_service:
                recent_messages = self.memory_service.get_recent_messages(
                    resolved_conversation_id,
                    limit=self.settings.context_window_size,
                )
            if self.topic_service:
                topic_memory = self.topic_service.get_topic(resolved_conversation_id)
            if self.summary_service:
                session_summary = self.summary_service.get_summary(resolved_conversation_id)

            # ── 2. Process query & Retrieve ──────────────────────────────────────
            processed_query, gate_result = self._process_and_retrieve(
                user_input=user_input,
                user_id=user_id,
                recent_messages=recent_messages,
                topic_memory=topic_memory,
            )

            # ── 3. Generate & Validate Complete Answer Internally ────────────────
            validated_answer, grounding_report = self._generate_and_validate_answer(
                user_input=user_input,
                processed_query=processed_query,
                gate_result=gate_result,
                recent_messages=recent_messages,
                topic_memory=topic_memory,
                session_summary=session_summary,
            )

            full_answer = validated_answer

            if self.settings.dev_debug_logging:
                logger.info(
                    "RAG_DEBUG_TRACE: query=%r rewritten=%r intent=%s top_score=%.4f decision=%s status=%s valid=%s",
                    user_input,
                    processed_query.rewritten_query,
                    processed_query.intent.value,
                    gate_result.top_score,
                    gate_result.decision.value,
                    grounding_report.overall_status.value if grounding_report else "N/A",
                    grounding_report.is_valid if grounding_report else True,
                )

            # ── 4. Stream Validated Answer Tokens to SSE Client ──────────────────
            # Chunk by tokens / words for smooth natural streaming
            chunks = re.findall(r"\S+|\n+|\s+", validated_answer)
            for chunk in chunks:
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                time.sleep(0.015)  # Natural token cadence

            # ── 5. Persist to database ───────────────────────────────────────────
            assistant_msg_id = None
            if self.memory_service and not resolved_conversation_id.startswith(
                self.settings.guest_session_prefix
            ):
                try:
                    if is_regenerate and hasattr(self.memory_service.chat_history_repository, "delete_latest_exchange"):
                        self.memory_service.chat_history_repository.delete_latest_exchange(
                            resolved_conversation_id, user_id
                        )

                    self.memory_service.save_message(
                        resolved_conversation_id, user_id, "user", user_input
                    )
                    assistant_msg_id = self.memory_service.save_message(
                        resolved_conversation_id, user_id, "assistant", full_answer
                    )
                    message_count = self.memory_service.get_message_count(resolved_conversation_id)
                    all_messages = recent_messages + [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": full_answer},
                    ]
                    self._run_background_memory_tasks(
                        resolved_conversation_id, all_messages, message_count
                    )
                    already_persisted = True
                except Exception as exc:
                    logger.warning("Persistence degraded: %s", exc)
                    yield f"data: {json.dumps({'warning': 'Persistence degraded - your messages may not be saved.'})}\n\n"

            if assistant_msg_id:
                yield f"data: {json.dumps({'message_id': str(assistant_msg_id)})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("Failed to stream answer")
            yield f"data: {json.dumps({'error': 'Unable to generate an answer right now'})}\n\n"
        finally:
            if (
                already_persisted
                or not full_answer
                or not self.memory_service
                or not resolved_conversation_id
                or resolved_conversation_id.startswith(self.settings.guest_session_prefix)
            ):
                return

            try:
                if is_regenerate and hasattr(self.memory_service.chat_history_repository, "delete_latest_exchange"):
                    self.memory_service.chat_history_repository.delete_latest_exchange(
                        resolved_conversation_id, user_id
                    )

                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "user", user_input
                )
                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "assistant", full_answer
                )
                message_count = self.memory_service.get_message_count(resolved_conversation_id)
                all_messages = recent_messages + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": full_answer},
                ]
                self._run_background_memory_tasks(
                    resolved_conversation_id, all_messages, message_count
                )
            except Exception as exc:
                logger.warning("Persistence recovery degraded: %s", exc)
