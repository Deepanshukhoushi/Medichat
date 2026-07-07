from __future__ import annotations

import logging
import threading
import time
import concurrent.futures
from functools import cached_property
from uuid import uuid4

from app.core.config.settings import AppSettings
from app.core.security.exceptions import ServiceError
from app.rag.chains import build_qa_chain, create_llm
from app.rag.prompt_builder import build_memory_prompt
from app.rag.retrieval import is_relevant_score
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

    @cached_property
    def qa_chain(self):
        return build_qa_chain(self.settings)

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
        from app.tasks.memory_tasks import extract_topic_task, generate_summary_task

        if self.topic_service and self.settings.topic_extraction_enabled and recent_messages:
            extract_topic_task.delay(session_id, recent_messages)

        if (
            self.summary_service
            and self.memory_service
            and self.summary_service.should_summarize(message_count)
        ):
            generate_summary_task.delay(session_id, recent_messages)

    # ------------------------------------------------------------------
    # Main answer method
    # ------------------------------------------------------------------

    def get_answer(
        self, user_input: str, user_id: str, conversation_id: str | None = None
    ) -> str:
        try:
            resolved_conversation_id, persistence_available, persistence_degraded = (
                self._resolve_conversation_context(user_id, conversation_id)
            )

            # ── 1. Fetch memory context ──────────────────────────────────────
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

            # ── 2. Retrieve documents from Pinecone ──────────────────────────
            retrieved_documents_with_scores = retrieve_documents_with_scores(
                self.settings,
                user_input,
                k=self.settings.retriever_k,
                namespaces=self._document_namespaces(user_id),
            )
            relevant_documents = [
                document
                for document, score in retrieved_documents_with_scores
                if is_relevant_score(score, self.settings.relevance_score_threshold)
            ]

            # ── 3. Build enriched prompt & call LLM ─────────────────────────
            answer = self._generate_answer(
                user_input=user_input,
                relevant_documents=relevant_documents,
                recent_messages=recent_messages,
                topic_memory=topic_memory,
                session_summary=session_summary,
                resolved_conversation_id=resolved_conversation_id,
            )

            # ── 4. Degrade notice ────────────────────────────────────────────
            if persistence_degraded:
                answer = (
                    f"{answer}\n\n"
                    "⚠️ Conversation storage is temporarily unavailable; "
                    "this reply may not be saved."
                )

            # ── 5. Persist messages + fire background tasks ──────────────
            if self.memory_service and not resolved_conversation_id.startswith(
                self.settings.guest_session_prefix
            ):
                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "user", user_input
                )
                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "assistant", answer
                )
                # ── 6. Background: update topic + maybe summarise ────────────
                message_count = self.memory_service.get_message_count(
                    resolved_conversation_id
                )
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

    def get_answer_stream(
        self,
        user_input: str,
        user_id: str,
        conversation_id: str | None = None,
        is_regenerate: bool = False,
    ):
        """
        Stream the answer via Server-Sent Events (SSE).
        Yields chunk tokens and performs background tasks after completion.
        """
        # Pre-initialize so the finally block never hits UnboundLocalError
        full_answer = ""
        stream_completed = False
        already_persisted = False  # guard: skip finally-block if try already persisted
        resolved_conversation_id = None
        is_new_conversation = False
        recent_messages: list[dict] = []
        relevant_documents: list = []

        try:
            resolved_conversation_id, is_new_conversation, persistence_degraded = (
                self._resolve_conversation_context(user_id, conversation_id)
            )
            yield f"data: {{\"conversation_id\": \"{resolved_conversation_id}\"}}\n\n"

            # ── 0. Set conversation title from the first message ─────────────────
            if is_new_conversation and not resolved_conversation_id.startswith(self.settings.guest_session_prefix):
                try:
                    title = user_input[:60].strip()
                    self.conversation_service.conversation_repository.update_title(resolved_conversation_id, title)
                except Exception:
                    pass  # non-critical

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

            # ── 2. Retrieve documents ────────────────────────────────────────────
            retrieved_documents_with_scores = retrieve_documents_with_scores(
                self.settings,
                user_input,
                k=self.settings.retriever_k,
                namespaces=self._document_namespaces(user_id),
            )
            relevant_documents = [
                document
                for document, score in retrieved_documents_with_scores
                if is_relevant_score(score, self.settings.relevance_score_threshold)
            ]

            # ── 3. Stream LLM Answer ─────────────────────────────────────────────
            stream_generator = self._generate_answer_stream(
                user_input=user_input,
                relevant_documents=relevant_documents,
                recent_messages=recent_messages,
                topic_memory=topic_memory,
                session_summary=session_summary,
            )

            for chunk in stream_generator:
                full_answer += chunk
                import json
                # Replace newlines so JSON doesn't break
                safe_chunk = chunk.replace('\n', '\\n').replace('\r', '\\r').replace('"', '\\"')
                yield f"data: {{\"token\": \"{safe_chunk}\"}}\n\n"
            stream_completed = True
            
            # ── 4. Suffix ────────────────────────────────────────────────────────
            suffix = (
                self.settings.indexed_answer_suffix
                if relevant_documents
                else self.settings.general_answer_suffix
            )
            if not full_answer.endswith(suffix):
                full_answer = f"{full_answer}\n\n{suffix}"
                safe_suffix = f"\\n\\n{suffix}".replace('"', '\\"')
                yield f"data: {{\"token\": \"{safe_suffix}\"}}\n\n"

            # ── 5. Persist to database ───────────────────────────────────────────
            if self.memory_service and not resolved_conversation_id.startswith(
                self.settings.guest_session_prefix
            ):
                try:
                    if is_regenerate:
                        # For regeneration, delete the last pair of messages before persisting the new ones
                        if hasattr(self.memory_service.chat_history_repository, "delete_latest_exchange"):
                            self.memory_service.chat_history_repository.delete_latest_exchange(resolved_conversation_id)

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
                    already_persisted = True  # prevent finally from duplicating
                except Exception as exc:
                    logger.warning("Persistence degraded: %s", exc)
                    yield f"data: {{\"warning\": \"Persistence degraded — your messages may not be saved.\"}}\n\n"

            # ── 6. Emit structured citation sources ──────────────────────────────
            if relevant_documents:
                import json as _json
                sources = []
                seen: set[str] = set()
                for doc in relevant_documents:
                    meta = getattr(doc, "metadata", {}) or {}
                    source_name = meta.get("source", "Medical Reference")
                    page = meta.get("page")
                    # Derive a human-readable title from the filename (strip path + ext)
                    title = source_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    title = title.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
                    chapter = f"Page {page}" if page is not None else "Reference"
                    key = f"{source_name}:{page}"
                    if key not in seen:
                        seen.add(key)
                        sources.append({"title": title, "source": source_name, "chapter": chapter})
                yield f"data: {_json.dumps({'sources': sources})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("Failed to stream answer")
            yield f"data: {{\"error\": \"Unable to generate an answer right now\"}}\n\n"
        finally:
            # Only run the persistence logic below as a disconnect-recovery path:
            # if the try block already persisted successfully, skip to avoid
            # writing duplicate rows and firing Celery tasks twice.
            if (
                already_persisted
                or not full_answer
                or not self.memory_service
                or not resolved_conversation_id
                or resolved_conversation_id.startswith(self.settings.guest_session_prefix)
            ):
                return


            try:
                persisted_answer = full_answer
                if stream_completed:
                    suffix = (
                        self.settings.indexed_answer_suffix
                        if relevant_documents
                        else self.settings.general_answer_suffix
                    )
                    if not persisted_answer.endswith(suffix):
                        persisted_answer = f"{persisted_answer}\n\n{suffix}"

                if is_regenerate and hasattr(self.memory_service.chat_history_repository, "delete_latest_exchange"):
                    self.memory_service.chat_history_repository.delete_latest_exchange(
                        resolved_conversation_id
                    )

                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "user", user_input
                )
                self.memory_service.save_message(
                    resolved_conversation_id, user_id, "assistant", persisted_answer
                )
                message_count = self.memory_service.get_message_count(resolved_conversation_id)
                all_messages = recent_messages + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": persisted_answer},
                ]
                self._run_background_memory_tasks(
                    resolved_conversation_id, all_messages, message_count
                )
            except Exception as exc:
                logger.warning("Persistence degraded: %s", exc)

    # ------------------------------------------------------------------
    # Answer generation (with fallback chain)
    # ------------------------------------------------------------------

    def _generate_answer_stream(
        self,
        user_input: str,
        relevant_documents: list,
        recent_messages: list[dict],
        topic_memory: dict | None,
        session_summary: str | None,
    ):
        """
        Yields chunks directly from the LLM. Currently uses the memory-aware path 
        for streaming.
        """
        enriched_prompt = build_memory_prompt(
            user_query=user_input,
            retrieved_documents=relevant_documents,
            topic_memory=topic_memory,
            session_summary=session_summary,
            chat_history=recent_messages,
        )
        try:
            for chunk in self.llm.stream(enriched_prompt):
                yield chunk.content if hasattr(chunk, "content") else str(chunk)
        except Exception:
            logger.exception("LLM stream failed")
            yield "Sorry, I encountered an error while streaming the response."

    def _generate_answer(
        self,
        user_input: str,
        relevant_documents: list,
        recent_messages: list[dict],
        topic_memory: dict | None,
        session_summary: str | None,
        resolved_conversation_id: str,
    ) -> str:
        """
        Try the memory-aware path first, then the legacy QA chain, then a
        bare LLM call as a last resort.
        """
        has_memory = bool(
            recent_messages or topic_memory or session_summary
        )

        # ── Memory-aware path (primary) ──────────────────────────────────────
        if has_memory or relevant_documents:
            enriched_prompt = build_memory_prompt(
                user_query=user_input,
                retrieved_documents=relevant_documents,
                topic_memory=topic_memory,
                session_summary=session_summary,
                chat_history=recent_messages,
            )

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.llm.invoke, enriched_prompt)
                    response = future.result(timeout=self.settings.llm_timeout_seconds)
                
                answer = (
                    response.content if hasattr(response, "content") else str(response)
                ).strip()
                suffix = (
                    self.settings.indexed_answer_suffix
                    if relevant_documents
                    else self.settings.general_answer_suffix
                )
                if not answer.endswith(suffix):
                    answer = f"{answer}\n\n{suffix}"
                return answer
            except concurrent.futures.TimeoutError:
                logger.warning("Memory-aware LLM call timed out after %s seconds", self.settings.llm_timeout_seconds)
            except Exception:
                logger.exception("Memory-aware LLM call failed; falling back to QA chain")

        # ── Legacy QA chain fallback ─────────────────────────────────────────
        if relevant_documents:
            try:
                from langchain_community.chat_message_histories import ChatMessageHistory
                history = ChatMessageHistory()
                for msg in reversed(recent_messages):
                    if msg.get("role") == "user":
                        history.add_user_message(msg.get("content", ""))
                    elif msg.get("role") == "assistant":
                        history.add_ai_message(msg.get("content", ""))
                        
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self.qa_chain.invoke,
                        {
                            "input": user_input,
                            "context": relevant_documents,
                            "history": history.messages,
                        }
                    )
                    response = future.result(timeout=self.settings.llm_timeout_seconds)
                
                if isinstance(response, dict):
                    answer = str(
                        response.get("answer") or response.get("output") or response
                    ).strip()
                elif hasattr(response, "content"):
                    answer = str(response.content).strip()
                else:
                    answer = str(response).strip()

                if not answer.endswith(self.settings.indexed_answer_suffix):
                    answer = f"{answer}\n\n{self.settings.indexed_answer_suffix}"
                return answer
            except concurrent.futures.TimeoutError:
                logger.warning("QA chain fallback timed out after %s seconds", self.settings.llm_timeout_seconds)
            except Exception:
                logger.exception("QA chain fallback failed; using direct LLM call")
                context_text = "\n".join(
                    doc.page_content[:500] for doc in relevant_documents[:3]
                )
                response = self.llm.invoke(
                    f"Answer this medical question using the provided context:\n\n"
                    f"Context: {context_text}\n\nQuestion: {user_input}\n\n"
                    "If the context is not enough, use general medical knowledge and say so clearly."
                )
                answer = f"{str(response.content).strip()}\n\n{self.settings.indexed_answer_suffix}"
                return answer

        # ── Bare LLM (no docs retrieved) ────────────────────────────────────
        response = self.llm.invoke(
            f"Answer this medical question using general medical knowledge:\n\n"
            f"Question: {user_input}\n\n"
            "The question was not matched in the indexed medical database."
        )
        return (
            f"{str(response.content).strip()}\n\n{self.settings.general_answer_suffix}"
        )
