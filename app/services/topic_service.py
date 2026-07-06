from __future__ import annotations

import json
import logging

from app.repositories.memory_repository import MemoryRepository


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------

_TOPIC_EXTRACTION_PROMPT = """\
You are a medical education assistant. Analyse the conversation below and identify the PRIMARY topic being studied.

Return ONLY valid JSON in this exact format (no extra text, no markdown):
{{
  "current_topic": "<primary medical topic, e.g. Nephrotic Syndrome>",
  "related_topics": ["<related topic 1>", "<related topic 2>"]
}}

Rules:
- current_topic must be a specific, named medical concept or disease (not generic like "medicine").
- related_topics should contain 1-4 closely related concepts mentioned or implied in the conversation.
- If the conversation has no clear medical topic, return {{"current_topic": null, "related_topics": []}}.

Conversation (most recent messages):
{history}
"""


class TopicService:
    """
    Extracts the current learning topic from a conversation and persists it
    in the ``session_topics`` table via :class:`MemoryRepository`.

    Topic extraction calls the LLM and is intended to be run asynchronously
    (in a background thread) so it never delays the main response.
    """

    def __init__(self, memory_repository: MemoryRepository) -> None:
        self._repo = memory_repository

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_topic(self, session_id: str) -> dict | None:
        """
        Return ``{"current_topic": str | None, "related_topics": [str]}``
        or ``None`` if no topic has been extracted yet.
        """
        if not session_id:
            return None
        try:
            return self._repo.get_topic(session_id)
        except Exception:
            logger.error("TopicService: could not fetch topic for session %s", session_id, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Extract + persist (call in background thread)
    # ------------------------------------------------------------------

    def extract_and_update_topic(
        self,
        session_id: str,
        recent_messages: list[dict],
        llm,
    ) -> None:
        """
        Use the LLM to identify the current topic from *recent_messages* and
        persist the result.  Designed to be called in a daemon thread so it
        never blocks the HTTP response.

        :param session_id:      The conversation / session identifier.
        :param recent_messages: List of ``{"role": ..., "content": ...}`` dicts.
        :param llm:             An instantiated Cohere LLM object (``ChatCohere``).
        """
        if not session_id or not recent_messages:
            return

        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages[-6:]
        )

        prompt = _TOPIC_EXTRACTION_PROMPT.format(history=history_text)

        try:
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)

            # Strip potential markdown fences and find the first JSON object using regex
            import re
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group()
            else:
                raw = "{}"

            data = json.loads(raw)
            current_topic = data.get("current_topic") or ""
            related_topics = [str(t) for t in (data.get("related_topics") or []) if t]

            if current_topic:
                self._repo.upsert_topic(session_id, current_topic, related_topics)
                logger.debug(
                    "TopicService: updated topic for %s → %s", session_id, current_topic
                )
        except json.JSONDecodeError:
            logger.warning("TopicService: LLM returned non-JSON for topic extraction")
        except Exception:
            logger.exception("TopicService: topic extraction failed for session %s", session_id)

    def update_topic(self, session_id: str, topic_data: dict) -> None:
        """Directly persist a known topic dict (bypasses LLM extraction)."""
        if not session_id or not topic_data.get("current_topic"):
            return
        try:
            self._repo.upsert_topic(
                session_id,
                topic_data["current_topic"],
                topic_data.get("related_topics") or [],
            )
        except Exception:
            logger.exception("TopicService: failed to persist topic for session %s", session_id)
