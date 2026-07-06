from __future__ import annotations

import logging

from app.repositories.memory_repository import MemoryRepository


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Summarisation prompt
# ------------------------------------------------------------------

_SUMMARIZE_PROMPT = """\
You are a medical education assistant creating a concise study summary.

{existing_summary_section}

New conversation messages to incorporate:
{history}

Generate an updated, structured study summary in bullet-point format.

Format:
## Current Learning Context
- Key concepts discussed
- Important clinical features mentioned
- Diagnostic criteria or management points covered
- Comparisons or differentials noted

Keep the summary focused, accurate, and useful for a medical student reviewing their session.
Return ONLY the summary text (no extra commentary).
"""

_EXISTING_SUMMARY_SECTION = """\
Existing summary (incorporate and expand):
{summary}
"""


class SummaryService:
    """
    Generates and persists rolling study summaries in ``conversation_summaries``.

    A new summary is generated every ``trigger_count`` messages.  When a
    summary already exists it is provided to the LLM so the new one extends
    (rather than replaces) the prior context.
    """

    def __init__(
        self,
        memory_repository: MemoryRepository,
        trigger_count: int = 10,
    ) -> None:
        self._repo = memory_repository
        self.trigger_count = trigger_count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> str | None:
        """Return the stored study summary string, or ``None``."""
        if not session_id:
            return None
        try:
            return self._repo.get_summary(session_id)
        except Exception:
            logger.error("SummaryService: could not fetch summary for session %s", session_id, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def should_summarize(self, message_count: int) -> bool:
        """True when message_count is a non-zero multiple of trigger_count."""
        return message_count > 0 and message_count % self.trigger_count == 0

    # ------------------------------------------------------------------
    # Generate + persist (call in background thread)
    # ------------------------------------------------------------------

    def generate_and_save_summary(
        self,
        session_id: str,
        recent_messages: list[dict],
        llm,
    ) -> None:
        """
        Generate a new study summary for the session and persist it.

        Designed to be called in a daemon thread so it never blocks the HTTP
        response.

        :param session_id:      The conversation / session identifier.
        :param recent_messages: List of ``{"role": ..., "content": ...}`` dicts.
        :param llm:             An instantiated Cohere LLM object.
        """
        if not session_id or not recent_messages:
            return

        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages
        )

        existing_summary = self.get_summary(session_id)
        if existing_summary:
            existing_section = _EXISTING_SUMMARY_SECTION.format(summary=existing_summary)
        else:
            existing_section = ""

        prompt = _SUMMARIZE_PROMPT.format(
            existing_summary_section=existing_section,
            history=history_text,
        )

        try:
            response = llm.invoke(prompt)
            summary_text = (
                response.content if hasattr(response, "content") else str(response)
            ).strip()

            if summary_text:
                self._repo.upsert_summary(session_id, summary_text)
                logger.info(
                    "SummaryService: generated summary for session %s (%d chars)",
                    session_id,
                    len(summary_text),
                )
        except Exception:
            logger.exception(
                "SummaryService: summary generation failed for session %s", session_id
            )
