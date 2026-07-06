from __future__ import annotations

import logging

from app.core.security.exceptions import RepositoryError


logger = logging.getLogger(__name__)


class MemoryRepository:
    """CRUD for session_topics and conversation_summaries tables."""

    def __init__(self, supabase_client) -> None:
        self.supabase = supabase_client

    # ------------------------------------------------------------------
    # session_topics
    # ------------------------------------------------------------------

    def get_topic(self, session_id: str) -> dict | None:
        """Return the stored topic dict or None if not set."""
        try:
            result = (
                self.supabase.table("session_topics")
                .select("current_topic, related_topics")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
            if result.data:
                row = result.data[0]
                return {
                    "current_topic": row.get("current_topic"),
                    "related_topics": row.get("related_topics") or [],
                }
            return None
        except Exception as exc:
            logger.exception("Failed to fetch topic for session %s", session_id)
            raise RepositoryError("Failed to fetch topic") from exc

    def upsert_topic(self, session_id: str, current_topic: str, related_topics: list[str]) -> None:
        """Insert or update the topic row for this session."""
        try:
            self.supabase.table("session_topics").upsert(
                {
                    "session_id": session_id,
                    "current_topic": current_topic,
                    "related_topics": related_topics,
                    "updated_at": "now()",
                },
                on_conflict="session_id",
            ).execute()
        except Exception as exc:
            logger.exception("Failed to upsert topic for session %s", session_id)
            raise RepositoryError("Failed to upsert topic") from exc

    # ------------------------------------------------------------------
    # conversation_summaries
    # ------------------------------------------------------------------

    def get_summary(self, session_id: str) -> str | None:
        """Return the stored study summary string or None."""
        try:
            result = (
                self.supabase.table("conversation_summaries")
                .select("summary")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0].get("summary")
            return None
        except Exception as exc:
            logger.exception("Failed to fetch summary for session %s", session_id)
            raise RepositoryError("Failed to fetch summary") from exc

    def upsert_summary(self, session_id: str, summary: str) -> None:
        """Insert or update the study summary for this session."""
        try:
            self.supabase.table("conversation_summaries").upsert(
                {
                    "session_id": session_id,
                    "summary": summary,
                    "updated_at": "now()",
                },
                on_conflict="session_id",
            ).execute()
        except Exception as exc:
            logger.exception("Failed to upsert summary for session %s", session_id)
            raise RepositoryError("Failed to upsert summary") from exc
