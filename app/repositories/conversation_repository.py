from __future__ import annotations

import logging
from uuid import uuid4

from app.core.config.settings import AppSettings
from app.core.security.exceptions import RepositoryError


logger = logging.getLogger(__name__)


class ConversationRepository:
    def __init__(self, supabase_client, settings: AppSettings) -> None:
        self.supabase = supabase_client
        self.settings = settings

    def ensure_conversation(self, user_id: str) -> str:
        """Always create a brand-new conversation row for this user."""
        if user_id.startswith(self.settings.guest_session_prefix):
            return user_id

        try:
            created = self.supabase.table("conversations").insert(
                {"user_id": user_id, "title": "New Conversation"}
            ).execute()
            return created.data[0]["id"]
        except Exception as exc:
            logger.exception("Failed to create conversation")
            raise RepositoryError("Failed to ensure conversation") from exc

    def update_title(self, conversation_id: str, title: str) -> None:
        """Set a human-readable title derived from the first user message."""
        try:
            self.supabase.table("conversations").update({"title": title}).eq("id", conversation_id).execute()
        except Exception as exc:
            logger.warning("Failed to update conversation title: %s", exc)

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        try:
            result = (
                self.supabase.table("conversations")
                .select("id, title, created_at")
                .eq("user_id", user_id)
                .is_("deleted_at", "null")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data
        except Exception as exc:
            logger.exception("Failed to list conversations")
            raise RepositoryError("Failed to list conversations") from exc

    def delete_conversation(self, conversation_id: str, user_id: str) -> None:
        try:
            self.supabase.table("conversations").update({"deleted_at": "now()"}).eq("id", conversation_id).eq("user_id", user_id).execute()
        except Exception as exc:
            logger.exception("Failed to delete conversation")
            raise RepositoryError("Failed to delete conversation") from exc

    def user_owns_conversation(self, conversation_id: str, user_id: str) -> bool:
        if user_id.startswith(self.settings.guest_session_prefix):
            return conversation_id == user_id

        try:
            result = (
                self.supabase.table("conversations")
                .select("id")
                .eq("id", conversation_id)
                .eq("user_id", user_id)
                .is_("deleted_at", "null")
                .limit(1)
                .execute()
            )
            return bool(result.data)
        except Exception as exc:
            logger.exception("Failed to verify conversation ownership")
            raise RepositoryError("Failed to verify conversation ownership") from exc
