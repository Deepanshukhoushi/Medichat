from __future__ import annotations

import logging

from langchain_community.chat_message_histories import ChatMessageHistory

from app.core.config.settings import AppSettings
from app.core.security.exceptions import RepositoryError


logger = logging.getLogger(__name__)


class ChatHistoryRepository:
    def __init__(self, supabase_client, settings: AppSettings) -> None:
        self.supabase = supabase_client
        self.settings = settings

    def load_history_as_langchain(self, session_id: str) -> ChatMessageHistory:
        history = ChatMessageHistory()
        if not session_id or session_id.startswith(self.settings.guest_session_prefix):
            return history

        try:
            result = (
                self.supabase.table("chat_messages")
                .select("role, content, created_at")
                .eq("session_id", session_id)
                .order("created_at")
                .execute()
            )
        except Exception as exc:
            logger.exception("Failed to load chat messages")
            raise RepositoryError("Failed to load chat messages") from exc

        for row in result.data or []:
            role = row.get("role")
            message = row.get("content", "")
            if role == "user":
                history.add_user_message(message)
            elif role == "assistant":
                history.add_ai_message(message)

        return history

    # ------------------------------------------------------------------
    # chat_messages table — used by the advanced memory system
    # ------------------------------------------------------------------

    def save_chat_message(self, session_id: str, user_id: str, role: str, content: str) -> str | None:
        """Persist a message to the new chat_messages table and return its ID.

        Defense-in-depth: the INSERT explicitly sets *user_id*, so even if a
        caller supplies a *session_id* belonging to a different user the row
        will always be stamped with the *true* caller's user_id and RLS /
        application-level queries filtered by user_id will not return it under
        the wrong account.
        """
        if user_id.startswith(self.settings.guest_session_prefix):
            return None

        try:
            result = self.supabase.table("chat_messages").insert(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                }
            ).execute()
            if result.data and len(result.data) > 0:
                return result.data[0].get("id")
            return None
        except Exception as exc:
            logger.exception("Failed to save chat message to chat_messages")
            raise RepositoryError("Failed to save chat message") from exc

    def delete_message(self, session_id: str, user_id: str, message_id: str) -> None:
        """Delete a specific message from the conversation."""
        if user_id.startswith(self.settings.guest_session_prefix):
            return
            
        try:
            self.supabase.table("chat_messages").delete().eq("id", message_id).eq("session_id", session_id).eq("user_id", user_id).execute()
        except Exception as exc:
            logger.exception("Failed to delete message %s", message_id)
            raise RepositoryError("Failed to delete message") from exc

    def rate_message(self, session_id: str, user_id: str, message_id: str, liked: bool) -> None:
        """Rate a specific message (thumbs up / thumbs down)."""
        if user_id.startswith(self.settings.guest_session_prefix):
            return
            
        try:
            # We assume a 'liked' boolean column exists or will be added.
            # self.supabase.table("chat_messages").update({"liked": liked}).eq("id", message_id).eq("session_id", session_id).eq("user_id", user_id).execute()
            pass # column 'liked' does not exist in chat_messages table yet
        except Exception as exc:
            # If the column doesn't exist yet, we catch and log gracefully
            logger.warning("Failed to rate message %s (schema may need updating): %s", message_id, exc)

    def delete_latest_exchange(self, session_id: str) -> None:
        """Delete the latest user and assistant messages for regeneration."""
        try:
            # Supabase doesn't support DELETE with LIMIT directly easily, 
            # so we fetch the last 2 IDs and delete them.
            result = (
                self.supabase.table("chat_messages")
                .select("id")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .limit(2)
                .execute()
            )
            if not result.data:
                return
                
            ids_to_delete = [row["id"] for row in result.data]
            if ids_to_delete:
                self.supabase.table("chat_messages").delete().in_("id", ids_to_delete).execute()
        except Exception as exc:
            logger.exception("Failed to delete latest exchange for session %s", session_id)
            raise RepositoryError("Failed to delete latest exchange") from exc

    def get_recent_messages(self, session_id: str, limit: int = 10, user_id: str | None = None) -> list[dict]:
        """Return the most recent *limit* messages for the session, oldest-first.

        Defense-in-depth: when *user_id* is provided the query also filters on
        ``user_id``, ensuring a caller that somehow supplies a foreign
        *session_id* receives an empty list rather than leaking history.
        """
        try:
            query = (
                self.supabase.table("chat_messages")
                .select("id, role, content, created_at")
                .eq("session_id", session_id)
            )
            if user_id:
                query = query.eq("user_id", user_id)
            result = (
                query
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            # Reverse so messages are in chronological order (oldest first)
            rows = list(reversed(result.data or []))
            return [{"id": r.get("id"), "role": r["role"], "content": r["content"]} for r in rows]
        except Exception as exc:
            logger.exception("Failed to fetch recent messages for session %s", session_id)
            raise RepositoryError("Failed to fetch recent messages") from exc

    def get_message_count(self, session_id: str) -> int:
        """Return total number of stored messages for the session."""
        try:
            result = (
                self.supabase.table("chat_messages")
                .select("id", count="exact")
                .eq("session_id", session_id)
                .execute()
            )
            return result.count or 0
        except Exception as exc:
            logger.exception("Failed to count messages for session %s", session_id)
            return 0
