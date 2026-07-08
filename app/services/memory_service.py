from __future__ import annotations

import logging

from app.core.security.exceptions import RepositoryError
from app.repositories.chat_history_repository import ChatHistoryRepository


logger = logging.getLogger(__name__)


class MemoryService:
    """
    Saves and retrieves messages from the ``chat_messages`` table.

    This is distinct from the legacy ``chat_history`` table: ``chat_messages``
    uses a ``content`` column (spec-compliant) and is the source of truth for
    the memory-aware context window.
    """

    def __init__(
        self,
        chat_history_repository: ChatHistoryRepository,
        context_window_size: int = 10,
    ) -> None:
        self._repo = chat_history_repository
        self.context_window_size = context_window_size

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_message(self, session_id: str, user_id: str, role: str, content: str) -> str | None:
        """Persist a single message to the chat_messages table (no-op for guests)."""
        if not session_id or not content:
            return None
        try:
            return self._repo.save_chat_message(session_id, user_id, role, content)
        except RepositoryError:
            logger.error("MemoryService: could not save message for session %s", session_id, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        """
        Return the most recent N messages for the session in chronological order.

        Each item is a dict ``{"role": "user"|"assistant", "content": "..."}``.
        Returns an empty list for guest sessions or on any error.
        """
        if not session_id:
            return []
        effective_limit = limit or self.context_window_size
        try:
            return self._repo.get_recent_messages(session_id, effective_limit)
        except RepositoryError:
            logger.error("MemoryService: could not fetch history for session %s", session_id, exc_info=True)
            return []

    def get_message_count(self, session_id: str) -> int:
        """Return total saved message count for the session."""
        if not session_id:
            return 0
        return self._repo.get_message_count(session_id)
