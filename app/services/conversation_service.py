from __future__ import annotations


from app.core.config.settings import AppSettings
from app.repositories.chat_history_repository import ChatHistoryRepository
from app.repositories.conversation_repository import ConversationRepository


class ConversationService:
    def __init__(
        self,
        conversation_repository: ConversationRepository | None,
        chat_history_repository: ChatHistoryRepository | None,
        settings: AppSettings,
    ) -> None:
        self.conversation_repository = conversation_repository
        self.chat_history_repository = chat_history_repository

    def ensure_conversation(self, user_id: str) -> str:
        if self.conversation_repository is None:
            return user_id
        return self.conversation_repository.ensure_conversation(user_id)
