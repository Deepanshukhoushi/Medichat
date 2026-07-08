from abc import ABC, abstractmethod
from typing import Any
from langchain_community.chat_message_histories import ChatMessageHistory

class IConversationRepository(ABC):
    @abstractmethod
    def ensure_conversation(self, user_id: str) -> str:
        pass

    @abstractmethod
    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        pass

    @abstractmethod
    def delete_conversation(self, conversation_id: str, user_id: str) -> None:
        pass

    @abstractmethod
    def user_owns_conversation(self, conversation_id: str, user_id: str) -> bool:
        pass

class IChatHistoryRepository(ABC):
    @abstractmethod
    def load_history_as_langchain(self, session_id: str) -> ChatMessageHistory:
        pass

    @abstractmethod
    def save_chat_message(self, session_id: str, user_id: str, role: str, content: str) -> str | None:
        """Persist a message and return its unique ID (if supported)."""
        pass

    @abstractmethod
    def delete_message(self, session_id: str, user_id: str, message_id: str) -> None:
        pass

    @abstractmethod
    def rate_message(self, session_id: str, user_id: str, message_id: str, liked: bool) -> None:
        pass

    @abstractmethod
    def get_recent_messages(self, session_id: str, limit: int = 10) -> list[dict]:
        pass

    @abstractmethod
    def get_message_count(self, session_id: str) -> int:
        pass

class IUserRepository(ABC):
    @abstractmethod
    def create_user(self, email: str, password_hash: str) -> dict:
        pass

    @abstractmethod
    def get_user_by_email(self, email: str) -> dict | None:
        pass

    @abstractmethod
    def update_password(self, user_id: str, password_hash: str) -> None:
        pass

class IMemoryRepository(ABC):
    @abstractmethod
    def get_topic(self, session_id: str) -> dict | None:
        pass

    @abstractmethod
    def upsert_topic(self, session_id: str, current_topic: str, related_topics: list[str]) -> None:
        pass

    @abstractmethod
    def get_summary(self, session_id: str) -> str | None:
        pass

    @abstractmethod
    def upsert_summary(self, session_id: str, summary: str) -> None:
        pass

class IFlashcardRepository(ABC):
    @abstractmethod
    def list_decks(self, user_id: str, limit: int = 30) -> list[dict]:
        pass

    @abstractmethod
    def save_deck(self, user_id: str, topic: str, cards: list[dict]) -> str:
        pass

class IQuizRepository(ABC):
    @abstractmethod
    def list_sessions(self, user_id: str, limit: int = 30) -> list[dict]:
        pass

    @abstractmethod
    def save_session(self, user_id: str, topic: str, questions: list[dict]) -> str:
        pass

    @abstractmethod
    def submit_score(self, session_id: str, user_id: str, score: float) -> None:
        pass

class IProfileRepository(ABC):
    @abstractmethod
    def get_profile(self, user_id: str) -> dict | None:
        pass

    @abstractmethod
    def update_profile(self, user_id: str, display_name: str, medical_year: int, specialty: str, university: str) -> dict:
        pass
