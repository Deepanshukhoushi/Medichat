from langchain_community.chat_message_histories import ChatMessageHistory
from app.repositories.interfaces import (
    IConversationRepository,
    IChatHistoryRepository,
    IMemoryRepository,
    IUserRepository,
    IFlashcardRepository,
    IQuizRepository,
    IProfileRepository
)
from app.core.security.exceptions import AppError

class DummyConversationRepository(IConversationRepository):
    def ensure_conversation(self, user_id: str) -> str: return "dummy-conv"
    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]: return []
    def delete_conversation(self, conversation_id: str, user_id: str) -> None: pass
    def user_owns_conversation(self, conversation_id: str, user_id: str) -> bool: return conversation_id == user_id

class DummyChatHistoryRepository(IChatHistoryRepository):
    def load_history_as_langchain(self, session_id: str) -> ChatMessageHistory: return ChatMessageHistory()
    def save_chat_message(self, session_id: str, user_id: str, role: str, content: str) -> str | None: return "dummy-msg-id"
    def delete_message(self, session_id: str, user_id: str, message_id: str) -> None: pass
    def rate_message(self, session_id: str, user_id: str, message_id: str, liked: bool) -> None: pass
    def get_recent_messages(self, session_id: str, limit: int = 10, user_id: str | None = None) -> list[dict]: return []
    def get_message_count(self, session_id: str) -> int: return 0

class DummyMemoryRepository(IMemoryRepository):
    def get_topic(self, session_id: str) -> dict | None: return None
    def upsert_topic(self, session_id: str, current_topic: str, related_topics: list[str]) -> None: pass
    def get_summary(self, session_id: str) -> str | None: return None
    def upsert_summary(self, session_id: str, summary: str) -> None: pass

class DummyUserRepository(IUserRepository):
    def create_user(self, email: str, password_hash: str) -> dict:
        # Refuse to create users when persistence is disabled — a silent fake
        # signup discards all data and is worse than an honest error message.
        raise AppError(
            "Signups are temporarily unavailable: the database is not connected. "
            "Please try again later or contact support.",
            status_code=503,
            error_type="service_unavailable"
        )
    def get_user_by_email(self, email: str) -> dict | None: return None
    def update_password(self, user_id: str, password_hash: str) -> None: pass

class DummyFlashcardRepository(IFlashcardRepository):
    def list_decks(self, user_id: str, limit: int = 30) -> list[dict]: return []
    def save_deck(self, user_id: str, topic: str, cards: list[dict]) -> str:
        raise AppError("Persistence is disabled", status_code=503, error_type="service_unavailable")

class DummyQuizRepository(IQuizRepository):
    def list_sessions(self, user_id: str, limit: int = 30) -> list[dict]: return []
    def save_session(self, user_id: str, topic: str, questions: list[dict]) -> str:
        raise AppError("Persistence is disabled", status_code=503, error_type="service_unavailable")
    def submit_score(self, session_id: str, user_id: str, score: float) -> None: pass

class DummyProfileRepository(IProfileRepository):
    def get_profile(self, user_id: str) -> dict | None: return None
    def update_profile(self, user_id: str, display_name: str, medical_year: int, specialty: str, university: str) -> dict: return {}
