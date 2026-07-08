import logging
from app.core.config.settings import AppSettings
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
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.services.topic_service import TopicService
from app.services.summary_service import SummaryService
from app.services.flashcard_service import FlashcardService
from app.services.quiz_service import QuizService
from app.services.study_tools_service import StudyToolsService
from app.services.analytics_service import AnalyticsService
from app.services.document_service import DocumentService
from app.services.audit_service import AuditService
from app.core.security.session_manager import SessionManager

logger = logging.getLogger(__name__)

from app.services.dummy_repositories import (
    DummyConversationRepository,
    DummyChatHistoryRepository,
    DummyMemoryRepository,
    DummyUserRepository,
    DummyFlashcardRepository,
    DummyQuizRepository,
    DummyProfileRepository
)

class ServiceFactory:
    """Creates fully initialized services, injecting dummy repos if persistence is disabled."""

    def __init__(self, settings: AppSettings, session_manager: SessionManager):
        self.settings = settings
        self.session_manager = session_manager
        self.persistence_enabled = settings.persistence_enabled and settings.supabase_url and settings.supabase_key
        
        if self.persistence_enabled:
            from supabase import create_client, ClientOptions
            import httpx
            # Disable HTTP/2 to prevent httpx RemoteProtocolError drops with Supabase
            options = ClientOptions(httpx_client=httpx.Client(http2=False))
            self.supabase = create_client(settings.supabase_url, settings.supabase_key, options=options)
        else:
            self.supabase = None
            logger.error(
                "PERSISTENCE IS DISABLED — database is not connected. "
                "Signups will be refused. Chat messages and all user data will NOT be saved. "
                "Set SUPABASE_URL and SUPABASE_KEY to enable persistence."
            )

    def _get_conversation_repo(self) -> IConversationRepository:
        if self.persistence_enabled:
            from app.repositories.conversation_repository import ConversationRepository
            return ConversationRepository(self.supabase, self.settings)
        return DummyConversationRepository()

    def _get_chat_history_repo(self) -> IChatHistoryRepository:
        if self.persistence_enabled:
            from app.repositories.chat_history_repository import ChatHistoryRepository
            return ChatHistoryRepository(self.supabase, self.settings)
        return DummyChatHistoryRepository()

    def _get_memory_repo(self) -> IMemoryRepository:
        if self.persistence_enabled:
            from app.repositories.memory_repository import MemoryRepository
            return MemoryRepository(self.supabase)
        return DummyMemoryRepository()

    def _get_user_repo(self) -> IUserRepository:
        if self.persistence_enabled:
            from app.repositories.user_repository import UserRepository
            return UserRepository(self.supabase, self.settings)
        return DummyUserRepository()

    def _get_flashcard_repo(self) -> IFlashcardRepository:
        if self.persistence_enabled:
            from app.repositories.flashcard_repository import FlashcardRepository
            return FlashcardRepository(self.supabase)
        return DummyFlashcardRepository()

    def _get_quiz_repo(self) -> IQuizRepository:
        if self.persistence_enabled:
            from app.repositories.quiz_repository import QuizRepository
            return QuizRepository(self.supabase)
        return DummyQuizRepository()

    def _get_profile_repo(self) -> IProfileRepository:
        if self.persistence_enabled:
            from app.repositories.profile_repository import ProfileRepository
            return ProfileRepository(self.supabase)
        return DummyProfileRepository()

    def create_auth_service(self) -> AuthService:
        return AuthService(user_repository=self._get_user_repo(), session_manager=self.session_manager)

    def create_memory_service(self) -> MemoryService:
        return MemoryService(chat_history_repository=self._get_chat_history_repo(), context_window_size=self.settings.context_window_size)

    def create_topic_service(self) -> TopicService:
        return TopicService(memory_repository=self._get_memory_repo())

    def create_summary_service(self) -> SummaryService:
        return SummaryService(memory_repository=self._get_memory_repo(), trigger_count=self.settings.summary_trigger_count)

    def create_audit_service(self) -> AuditService | None:
        if self.persistence_enabled:
            return AuditService(self.supabase)
        return None

    def create_flashcard_service(self) -> FlashcardService:
        return FlashcardService(self._get_flashcard_repo())

    def create_quiz_service(self) -> QuizService:
        return QuizService(self._get_quiz_repo())

    def create_study_tools_service(self) -> StudyToolsService:
        return StudyToolsService()

    def create_analytics_service(self) -> AnalyticsService | None:
        if self.persistence_enabled:
            return AnalyticsService(self.supabase)
        return None

    def create_document_service(self) -> DocumentService:
        return DocumentService()

    def create_chat_service(self) -> ChatService:
        return ChatService(
            settings=self.settings,
            conversation_repository=self._get_conversation_repo(),
            chat_history_repository=self._get_chat_history_repo(),
            memory_service=self.create_memory_service(),
            topic_service=self.create_topic_service(),
            summary_service=self.create_summary_service(),
        )
