from unittest.mock import MagicMock, patch

from tests.stubs import setup_stubs
setup_stubs()

from app.tasks.memory_tasks import extract_topic_task, generate_summary_task

def test_extract_topic_task():
    with patch("app.services.factory.ServiceFactory") as mock_factory_cls:
        mock_factory = MagicMock()
        mock_factory_cls.return_value = mock_factory
        
        mock_topic_service = MagicMock()
        mock_chat_service = MagicMock()
        
        mock_factory.create_topic_service.return_value = mock_topic_service
        mock_factory.create_chat_service.return_value = mock_chat_service

        # Self mock for celery task
        mock_self = MagicMock()

        extract_topic_task(mock_self, "session_123", [{"role": "user", "content": "hi"}])
        
        mock_topic_service.extract_and_update_topic.assert_called_once_with(
            "session_123", [{"role": "user", "content": "hi"}], mock_chat_service.llm
        )

def test_generate_summary_task():
    with patch("app.services.factory.ServiceFactory") as mock_factory_cls:
        mock_factory = MagicMock()
        mock_factory_cls.return_value = mock_factory
        
        mock_summary_service = MagicMock()
        mock_chat_service = MagicMock()
        
        mock_factory.create_summary_service.return_value = mock_summary_service
        mock_factory.create_chat_service.return_value = mock_chat_service

        mock_self = MagicMock()

        generate_summary_task(mock_self, "session_123", [{"role": "user", "content": "hi"}])
        
        mock_summary_service.generate_and_save_summary.assert_called_once_with(
            "session_123", [{"role": "user", "content": "hi"}], mock_chat_service.llm
        )
