from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask

from app.api.controllers.chat_controller import ChatController
from app.api.routes.web import register_web_routes
from app.core.config.settings import AppSettings, get_settings
from app.core.logging.logger import configure_logging, register_request_logging
from app.core.security.error_handlers import register_error_handlers
from app.core.security.session_manager import SessionManager
from app.core.security.web import register_security_guards
from app.services.factory import ServiceFactory


@lru_cache(maxsize=1)
def _default_app() -> Flask:
    return create_app()


def create_app(settings: AppSettings | None = None) -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    template_folder = project_root / "frontend" / "templates"
    static_folder = project_root / "frontend" / "static"

    app_settings = settings or get_settings()
    logger = configure_logging()
    session_manager = SessionManager(
        secret_key=app_settings.flask_secret_key,
        cookie_name=app_settings.session_cookie_name,
        guest_prefix=app_settings.guest_session_prefix,
        max_age_seconds=app_settings.session_cookie_max_age_seconds,
    )
    factory = ServiceFactory(app_settings, session_manager)

    services = {
        "auth_service": factory.create_auth_service(),
        "chat_service": factory.create_chat_service(),
        "memory_service": factory.create_memory_service(),
        "topic_service": factory.create_topic_service(),
        "summary_service": factory.create_summary_service(),
        "audit_service": factory.create_audit_service(),
        "flashcard_service": factory.create_flashcard_service(),
        "quiz_service": factory.create_quiz_service(),
        "study_tools_service": factory.create_study_tools_service(),
        "analytics_service": factory.create_analytics_service(),
        "document_service": factory.create_document_service(),
        "conversation_repository": factory._get_conversation_repo(),
        "profile_repository": factory._get_profile_repo(),
    }

    application = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
        static_url_path="/static",
    )

    from werkzeug.middleware.proxy_fix import ProxyFix
    application.wsgi_app = ProxyFix(application.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    application.config.update(
        SECRET_KEY=app_settings.flask_secret_key,
        MAX_CONTENT_LENGTH=max(
            app_settings.max_content_length_bytes,
            app_settings.max_upload_size_bytes + 4096,
        ),
    )

    register_request_logging(application, logger)
    register_error_handlers(application, logger)
    register_security_guards(application, app_settings, services["auth_service"], session_manager)

    controller = ChatController(
        chat_service=services["chat_service"],
        auth_service=services["auth_service"],
        settings=app_settings,
        memory_service=services["memory_service"],
        topic_service=services["topic_service"],
        summary_service=services["summary_service"],
        audit_service=services["audit_service"],
        flashcard_service=services["flashcard_service"],
        quiz_service=services["quiz_service"],
        study_tools_service=services["study_tools_service"],
        analytics_service=services["analytics_service"],
        document_service=services["document_service"],
        conversation_repository=services["conversation_repository"],
        profile_repository=services["profile_repository"],
    )
    register_web_routes(application, controller)

    application.extensions["settings"] = app_settings
    application.extensions["session_manager"] = session_manager
    application.extensions["services"] = services
    return application


def get_app() -> Flask:
    return _default_app()


def get_answer(user_input: str, user_id: str, conversation_id: str | None = None) -> str:
    services = get_app().extensions["services"]
    return services["chat_service"].get_answer(user_input=user_input, user_id=user_id, conversation_id=conversation_id)


def __getattr__(name: str) -> Any:
    if name == "app":
        return get_app()
    raise AttributeError(name)
