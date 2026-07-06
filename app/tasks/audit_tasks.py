from app.celery_app import celery_app
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def log_audit_event_task(self, event_type: str, user_id: str | None, remote_addr: str | None, details: dict):
    from app.core.config.settings import get_settings
    from app.services.factory import ServiceFactory
    from app.core.security.session_manager import SessionManager

    settings = get_settings()
    session_manager = SessionManager(
        secret_key=settings.flask_secret_key,
        cookie_name=settings.session_cookie_name,
        guest_prefix=settings.guest_session_prefix,
        max_age_seconds=settings.session_cookie_max_age_seconds,
    )
    factory = ServiceFactory(settings, session_manager)

    audit_service = factory.create_audit_service()
    if not audit_service:
        logger.warning("Audit service not initialized. Skipping audit log task.")
        return

    try:
        audit_service._write(event_type, user_id, remote_addr, details)
    except Exception as exc:
        logger.exception("Audit task failed")
        raise self.retry(exc=exc)
