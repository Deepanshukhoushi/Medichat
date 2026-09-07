import logging
from celery import shared_task
from app.celery_app import celery_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix #20 — worker-level singleton
# ---------------------------------------------------------------------------
# ServiceFactory (and the Supabase/Pinecone clients it wraps) is expensive to
# construct.  Previously, each task invocation built a brand-new stack.
# We initialize ONE factory per worker process via the worker_process_init
# signal and store it as a module-level singleton that all tasks reuse.
_factory = None


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def _init_worker_task():
    pass  # placeholder


def _register_worker_init():
    """Register the worker_process_init signal lazily to avoid import-time collision
    with app/celery_app.py shadowing the 'celery' package name."""
    try:
        from celery.signals import worker_process_init as wpi

        @wpi.connect
        def _handler(**kwargs):
            _init_worker()
    except Exception:
        pass


_register_worker_init()


def _get_factory():
    """Return the singleton factory, constructing it on-demand if the signal never fired."""
    if _factory is not None:
        return _factory
    # Fallback for test environments or workers that don't emit the signal.
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
    return ServiceFactory(settings, session_manager)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def extract_topic_task(self, session_id: str, recent_messages: list[dict]):
    factory = _get_factory()
    topic_service = factory.create_topic_service()
    chat_service = factory.create_chat_service()

    if not topic_service or not chat_service:
        logger.warning("Topic service or chat service not initialized. Skipping topic extraction task.")
        return
    try:
        topic_service.extract_and_update_topic(session_id, recent_messages, chat_service.llm)
    except Exception as exc:
        logger.error(f"Failed topic extraction in celery for {session_id}", exc_info=True)
        raise self.retry(exc=exc)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def generate_summary_task(self, session_id: str, recent_messages: list[dict]):
    factory = _get_factory()
    summary_service = factory.create_summary_service()
    chat_service = factory.create_chat_service()

    if not summary_service or not chat_service:
        logger.warning("Summary service or chat service not initialized. Skipping summary generation task.")
        return
    try:
        summary_service.generate_and_save_summary(session_id, recent_messages, chat_service.llm)
    except Exception as exc:
        logger.error(f"Failed summary generation in celery for {session_id}", exc_info=True)
        raise self.retry(exc=exc)
