import os
from celery import Celery
from app.core.config.settings import get_settings

settings = get_settings()

# Use REDIS_URL for broker if available, else a local default for dev
broker_url = settings.redis_url if settings.redis_url else "redis://localhost:6379/0"

celery_app = Celery(
    "medichat_worker",
    broker=broker_url,
    backend=broker_url,
    include=["app.tasks.memory_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
)
