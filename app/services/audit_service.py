from __future__ import annotations

"""
audit_service.py
~~~~~~~~~~~~~~~~
Fire-and-forget audit logger.

All writes to the ``audit_log`` Supabase table happen in daemon threads so
they never block the HTTP response.  Any exception is swallowed and only
logged — audit logging must never break authentication or chat flows.
"""

import logging

logger = logging.getLogger(__name__)

# Recognised event types — add new ones here as needed.
EVENT_LOGIN = "login"
EVENT_LOGIN_FAILED = "login_failed"
EVENT_SIGNUP = "signup"
EVENT_LOGOUT = "logout"
EVENT_PASSWORD_RESET_REQUEST = "password_reset_request"
EVENT_PASSWORD_CHANGED = "password_changed"
EVENT_RATE_LIMITED = "rate_limited"


class AuditService:
    """
    Persist security-relevant events to the ``audit_log`` table.

    Degrades gracefully to a no-op when:
    - ``supabase_client`` is ``None`` (persistence disabled).
    - The database write fails for any reason.
    """

    def __init__(self, supabase_client) -> None:
        self._supabase = supabase_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        *,
        user_id: str | None = None,
        remote_addr: str | None = None,
        details: dict | None = None,
    ) -> None:
        """
        Queue an audit event for async persistence.

        This method returns immediately; the write happens in a daemon thread.

        :param event_type:  One of the ``EVENT_*`` constants in this module.
        :param user_id:     Authenticated user ID, or ``None`` for guests.
        :param remote_addr: Client IP address.
        :param details:     Optional dict of additional context (e.g. email, endpoint).
        """
        if self._supabase is None:
            return  # persistence disabled — no-op

        try:
            from app.tasks.audit_tasks import log_audit_event_task
            log_audit_event_task.delay(event_type, user_id, remote_addr, details or {})
        except Exception as exc:
            logger.warning("AuditService: failed to queue audit event: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(
        self,
        event_type: str,
        user_id: str | None,
        remote_addr: str | None,
        details: dict,
    ) -> None:
        try:
            self._supabase.table("audit_log").insert(
                {
                    "event_type": event_type,
                    "user_id": user_id,
                    "remote_addr": remote_addr,
                    "details": details,
                }
            ).execute()
        except Exception:
            # Intentionally swallowed — audit must never crash the app.
            logger.warning(
                "AuditService: failed to write event=%s user_id=%s",
                event_type,
                user_id,
            )
