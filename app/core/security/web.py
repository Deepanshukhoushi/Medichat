from __future__ import annotations

import logging

from flask import g, request

from app.core.config.settings import AppSettings
from app.core.security.csrf import CsrfManager
from app.core.security.rate_limiter import RateLimiter
from app.core.security.session_manager import SessionManager
from app.core.security.exceptions import AppError


logger = logging.getLogger(__name__)

# Endpoints that require POST rate-limiting.
_RATE_LIMITED_POST_ENDPOINTS = {
    "chat",
    "chat_stream",
    "signup",
    "login",
    "reset_password_request",
    "reset_password",
    "upload_document",
    "generate_flashcard_deck",
    "generate_quiz",
    "explain_topic",
    "summarize_text",
    "generate_mnemonics",
}

_POST_RATE_LIMIT_CONFIG = {
    "chat": ("chat_rate_limit", "chat_rate_window_seconds"),
    "chat_stream": ("chat_rate_limit", "chat_rate_window_seconds"),
    "signup": ("login_rate_limit", "login_rate_window_seconds"),
    "login": ("login_rate_limit", "login_rate_window_seconds"),
    "reset_password_request": ("login_rate_limit", "login_rate_window_seconds"),
    "reset_password": ("login_rate_limit", "login_rate_window_seconds"),
    "upload_document": ("document_upload_rate_limit", "document_upload_rate_window_seconds"),
    "generate_flashcard_deck": ("content_generation_rate_limit", "content_generation_rate_window_seconds"),
    "generate_quiz": ("content_generation_rate_limit", "content_generation_rate_window_seconds"),
    "explain_topic": ("content_generation_rate_limit", "content_generation_rate_window_seconds"),
    "summarize_text": ("content_generation_rate_limit", "content_generation_rate_window_seconds"),
    "generate_mnemonics": ("content_generation_rate_limit", "content_generation_rate_window_seconds"),
}

# New memory API endpoints that need GET rate-limiting (Issue #1)
_MEMORY_API_ENDPOINTS = {"get_topic_memory", "get_session_summary"}

# Security headers applied to every response (Issue #5)
_STATIC_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def _build_csp(frontend_origins: tuple[str, ...]) -> str:
    """Build a tight Content-Security-Policy header value."""
    connect_src_extra = " ".join(frontend_origins)
    connect_src = f"'self' {connect_src_extra}".strip()
    return (
        "default-src 'self' data: blob:; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        f"connect-src {connect_src}; "
        "font-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )


def register_security_guards(
    application, settings: AppSettings, auth_service, session_manager: SessionManager
) -> None:
    csrf_manager = CsrfManager(settings.csrf_cookie_name, settings.csrf_header_name)
    rate_limiter = RateLimiter(settings.redis_url, strict_mode=settings.require_redis)
    csp_value = _build_csp(settings.frontend_origins) if settings.csp_enabled else None

    @application.before_request
    def _security_context() -> None:
        cookie_value = request.cookies.get(settings.session_cookie_name)
        session_context = session_manager.resolve(cookie_value)

        if not settings.persistence_enabled and session_context.is_authenticated:
            session_context = session_manager.create_guest_cookie()
        elif (
            session_context.is_authenticated
            and auth_service is not None
            and settings.persistence_enabled
        ):
            if not auth_service.verify_session(session_context.access_token):
                logger.warning(
                    "Invalid auth session detected user_id=%s", session_context.user_id
                )
                session_context = session_manager.create_guest_cookie()

        g.session_context = session_context

        csrf_cookie = request.cookies.get(settings.csrf_cookie_name)
        g.csrf_token = csrf_cookie if csrf_cookie else csrf_manager.generate()

        endpoint = request.endpoint

        # CSRF + POST rate-limiting
        if request.method == "POST" and endpoint in _RATE_LIMITED_POST_ENDPOINTS:
            header_token = request.headers.get(settings.csrf_header_name)
            if not csrf_manager.is_valid(csrf_cookie, header_token):
                logger.warning(
                    "CSRF validation failed endpoint=%s remote_addr=%s",
                    endpoint,
                    request.remote_addr,
                )
                raise AppError("CSRF validation failed", status_code=403, error_type="csrf_error")

            if endpoint in ("login", "signup", "reset_password", "reset_password_request"):
                rate_key = f"rl:{endpoint}:{request.remote_addr}"
            else:
                rate_key = f"rl:{endpoint}:{request.remote_addr}:{session_context.user_id}"

            rate_limit_attr, rate_window_attr = _POST_RATE_LIMIT_CONFIG.get(
                endpoint,
                ("login_rate_limit", "login_rate_window_seconds"),
            )
            rate_limit = getattr(settings, rate_limit_attr)
            rate_window = getattr(settings, rate_window_attr)
            if not rate_limiter.check(rate_key, rate_limit, rate_window):
                logger.warning(
                    "Rate limit exceeded endpoint=%s user_id=%s remote_addr=%s",
                    endpoint,
                    session_context.user_id,
                    request.remote_addr,
                )
                raise AppError("Too many requests", status_code=429, error_type="rate_limited")

        # Memory API GET rate-limiting (Issue #1)
        if request.method == "GET" and endpoint in _MEMORY_API_ENDPOINTS:
            rate_key = f"rl:{endpoint}:{request.remote_addr}:{session_context.user_id}"
            if not rate_limiter.check(
                rate_key,
                settings.memory_api_rate_limit,
                settings.memory_api_rate_window_seconds,
            ):
                logger.warning(
                    "Memory API rate limit exceeded endpoint=%s user_id=%s remote_addr=%s",
                    endpoint,
                    session_context.user_id,
                    request.remote_addr,
                )
                raise AppError("Too many requests", status_code=429, error_type="rate_limited")

    @application.after_request
    def _set_security_response(response):
        session_context = getattr(g, "session_context", None)

        # Session cookie - use shorter TTL for guests or normal sessions
        if session_context and request.cookies.get(settings.session_cookie_name) != session_context.cookie_value:
            is_guest = session_context.kind == "guest"
            remember_me = getattr(g, "remember_me", False)
            if is_guest:
                session_max_age = settings.guest_session_max_age_seconds
            elif remember_me:
                session_max_age = settings.session_cookie_max_age_seconds
            else:
                session_max_age = None

            response.set_cookie(
                settings.session_cookie_name,
                session_context.cookie_value,
                max_age=session_max_age,
                httponly=True,
                secure=settings.secure_cookies,
                samesite="None" if settings.secure_cookies else "Lax",
            )

        # CSRF cookie
        if getattr(g, "csrf_token", None):
            response.set_cookie(
                settings.csrf_cookie_name,
                g.csrf_token,
                max_age=settings.session_cookie_max_age_seconds,
                httponly=False,
                secure=settings.secure_cookies,
                samesite="None" if settings.secure_cookies else "Lax",
            )

        # CORS
        origin = request.headers.get("Origin")
        host_origin = request.host_url.rstrip("/")
        if origin and origin != host_origin and origin in settings.frontend_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CSRF-Token"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            response.headers.setdefault("Vary", "Origin")

        # Security headers
        for header, value in _STATIC_SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if csp_value:
            response.headers.setdefault("Content-Security-Policy", csp_value)

        if request.is_secure:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

        response.headers.pop("Server", None)
        return response
