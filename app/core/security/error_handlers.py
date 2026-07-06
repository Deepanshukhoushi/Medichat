from __future__ import annotations

from flask import jsonify

try:
    from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
except ImportError:  # pragma: no cover - compatibility with lightweight test stubs
    from werkzeug.exceptions import HTTPException

    RequestEntityTooLarge = None

from app.core.security.exceptions import AppError


def register_error_handlers(application, logger) -> None:
    if getattr(application, "_error_handlers_registered", False):
        return
    application._error_handlers_registered = True

    @application.errorhandler(AppError)
    def _handle_app_error(error: AppError):
        payload = error.to_dict()
        logger.warning("app_error type=%s message=%s", payload["type"], payload["error"])
        return jsonify(payload), error.status_code

    @application.errorhandler(Exception)
    def _handle_unexpected_error(error: Exception):
        if isinstance(error, HTTPException):
            if getattr(error, "code", None) == 413:
                logger.warning("request_too_large path=%s", getattr(error, "description", ""))
                return jsonify(
                    {"error": "Request payload is too large", "type": "payload_too_large", "status": 413}
                ), 413
            return error
        logger.exception("unexpected_error")
        return jsonify({"error": "Internal server error", "type": "internal_error"}), 500

    if RequestEntityTooLarge is not None:
        @application.errorhandler(RequestEntityTooLarge)
        def _handle_payload_too_large(error: RequestEntityTooLarge):
            logger.warning("request_too_large path=%s", getattr(error, "description", ""))
            return jsonify({"error": "Request payload is too large", "type": "payload_too_large", "status": 413}), 413
