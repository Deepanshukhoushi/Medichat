from __future__ import annotations

import logging
import time

from flask import g, request


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("medichat")


def register_request_logging(application, logger: logging.Logger) -> None:
    @application.before_request
    def _start_timer() -> None:
        g.request_started_at = time.perf_counter()

    @application.after_request
    def _log_request(response):
        elapsed_ms = None
        started_at = getattr(g, "request_started_at", None)
        if started_at is not None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

        logger.info(
            "request method=%s path=%s status=%s duration_ms=%s remote_addr=%s",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
            request.headers.get("X-Forwarded-For", request.remote_addr),
        )
        return response

