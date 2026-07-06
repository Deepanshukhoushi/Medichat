from __future__ import annotations


class AppError(Exception):
    status_code = 500
    error_type = "app_error"

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_type: str | None = None,
        details: dict | list | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type

    def to_dict(self) -> dict[str, str | int]:
        payload: dict[str, str | int | dict | list] = {
            "error": self.message,
            "type": self.error_type,
            "status": self.status_code,
        }
        if self.details is not None:
            payload["details"] = self.details
        return payload


class ConfigurationError(AppError):
    status_code = 500
    error_type = "configuration_error"


class RepositoryError(AppError):
    status_code = 502
    error_type = "repository_error"


class ServiceError(AppError):
    status_code = 500
    error_type = "service_error"


class AuthenticationError(AppError):
    status_code = 401
    error_type = "authentication_error"


class ValidationError(AppError):
    status_code = 400
    error_type = "validation_error"


class PersistenceUnavailableError(AppError):
    status_code = 503
    error_type = "persistence_unavailable"
