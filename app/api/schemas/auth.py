from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError, field_validator

from app.core.security.exceptions import ValidationError


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3)
    password: str = Field(min_length=8)
    display_name: str | None = Field(default=None, min_length=1)
    remember_me: bool = Field(default=False)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not EMAIL_PATTERN.match(normalized):
            raise ValueError("invalid email format")
        return normalized

    @classmethod
    def from_request_payload(cls, payload: dict | None) -> "AuthRequest":
        if not isinstance(payload, dict):
            raise ValidationError("Request body must be a JSON object", details={"payload": "object expected"})

        try:
            return cls.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError("Invalid auth payload", details=exc.errors()) from exc
