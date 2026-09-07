from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError, field_validator

from app.core.security.exceptions import RepositoryError, ValidationError


logger = logging.getLogger(__name__)


class ProfileUpdate(BaseModel):
    """Validates and sanitizes inbound profile data before any DB write."""

    display_name: str | None = Field(default=None, max_length=100)
    medical_year: int | None = Field(default=None, ge=1, le=10)
    specialty: str | None = Field(default=None, max_length=100)
    university: str | None = Field(default=None, max_length=200)

    @field_validator("display_name", "specialty", "university", mode="before")
    @classmethod
    def _strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v


class ProfileRepository:
    def __init__(self, supabase_client) -> None:
        self.supabase = supabase_client

    def get_profile(self, user_id: str) -> dict:
        if user_id.startswith("guest_"):
            return {"user_id": user_id, "display_name": "", "medical_year": None, "specialty": "", "university": ""}

        try:
            res = self.supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
            if not res.data:
                # Return empty default profile
                display_name = ""
                try:
                    auth_user = self.supabase.auth.admin.get_user_by_id(user_id)
                    meta = getattr(auth_user.user, "user_metadata", {}) or {}
                    email = getattr(auth_user.user, "email", "") or ""
                    email_prefix = email.split("@")[0] if email else ""
                    display_name = meta.get("display_name") or meta.get("full_name") or meta.get("name") or email_prefix or ""
                except Exception:
                    pass
                return {"user_id": user_id, "display_name": display_name, "medical_year": None, "specialty": "", "university": ""}
            return res.data[0]
        except Exception as exc:
            logger.exception("Failed to get profile")
            raise RepositoryError("Failed to get profile") from exc

    def upsert_profile(self, user_id: str, data: dict) -> dict:
        # Fix #16: validate and sanitize before writing to DB.
        try:
            validated = ProfileUpdate.model_validate(data)
        except PydanticValidationError as exc:
            raise ValidationError("Invalid profile data", details=exc.errors()) from exc

        try:
            payload = {
                "user_id": user_id,
                "display_name": validated.display_name,
                "medical_year": validated.medical_year,
                "specialty": validated.specialty,
                "university": validated.university,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            res = self.supabase.table("user_profiles").upsert(payload).execute()
            return res.data[0]
        except ValidationError:
            raise
        except Exception as exc:
            logger.exception("Failed to upsert profile")
            raise RepositoryError("Failed to upsert profile") from exc
