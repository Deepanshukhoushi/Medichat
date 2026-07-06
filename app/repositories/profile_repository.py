from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.core.security.exceptions import RepositoryError


logger = logging.getLogger(__name__)


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
        try:
            payload = {
                "user_id": user_id,
                "display_name": data.get("display_name"),
                "medical_year": data.get("medical_year"),
                "specialty": data.get("specialty"),
                "university": data.get("university"),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            res = self.supabase.table("user_profiles").upsert(payload).execute()
            return res.data[0]
        except Exception as exc:
            logger.exception("Failed to upsert profile")
            raise RepositoryError("Failed to upsert profile") from exc
