from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.core.security.exceptions import RepositoryError


logger = logging.getLogger(__name__)


class QuizRepository:
    def __init__(self, supabase_client) -> None:
        self.supabase = supabase_client

    def save_session(self, user_id: str, topic: str, questions: list[dict]) -> str:
        try:
            res = self.supabase.table("quiz_sessions").insert({
                "user_id": user_id,
                "topic": topic,
                "questions": questions
            }).execute()
            return res.data[0]["id"]
        except Exception as exc:
            logger.exception("Failed to save quiz session")
            raise RepositoryError("Failed to save quiz session") from exc

    def list_sessions(self, user_id: str) -> list[dict]:
        try:
            res = self.supabase.table("quiz_sessions").select("id, topic, score, completed_at, created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
            return res.data
        except Exception as exc:
            logger.exception("Failed to list quiz sessions")
            raise RepositoryError("Failed to list quiz sessions") from exc

    def get_session(self, session_id: str, user_id: str) -> dict:
        try:
            res = self.supabase.table("quiz_sessions").select("*").eq("id", session_id).eq("user_id", user_id).single().execute()
            return res.data
        except Exception as exc:
            logger.exception("Failed to get quiz session")
            raise RepositoryError("Failed to get quiz session") from exc

    def submit_score(self, session_id: str, user_id: str, score: int) -> None:
        try:
            now_str = datetime.now(timezone.utc).isoformat()
            self.supabase.table("quiz_sessions").update({
                "score": score,
                "completed_at": now_str
            }).eq("id", session_id).eq("user_id", user_id).execute()
        except Exception as exc:
            logger.exception("Failed to submit quiz score")
            raise RepositoryError("Failed to submit quiz score") from exc
