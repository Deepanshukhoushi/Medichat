from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self, supabase_client) -> None:
        self.supabase = supabase_client

    def get_study_stats(self, user_id: str) -> dict:
        """
        Aggregate user study stats from chat_messages, quiz_sessions, and flashcard_decks.
        Returns total_sessions, total_messages, unique_topics, and streak_days.
        """
        if not self.supabase:
            return {"total_sessions": 0, "total_messages": 0, "unique_topics": 0, "streak_days": 0}

        try:
            # Message count
            msg_res = self.supabase.table("chat_messages").select("id", count="exact").eq("user_id", user_id).execute()
            total_messages = msg_res.count if msg_res.count else 0

            # Session count (from conversations)
            conv_res = self.supabase.table("conversations").select("id", count="exact").eq("user_id", user_id).execute()
            total_sessions = conv_res.count if conv_res.count else 0

            # Unique topics — primary: session_topics table (populated by Celery topic extraction)
            # Fallback: count distinct conversation titles, which are set to the first user message
            # and act as a reliable proxy for topic diversity when Celery is not running.
            conv_ids = [c["id"] for c in conv_res.data] if conv_res.data else []
            if not conv_ids:
                unique_topics = 0
            else:
                topic_res = self.supabase.table("session_topics").select("current_topic").in_("session_id", conv_ids).execute()
                topics = {t["current_topic"] for t in topic_res.data if t.get("current_topic")}
                if topics:
                    unique_topics = len(topics)
                else:
                    # Fallback: derive topic count from conversation titles.
                    # Each conversation title is set to the first user message, making it a
                    # reliable proxy for unique topics studied.
                    title_res = self.supabase.table("conversations").select("title").eq("user_id", user_id).execute()
                    unique_titles = {
                        row["title"].strip().lower()
                        for row in (title_res.data or [])
                        if row.get("title") and len(row["title"].strip()) > 3
                    }
                    unique_topics = len(unique_titles)

            # Streak days
            activity_res = self.supabase.table("chat_messages").select("created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
            streak = self._calculate_streak(activity_res.data)

            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "unique_topics": unique_topics,
                "streak_days": streak,
            }
        except Exception as exc:
            logger.exception("Failed to get study stats")
            return {"total_sessions": 0, "total_messages": 0, "unique_topics": 0, "streak_days": 0}

    def _calculate_streak(self, activity_data: list[dict]) -> int:
        if not activity_data:
            return 0
        
        dates = set()
        for item in activity_data:
            try:
                # parse ISO 8601 string
                dt = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
                dates.add(dt.date())
            except Exception:
                pass
        
        if not dates:
            return 0
            
        sorted_dates = sorted(list(dates), reverse=True)
        streak = 0
        current_date = datetime.now(timezone.utc).date()
        
        # Streak allows for today or yesterday to be the start
        if sorted_dates[0] < current_date - timedelta(days=1):
            return 0
            
        check_date = sorted_dates[0]
        for d in sorted_dates:
            if d == check_date:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break
                
        return streak
