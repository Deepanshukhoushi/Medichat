"""
Script to hard-delete guest sessions older than 7 days.
Run this as a cron job to enforce the data retention policy.
"""
import logging
from datetime import datetime, timedelta, timezone
from supabase import create_client
from app.core.config.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_cleanup():
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        logger.error("Supabase credentials not configured. Exiting.")
        return

    supabase = create_client(settings.supabase_url, settings.supabase_key)
    
    # 7 days ago
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    
    logger.info(f"Cleaning up guest sessions older than {cutoff_date}...")
    
    try:
        # Conversations (cascade deletes typically handled by DB, but we explicitly delete here)
        result = supabase.table("conversations")\
            .delete()\
            .like("user_id", f"{settings.guest_session_prefix}%")\
            .lt("created_at", cutoff_date)\
            .execute()
        
        deleted_count = len(result.data) if result.data else 0
        logger.info(f"Deleted {deleted_count} stale guest conversations.")

        # Note: Depending on foreign key cascades, deleting the conversation might automatically 
        # delete chat_messages, memory summaries, etc. 
        # But if chat_messages is directly linked to user_id, we can also delete them:
        msg_result = supabase.table("chat_messages")\
            .delete()\
            .like("user_id", f"{settings.guest_session_prefix}%")\
            .lt("created_at", cutoff_date)\
            .execute()
            
        msg_deleted_count = len(msg_result.data) if msg_result.data else 0
        logger.info(f"Deleted {msg_deleted_count} stale guest chat messages.")

    except Exception as e:
        logger.exception("Error occurred during guest session cleanup")

if __name__ == "__main__":
    run_cleanup()
