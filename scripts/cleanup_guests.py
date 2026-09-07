"""
Script to hard-delete guest sessions older than 7 days.
Run this as a cron job to enforce the data retention policy.

NOTE (Issue #31): Under the current architecture this script is intentionally a no-op.
Guest sessions (user_id starting with settings.guest_session_prefix) are NEVER persisted
to Supabase:
  - ConversationRepository.ensure_conversation() returns early for guest IDs without
    inserting a row.
  - ChatHistoryRepository.save_chat_message() and MemoryService.save_message() also
    no-op for guest IDs.

The guest data retention policy is therefore enforced at write-time (by simply not
writing), not at deletion-time. Running this script against production is safe — it will
connect, execute two DELETE queries, find zero matching rows, and report 0 deletions.

This script is retained so that any cron job referencing it is not broken, and to serve
as a starting point if the architecture ever changes to persist-then-expire guest data.
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
